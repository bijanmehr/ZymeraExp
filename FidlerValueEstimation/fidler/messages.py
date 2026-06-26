"""The message-design space: one configurable message-passing round.

`aggregate(z, adj, positions, *, op, content, params, key, dropedge)` runs ONE round of
message passing and returns per-node aggregated messages `(N, hidden)`.

Two orthogonal axes:
  * **content** — what each neighbor `j` contributes *before* aggregation:
      - `value`   : `z_j`                         (no params)
      - `learned` : `MLP(z_j)`                    (node-only message MLP)
      - `geom`    : `MLP([z_j, edge_feat_ij])`    (edge-dependent; edge_feat = [dx,dy,dist]/comm_r)
  * **op** — how node `i` combines its incoming messages:
      - `mean`               : (A_hat @ m)/deg                      (size-invariant)
      - `gcn`                : D^-1/2 A_hat D^-1/2 @ m               (size-invariant)
      - `max`                : masked elementwise max over neighbors (size-invariant)
      - `sum`                : A_hat @ m            **ABLATION** — magnitude scales with N/deg
      - `attention`          : single-head masked-softmax(q.k/sqrt d) weighted v (size-invariant)
      - `multihead_attention`: 4 heads of the above                 (size-invariant)
      - `gated`              : per-edge learned gate sigma(MLP([z_i,z_j])) -> normalized weighted mean
      - `laplacian`          : -(L_hat @ m) spectral propagation, L_hat = I - D^-1/2 A_hat D^-1/2

A_hat strips self-loops; degrees use max(deg, 1). All learnable pieces live in a small
`MessageParams` eqx.Module so the SAME params instance runs at any N (size-invariance) —
the only deliberate exception is `op="sum"`.

DropEdge: with probability `dropedge`, comm edges are zeroed at random (training only;
applied symmetrically, self-loops never matter since A_hat strips them).
"""
import equinox as eqx
import jax
import jax.numpy as jnp

NEG_INF = -1e9


class MessageParams(eqx.Module):
    """Holds every learnable message-passing sub-module (uniform across op/content).

    All sub-modules are always constructed (they are tiny); which ones actually get
    used depends on `op`/`content`. Keeping the module uniform avoids `None` leaves in
    the pytree and makes the same instance valid for any aggregation op.
    """
    content_mlp: eqx.nn.MLP            # learned/geom message encoder (value -> identity, unused)
    q_proj: eqx.nn.Linear             # attention query
    k_proj: eqx.nn.Linear             # attention key
    v_proj: eqx.nn.Linear             # attention value
    gate_mlp: eqx.nn.MLP              # gated: sigma(MLP([z_i, z_j])) -> scalar per edge
    hidden: int = eqx.field(static=True)
    heads: int = eqx.field(static=True)
    content: str = eqx.field(static=True)
    op: str = eqx.field(static=True)
    comm_r: float = eqx.field(static=True)

    def __init__(self, hidden=64, heads=4, content="value", op="mean", *, key, comm_r=5.0):
        kc, kq, kk, kv, kg = jax.random.split(key, 5)
        self.hidden = int(hidden)
        self.heads = int(heads)
        self.content = content
        self.op = op
        self.comm_r = float(comm_r)
        # content message MLP: geom takes [z_j, dx, dy, dist] (hidden+3), else hidden.
        in_msg = hidden + 3 if content == "geom" else hidden
        self.content_mlp = eqx.nn.MLP(in_msg, hidden, width_size=hidden, depth=1, key=kc)
        self.q_proj = eqx.nn.Linear(hidden, hidden, key=kq)
        self.k_proj = eqx.nn.Linear(hidden, hidden, key=kk)
        self.v_proj = eqx.nn.Linear(hidden, hidden, key=kv)
        self.gate_mlp = eqx.nn.MLP(2 * hidden, 1, width_size=hidden, depth=1, key=kg)


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def _a_hat(adj):
    """bool/float (N,N) adjacency -> float (N,N) with self-loops stripped."""
    a = adj.astype(jnp.float32)
    return a - jnp.diag(jnp.diag(a))


def _edge_feats(positions, comm_r):
    """positions (N,2) -> (N,N,3) edge features [dx, dy, dist]/comm_r for pair (i,j) = i - j."""
    pos = positions.astype(jnp.float32)
    diff = pos[:, None, :] - pos[None, :, :]           # (N,N,2)  i - j
    dist = jnp.sqrt((diff ** 2).sum(-1, keepdims=True) + 1e-12)  # (N,N,1)
    feats = jnp.concatenate([diff, dist], axis=-1) / float(comm_r)
    return feats                                        # (N,N,3)


def _edge_messages(z, positions, params):
    """Per-edge message tensor E[i,j,:] = message FROM j TO i, shape (N, N, hidden).

    * value   : E[i,j] = z_j                              (broadcast over i)
    * learned : E[i,j] = MLP(z_j)                         (broadcast over i)
    * geom    : E[i,j] = MLP([z_j, edge_feat_ij])         (depends on both i and j)
    """
    N = z.shape[0]
    content = params.content
    if content == "value":
        m_node = z                                       # (N, hidden)
        return jnp.broadcast_to(m_node[None, :, :], (N, N, params.hidden))
    if content == "learned":
        m_node = jax.vmap(params.content_mlp)(z)         # (N, hidden)
        return jnp.broadcast_to(m_node[None, :, :], (N, N, params.hidden))
    if content == "geom":
        ef = _edge_feats(positions, params.comm_r)       # (N,N,3)
        zj = jnp.broadcast_to(z[None, :, :], (N, N, params.hidden))   # z_j over i
        inp = jnp.concatenate([zj, ef], axis=-1)         # (N,N,hidden+3)
        flat = inp.reshape(N * N, -1)
        out = jax.vmap(params.content_mlp)(flat).reshape(N, N, params.hidden)
        return out
    raise ValueError(f"unknown content {content!r}")


def attention_weights(z, adj, params):
    """Single-head dot-product attention weights over in-neighbors -> (N, N) row-normalized.

    alpha[i,j] = softmax_j( q_i . k_j / sqrt(d) ) masked to A_hat[i,j]; rows sum to 1
    (isolated nodes get a uniform-zero row). Used by `op in {attention}` and exposed for tests.
    """
    a = _a_hat(adj)
    q = jax.vmap(params.q_proj)(z)                       # (N, hidden)
    k = jax.vmap(params.k_proj)(z)                       # (N, hidden)
    d = z.shape[-1]
    scores = (q @ k.T) / jnp.sqrt(jnp.asarray(d, jnp.float32))   # (N,N)
    mask = a > 0
    scores = jnp.where(mask, scores, NEG_INF)
    w = jax.nn.softmax(scores, axis=-1)                  # (N,N)
    # zero out rows with no neighbours (softmax over all -inf would be uniform/NaN-safe-handled)
    has = mask.any(-1, keepdims=True)
    w = jnp.where(mask, w, 0.0)
    w = jnp.where(has, w, 0.0)
    return w


def _multihead_attention_weights(z, adj, params):
    """4-head attention weights -> (heads, N, N), each head's rows sum to 1 over neighbors."""
    a = _a_hat(adj)
    H = params.heads
    hidden = params.hidden
    hd = hidden // H
    q = jax.vmap(params.q_proj)(z).reshape(-1, H, hd)    # (N, H, hd)
    k = jax.vmap(params.k_proj)(z).reshape(-1, H, hd)    # (N, H, hd)
    # per head scores (H, N, N)
    scores = jnp.einsum("ihd,jhd->hij", q, k) / jnp.sqrt(jnp.asarray(hd, jnp.float32))
    mask = (a > 0)[None, :, :]                           # (1,N,N)
    scores = jnp.where(mask, scores, NEG_INF)
    w = jax.nn.softmax(scores, axis=-1)                  # (H,N,N)
    w = jnp.where(mask, w, 0.0)
    return w                                             # (H, N, N)


# --------------------------------------------------------------------------------------
# aggregation ops
# --------------------------------------------------------------------------------------
def _weighted_sum(E, W):
    """E (N,N,hidden), W (N,N) edge weights -> (N,hidden) = sum_j W[i,j] E[i,j]."""
    return jnp.einsum("ij,ijh->ih", W, E)


def aggregate(z, adj, positions, *, op, content, params, key, dropedge=0.0):
    """One message-passing round -> per-node aggregated messages (N, hidden).

    `params` is a `MessageParams` whose static `op`/`content` should match the args
    passed here (the args are the source of truth; `params` only supplies weights).
    DropEdge zeros random edges of A_hat with prob `dropedge` (symmetric, key-driven).
    """
    N = z.shape[0]
    a = _a_hat(adj)                                      # (N,N) float, no self-loops

    # ---- DropEdge (symmetric) ----------------------------------------------------------
    if dropedge and dropedge > 0.0:
        # upper-triangular Bernoulli keep-mask, mirrored -> symmetric edge dropping
        keep = jax.random.bernoulli(key, 1.0 - dropedge, (N, N)).astype(jnp.float32)
        keep = jnp.triu(keep, 1)
        keep = keep + keep.T
        a = a * keep

    deg = jnp.maximum(a.sum(-1, keepdims=True), 1.0)     # (N,1)

    E = _edge_messages(z, positions, params)             # (N,N,hidden) message j->i

    if op == "mean":
        W = a / deg                                      # row-normalized
        return _weighted_sum(E, W)

    if op == "sum":                                      # ABLATION (scales with N/degree)
        return _weighted_sum(E, a)

    if op == "gcn":
        d_inv_sqrt = 1.0 / jnp.sqrt(deg)                 # (N,1) (deg>=1)
        W = a * d_inv_sqrt * d_inv_sqrt.T                # D^-1/2 A D^-1/2
        return _weighted_sum(E, W)

    if op == "laplacian":
        # L_hat = I - D^-1/2 A D^-1/2 ; agg = -(L_hat @ m) = (gcn_agg) - m_self
        d_inv_sqrt = 1.0 / jnp.sqrt(deg)
        W = a * d_inv_sqrt * d_inv_sqrt.T
        gcn_agg = _weighted_sum(E, W)                    # (N,hidden)
        # self message m_i (diagonal of E): message from i to i with zero edge geometry
        m_self = jnp.diagonal(E, axis1=0, axis2=1).T     # (N, hidden)
        return gcn_agg - m_self

    if op == "max":
        # masked elementwise max over neighbors; isolated/absent edges -> -inf then ->0
        mask = (a > 0)[:, :, None]                       # (N,N,1)
        masked = jnp.where(mask, E, -jnp.inf)
        out = jnp.max(masked, axis=1)                    # (N,hidden)
        out = jnp.where(jnp.isfinite(out), out, 0.0)     # nodes with no neighbours -> 0
        return out

    if op == "gated":
        # per-edge gate g[i,j] = sigma(MLP([z_i, z_j])); normalized weighted mean over j.
        zi = jnp.broadcast_to(z[:, None, :], (N, N, params.hidden))
        zj = jnp.broadcast_to(z[None, :, :], (N, N, params.hidden))
        pair = jnp.concatenate([zi, zj], axis=-1).reshape(N * N, -1)
        g = jax.nn.sigmoid(jax.vmap(params.gate_mlp)(pair)).reshape(N, N)  # (N,N) in (0,1)
        g = g * a                                        # only real edges gate
        denom = jnp.maximum(g.sum(-1, keepdims=True), 1e-6)
        W = g / denom                                    # normalized -> size-invariant
        return _weighted_sum(E, W)

    if op == "attention":
        # single head: weight by attention, message values projected by v_proj.
        W = attention_weights(z, adj if not (dropedge and dropedge > 0.0) else (a > 0), params)
        # apply v-projection to the per-edge message tensor
        Ev = jax.vmap(lambda e: jax.vmap(params.v_proj)(e))(E)   # (N,N,hidden)
        return _weighted_sum(Ev, W)

    if op == "multihead_attention":
        Wh = _multihead_attention_weights(z, adj if not (dropedge and dropedge > 0.0) else (a > 0), params)
        # split v into heads, aggregate per head, concat back to hidden.
        H = params.heads
        hidden = params.hidden
        hd = hidden // H
        v = jax.vmap(lambda e: jax.vmap(params.v_proj)(e))(E)    # (N,N,hidden)
        v = v.reshape(N, N, H, hd)
        # out[i,h,:] = sum_j Wh[h,i,j] v[i,j,h,:]
        out = jnp.einsum("hij,ijhd->ihd", Wh, v)         # (N,H,hd)
        return out.reshape(N, hidden)

    raise ValueError(f"unknown op {op!r}")
