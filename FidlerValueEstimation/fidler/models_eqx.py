"""Equinox learned lambda2 estimators (size-invariant).

Both models map (x_node (H,N,6), x_adj (H,N,N)) -> (N,) predicted **log** lambda2,
one estimate per agent. They are size-invariant: every learned block acts per-node
(or per-edge via a normalized-mean aggregation `agg=(A_hat@z)/deg`, NEVER sum — sum
scales with N/degree and destroys size-transfer), so the SAME params run at any N.

- GRUEstimator: per-agent GRU over the H-step own-feature sequence (ignores x_adj).
- GCRNEstimator: spatio-temporal; lax.scan over H steps, one message-passing round
  per step (normalized mean aggregation) + a per-node GRU temporal update.
"""
import equinox as eqx
import jax
import jax.numpy as jnp

from . import messages as _messages


class GRUEstimator(eqx.Module):
    """Per-agent GRU over its own (H,6) feature history -> log-lambda2. Ignores x_adj."""
    cell: eqx.nn.GRUCell
    head: eqx.nn.MLP
    hidden: int = eqx.field(static=True)

    def __init__(self, in_size: int = 6, hidden: int = 64, *, key):
        kc, kh = jax.random.split(key)
        self.hidden = hidden
        self.cell = eqx.nn.GRUCell(in_size, hidden, key=kc)
        self.head = eqx.nn.MLP(hidden, 1, width_size=hidden, depth=1, key=kh)

    def __call__(self, x_node, x_adj):
        # x_node: (H, N, 6); x_adj ignored.
        H, N, _ = x_node.shape

        def per_agent(seq):  # seq: (H, 6)
            h0 = jnp.zeros((self.hidden,))

            def step(h, x):
                return self.cell(x, h), None

            h, _ = jax.lax.scan(step, h0, seq)
            return self.head(h)[0]  # scalar log-lambda2

        seqs = jnp.transpose(x_node, (1, 0, 2))            # (N, H, 6)
        return jax.vmap(per_agent)(seqs)                   # (N,)


class GCRNEstimator(eqx.Module):
    """Spatio-temporal: per step encode -> 1 MP round (normalized mean agg) -> per-node GRU."""
    encoder: eqx.nn.Linear
    message: eqx.nn.MLP
    cell: eqx.nn.GRUCell
    readout: eqx.nn.MLP
    hidden: int = eqx.field(static=True)

    def __init__(self, in_size: int = 6, hidden: int = 64, *, key):
        ke, km, kc, kr = jax.random.split(key, 4)
        self.hidden = hidden
        self.encoder = eqx.nn.Linear(in_size, hidden, key=ke)
        self.message = eqx.nn.MLP(2 * hidden, hidden, width_size=hidden, depth=1, key=km)
        self.cell = eqx.nn.GRUCell(hidden, hidden, key=kc)
        self.readout = eqx.nn.MLP(hidden, 1, width_size=hidden, depth=1, key=kr)

    def __call__(self, x_node, x_adj):
        # x_node: (H, N, 6); x_adj: (H, N, N) bool/float.
        H, N, _ = x_node.shape
        h0 = jnp.zeros((N, self.hidden))

        def step(h, inp):
            feats, adj = inp                              # (N,6), (N,N)
            z = jax.vmap(self.encoder)(feats)             # (N, hidden)

            a = adj.astype(jnp.float32)
            a = a - jnp.diag(jnp.diag(a))                 # strip self-loops -> A_hat
            deg = a.sum(-1, keepdims=True)                # (N,1)
            agg = (a @ z) / jnp.maximum(deg, 1.0)         # NORMALIZED MEAN (never sum)

            msg = jax.vmap(self.message)(jnp.concatenate([z, agg], axis=-1))  # (N, hidden)
            h_new = jax.vmap(self.cell)(msg, h)           # per-node temporal GRU update
            return h_new, None

        adj_f = x_adj.astype(jnp.float32)
        h, _ = jax.lax.scan(step, h0, (x_node, adj_f))    # scan over H steps
        return jax.vmap(self.readout)(h)[:, 0]            # (N,) log-lambda2


class ConfigurableGCRN(eqx.Module):
    """Configurable spatio-temporal estimator with a swappable message-passing round.

    Per `lax.scan` step over H: encode features -> run `n_rounds` message-passing rounds
    (via `fidler.messages.aggregate`, op/content configurable) with a residual combine ->
    per-node GRU temporal update. After H steps, THREE per-node heads read out from the
    GRU memory:
        * reg   -> log lambda2          ("logl2")
        * cflag -> connected logit      ("cflag")
        * unc   -> log sigma            ("logsig")

    Size-invariant for every op except `sum` (the deliberate ablation): all blocks act
    per-node / via normalized aggregation, so the SAME params run at any N.

    `__call__(x_node, x_adj, x_pos)` -> dict of (N,) arrays. An optional `key` enables
    DropEdge (training only); with `key=None` the forward pass is deterministic.
    """
    encoder: eqx.nn.Linear
    mp: _messages.MessageParams
    combine: eqx.nn.MLP                # residual round update: [z, agg] -> hidden
    cell: eqx.nn.GRUCell
    head_reg: eqx.nn.MLP
    head_cflag: eqx.nn.MLP
    head_unc: eqx.nn.MLP
    hidden: int = eqx.field(static=True)
    n_rounds: int = eqx.field(static=True)
    dropedge: float = eqx.field(static=True)

    def __init__(self, in_size=6, hidden=64, n_rounds=1, op="mean", content="value",
                 heads=4, *, key, dropedge=0.0, comm_r=5.0):
        ke, kmp, kcomb, kc, kr, kf, ku = jax.random.split(key, 7)
        self.hidden = int(hidden)
        self.n_rounds = int(n_rounds)
        self.dropedge = float(dropedge)
        self.encoder = eqx.nn.Linear(in_size, hidden, key=ke)
        self.mp = _messages.MessageParams(hidden=hidden, heads=heads, content=content,
                                          op=op, key=kmp, comm_r=comm_r)
        self.combine = eqx.nn.MLP(2 * hidden, hidden, width_size=hidden, depth=1, key=kcomb)
        self.cell = eqx.nn.GRUCell(hidden, hidden, key=kc)
        self.head_reg = eqx.nn.MLP(hidden, 1, width_size=hidden, depth=1, key=kr)
        self.head_cflag = eqx.nn.MLP(hidden, 1, width_size=hidden, depth=1, key=kf)
        self.head_unc = eqx.nn.MLP(hidden, 1, width_size=hidden, depth=1, key=ku)

    def __call__(self, x_node, x_adj, x_pos, *, key=None):
        # x_node: (H,N,6); x_adj: (H,N,N); x_pos: (H,N,2).
        H, N, _ = x_node.shape
        h0 = jnp.zeros((N, self.hidden))
        op = self.mp.op
        content = self.mp.content
        drop = self.dropedge if key is not None else 0.0

        # per-step keys for DropEdge (one key per step, further split per round below)
        if key is None:
            step_keys = jnp.zeros((H, 2), dtype=jnp.uint32)
        else:
            step_keys = jax.random.split(key, H)

        def step(h, inp):
            feats, adj, pos, sk = inp                     # (N,6),(N,N),(N,2),(2,)
            z = jax.vmap(self.encoder)(feats)             # (N, hidden)
            round_keys = jax.random.split(sk, self.n_rounds)
            for r in range(self.n_rounds):
                agg = _messages.aggregate(z, adj, pos, op=op, content=content,
                                          params=self.mp, key=round_keys[r], dropedge=drop)
                z = z + jax.vmap(self.combine)(jnp.concatenate([z, agg], axis=-1))  # residual
            h_new = jax.vmap(self.cell)(z, h)             # per-node temporal GRU
            return h_new, None

        adj_f = x_adj.astype(jnp.float32)
        pos_f = x_pos.astype(jnp.float32)
        h, _ = jax.lax.scan(step, h0, (x_node, adj_f, pos_f, step_keys))
        return {
            "logl2": jax.vmap(self.head_reg)(h)[:, 0],    # (N,)
            "cflag": jax.vmap(self.head_cflag)(h)[:, 0],  # (N,)
            "logsig": jax.vmap(self.head_unc)(h)[:, 0],   # (N,)
        }
