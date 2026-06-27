"""Grounded CTDE v0 networks (Equinox) — LPAC backbone + GNN-KB, multi-level
goal head, decentralized λ̂₂ head, and a centralized critic.

This is the TeamBlue agent (agent_architecture.md), NOT a flat actor-critic. The
learning stack does NOT emit raw moves: L3 picks a goal off the belief; a fixed
L1 controller (``controller.py``) turns it into the env move.

Pipeline (decentralized, runs per agent at execution):

  obs_i (C,H,W)
    └─[1] CNN local-perception (depth conv, same-pad, ReLU) ─ GAP ──▶ f_i (W,)
         (GAP, not flatten: latent dim independent of H/W -> scale-invariant;
          conv weight-sharing is translation-equivariant in perception.)
    └─[2] GNN message-passing KB: fuse in-range NEIGHBOURS' features over the
         comm graph (adjacency from positions at comm_r), ``mp_rounds`` rounds,
         a configurable NORMALIZED aggregator (mean | max | multihead) ──▶ z_i
         (the "KB" + "comms/aggregation agent-count-invariant" modules).
    └─ off z_i:
         (a) GOAL head  -> logits over K candidate relative waypoints (L3 intent)
         (b) λ̂₂  head   -> the decentralized local-Fiedler estimate (one scalar)
         (c) value head -> per-agent baseline (diagnostic / IPPO fallback)

The **goal** policy is what PPO optimizes; the controller is fixed. The centralized
**Critic** (CTDE, training only) reads ``central_obs`` (Cg,H,W).

The GNN aggregator is the heart of scale-invariance: it never raw-sums neighbours
(which would scale with team size); mean / max / softmax-attention are all
agent-count-invariant. Adjacency is derived from positions at ``comm_r`` so the
KB fuses exactly the in-range team — the comm graph the formalism's *bridge*
exposes.
"""
from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from . import controller as _ctrl

# Obs channel indices for the comm-coverage recipe's GridObs stack
# ("known", "own_pos", "known_walls", "neighbors", "local_frontier"). The
# frontier module reads two of them: ``known`` (the post-gossip belief — 1.0
# where a cell is known/covered, 0.0 where UNKNOWN, so 1 - known = the frontier
# of uncovered cells) and ``own_pos`` (the one-hot of the agent's own cell, from
# which the agent's grid position is recovered as a centroid — no absolute
# coordinate is ever fed in, only the displacement geometry it induces). The
# compass module additionally reads ``neighbors`` (one-hots of the agent's in-range
# teammates — its centroid gives the team "gather" direction).
_CH_KNOWN = 0
_CH_OWN_POS = 1
_CH_NEIGHBORS = 3


# =============================================================================
# [1] CNN local-perception (per agent)
# =============================================================================


def _conv_stack(in_ch: int, width: int, depth: int, key):
    """Same-padding 3x3 conv stack: in_ch -> width, ``depth`` layers."""
    keys = jax.random.split(key, depth)
    layers, ch = [], in_ch
    for i in range(depth):
        layers.append(eqx.nn.Conv2d(ch, width, kernel_size=3, padding=1, key=keys[i]))
        ch = width
    return layers


def _encode(layers, x):
    """(C,H,W) -> (width,) via conv+ReLU then global-average-pool [size-invariant]."""
    h = x
    for conv in layers:
        h = jax.nn.relu(conv(h))
    return h.mean(axis=(1, 2))


# =============================================================================
# [2] GNN message-passing KB (configurable, normalized aggregator)
# =============================================================================


# Per-sender IDENTITY signal dimension for ``message_content == 'index'`` — a small
# FIXED (non-learned) sinusoidal embedding of the agent's NORMALIZED index i/N. Fixed
# (not a per-N learned table) and a function of i/N (not i) so it is agent-count-
# invariant: a model trained @4 agents reads the SAME identity geometry @10 agents.
_IDX_DIM = 8


def _index_signal(n: int) -> jax.Array:
    """(N, ``_IDX_DIM``) float32 per-sender IDENTITY embedding — a fixed sinusoidal
    code of each agent's NORMALIZED index ``i/N`` so a receiver can tell its neighbours
    apart WITHOUT a learned per-N table.

    AGENT-COUNT-INVARIANT by construction: the code is a function of the fraction
    ``u_i = i/N ∈ [0,1)`` (not the raw index i) evaluated at a fixed bank of geometric
    frequencies, so the SAME function maps any team size — identity ``j=2`` in a team of
    4 (``u=0.5``) and identity ``j=5`` in a team of 10 (``u=0.5``) get the same signal,
    and the embedding dimension ``_IDX_DIM`` never depends on N. The frequencies are the
    standard transformer geometric ladder; ``_IDX_DIM`` must be even (sin/cos pairs).
    Pure JAX (jit-safe; ``n`` is static under the rollout/loss scans)."""
    half = _IDX_DIM // 2
    u = jnp.arange(n, dtype=jnp.float32) / jnp.maximum(float(n), 1.0)   # (N,) i/N in [0,1)
    freqs = 2.0 * jnp.pi * (2.0 ** jnp.arange(half, dtype=jnp.float32))  # (half,) geometric
    ang = u[:, None] * freqs[None, :]                                   # (N, half)
    return jnp.concatenate([jnp.sin(ang), jnp.cos(ang)], axis=-1)       # (N, _IDX_DIM)


# extra per-receiver message-content channels appended to the aggregated neighbour
# summary, keyed by ``message_content``. ``learned`` adds NOTHING (extra dim 0) so the
# update Linear is the original (2W -> W) and the layer is byte-identical to v0.
_EXTRA_DIM = {"learned": 0, "edge_distance": 2, "index": _IDX_DIM}


class MPLayer(eqx.Module):
    """One message-passing round over the comm graph.

    Each node updates from (its own feature) + (aggregated neighbour messages).
    The aggregator is configurable and ALWAYS normalized / size-invariant:

      * "mean" — degree-normalized average of neighbour messages.
      * "max"  — elementwise max over neighbours (default; magnitude-invariant).
      * "multihead" — softmax attention over neighbours (``heads`` heads), the
        attention weights sum to 1 so the readout is agent-count-invariant.

    ``message_content`` (the I2 "message design" dial) selects WHAT each agent puts in
    its comm message BEYOND the learned ``msg`` feature transform — an EXTRA per-edge
    channel appended to the aggregated summary before the update (the receiver fuses it
    alongside the learned messages). It is ALWAYS aggregated with a degree-normalized
    MEAN (count-invariant regardless of ``agg``) so adding it never breaks size-transfer:

      * "learned" (default) — NOTHING extra; the message is ``msg(feats)`` exactly, the
        update reads ``[self || agg]`` (2W) and the layer is BYTE-IDENTICAL to v0.
      * "edge_distance" — append the (comm_r-normalized) sender→receiver Chebyshev
        distance per edge, summarized to the receiver as its [mean, min] neighbour
        distance (2 channels). The receiver thus knows HOW FAR each neighbour is; the
        normalization by ``comm_r`` keeps it in [0,1] = scale-invariant.
      * "index" — append a fixed sinusoidal embedding of the SENDER's normalized index
        (``_index_signal``; ``_IDX_DIM`` channels), mean-pooled over neighbours, so the
        receiver can tell its neighbours apart. Fixed (not a learned per-N table) and a
        function of i/N -> agent-count-invariant.

    For the two non-default modes the update Linear widens to ``2W + extra`` and a ``dist``
    (N,N) matrix (normalized sender→receiver distance, diagonal 0) is threaded in by the
    caller; the ``learned`` path ignores ``dist`` and keeps the (2W -> W) update.

    A boolean ``adj`` (N,N, self-loops removed by the caller for neighbour msgs)
    selects who is in range. ``__call__(feats, adj_off, dist=None)`` -> (N, width).
    """
    msg: eqx.nn.Linear          # neighbour-message transform
    upd: eqx.nn.Linear          # node update from [self || aggregated || extra]
    q: eqx.nn.Linear            # attention query (multihead)
    k: eqx.nn.Linear            # attention key   (multihead)
    agg: str = eqx.field(static=True)
    heads: int = eqx.field(static=True)
    width: int = eqx.field(static=True)
    message_content: str = eqx.field(static=True)

    def __init__(self, width: int, agg: str, heads: int, *, key,
                 message_content: str = "learned"):
        km, ku, kq, kk = jax.random.split(key, 4)
        extra = _EXTRA_DIM[str(message_content)]
        self.msg = eqx.nn.Linear(width, width, key=km)
        # update input = [self (W) || agg (W) || message-content extra (E)]; E==0 for
        # the default 'learned' -> Linear(2W, W) exactly as v0 (byte-identical param tree).
        self.upd = eqx.nn.Linear(2 * width + extra, width, key=ku)
        self.q = eqx.nn.Linear(width, width, key=kq)
        self.k = eqx.nn.Linear(width, width, key=kk)
        self.agg = agg
        self.heads = int(heads)
        self.width = int(width)
        self.message_content = str(message_content)

    def _content_extra(self, n: int, mask, dist) -> jax.Array:
        """(N, E) the message-content EXTRA channels for the receivers — what each agent
        appends to its message beyond the learned ``msg`` transform, mean-pooled over its
        in-range neighbours (degree-normalized -> agent-count-invariant). ``mask`` (N,N)
        bool selects neighbours; ``dist`` (N,N) is the normalized sender→receiver distance.

          * edge_distance — [mean, min] of each receiver's neighbour distances (2 chans).
          * index         — mean of the SENDER identity codes over neighbours (_IDX_DIM).

        ``learned`` never calls this (extra dim 0). Pure JAX (jit-safe)."""
        deg = jnp.maximum(mask.sum(-1, keepdims=True), 1.0)             # (N,1) neighbour count
        if self.message_content == "edge_distance":
            # dist is normalized (in [0,1]); summarize each receiver's neighbourhood by the
            # MEAN and MIN (nearest) neighbour distance -> a near vs far neighbour shifts it.
            mf = mask.astype(jnp.float32)
            mean_d = (mf * dist).sum(-1, keepdims=True) / deg           # (N,1) mean dist
            far = jnp.where(mask, dist, jnp.inf)                        # (N,N) non-edges -> +inf
            min_d = jnp.min(far, axis=-1, keepdims=True)               # (N,1) nearest nbr
            min_d = jnp.where(jnp.isfinite(min_d), min_d, 0.0)        # isolated -> 0
            return jnp.concatenate([mean_d, min_d], axis=-1)           # (N,2)
        if self.message_content == "index":
            ids = _index_signal(n)                                      # (N,_IDX_DIM) sender codes
            return (mask.astype(jnp.float32) @ ids) / deg              # (N,_IDX_DIM) mean over nbrs
        raise ValueError(f"unknown message_content {self.message_content!r}")

    def __call__(self, feats, adj_off, dist=None):
        # feats (N,W); adj_off (N,N) bool with diagonal already cleared. dist (N,N) is the
        # normalized sender->receiver distance (only the non-'learned' modes consume it).
        n = feats.shape[0]
        m = jax.vmap(self.msg)(feats)                       # (N,W) messages
        mask = adj_off                                      # (N,N) i has neighbour j

        if self.agg == "mean":
            deg = jnp.maximum(mask.sum(-1, keepdims=True), 1.0)          # (N,1)
            agg = (mask.astype(jnp.float32) @ m) / deg                   # (N,W)
        elif self.agg == "max":
            # masked elementwise max over neighbours; -inf where no edge, 0 if isolated.
            neigh = jnp.where(mask[:, :, None], m[None, :, :], -jnp.inf)  # (N,N,W)
            agg = jnp.max(neigh, axis=1)                                  # (N,W)
            agg = jnp.where(jnp.isfinite(agg), agg, 0.0)
        elif self.agg == "multihead":
            q = jax.vmap(self.q)(feats).reshape(n, self.heads, -1)       # (N,Hd,dh)
            k = jax.vmap(self.k)(feats).reshape(n, self.heads, -1)
            v = m.reshape(n, self.heads, -1)                             # (N,Hd,dh)
            dh = q.shape[-1]
            # scores[i,j,head] = q_i . k_j / sqrt(dh)
            scores = jnp.einsum("ihd,jhd->ijh", q, k) / jnp.sqrt(dh)     # (N,N,Hd)
            scores = jnp.where(mask[:, :, None], scores, -jnp.inf)
            # rows with no neighbour: all -inf -> softmax NaN; guard to 0 weights.
            has = mask.any(-1)                                           # (N,)
            attn = jax.nn.softmax(scores, axis=1)                       # (N,N,Hd)
            attn = jnp.where(has[:, None, None], attn, 0.0)
            ctx = jnp.einsum("ijh,jhd->ihd", attn, v).reshape(n, -1)     # (N,W)
            agg = ctx
        else:
            raise ValueError(f"unknown aggregator {self.agg!r}")

        # message-content EXTRA (the I2 message-design dial): for 'learned' (default) the
        # branch is skipped entirely (message_content is STATIC) -> cat is [self || agg]
        # (2W) and the update is byte-identical to v0; the non-default modes append their
        # extra (distance / identity) channels, aggregated count-invariantly.
        if self.message_content == "learned":
            cat = jnp.concatenate([feats, agg], axis=-1)                # (N,2W)
        else:
            extra = self._content_extra(n, mask, dist)                 # (N,E)
            cat = jnp.concatenate([feats, agg, extra], axis=-1)         # (N,2W+E)
        out = jax.vmap(self.upd)(cat)                                   # (N,W)
        return jax.nn.relu(out)


class Backbone(eqx.Module):
    """LPAC backbone: per-agent CNN -> GAP -> feature, then ``mp_rounds`` of GNN
    message passing over the comm graph -> per-agent belief ``z_i`` (N, width).

    ``__call__(obs, adj_off, *, dist=None, key)`` with ``obs`` (N,C,H,W) and ``adj_off``
    (N,N) bool (in-range neighbours, diagonal cleared). ``dist`` (N,N) is the NORMALIZED
    sender→receiver distance (in [0,1], diagonal 0) the non-default ``message_content``
    modes append to each message; the default ``learned`` mode ignores it entirely (so
    the forward is byte-identical to v0 whether or not a ``dist`` is supplied). Optional
    LayerNorm + dropout on the belief. Returns ``z`` (N, width).

    ``message_content`` (the I2 message-design dial) is threaded into every ``MPLayer``;
    see :class:`MPLayer` for the modes (learned | edge_distance | index).
    """
    conv: list
    mp: list
    ln: eqx.nn.LayerNorm | None
    drop: eqx.nn.Dropout | None
    width: int = eqx.field(static=True)
    message_content: str = eqx.field(static=True)

    def __init__(self, in_ch: int, width: int, depth: int, mp_rounds: int,
                 agg: str, heads: int, norm: str, dropout: float, *, key,
                 message_content: str = "learned"):
        kc, kmp = jax.random.split(key)
        self.conv = _conv_stack(in_ch, width, depth, kc)
        mp_keys = jax.random.split(kmp, max(mp_rounds, 1))
        self.mp = [MPLayer(width, agg, heads, key=mp_keys[i],
                           message_content=message_content)
                   for i in range(mp_rounds)]
        self.ln = eqx.nn.LayerNorm(width) if norm == "layer" else None
        self.drop = eqx.nn.Dropout(dropout) if dropout > 0 else None
        self.width = int(width)
        self.message_content = str(message_content)

    def __call__(self, obs, adj_off, *, dist=None, key=None, inference: bool = False):
        feats = jax.vmap(lambda o: _encode(self.conv, o))(obs)         # (N,W)
        z = feats
        for layer in self.mp:
            z = layer(z, adj_off, dist)                                # (N,W)
        if self.ln is not None:
            z = jax.vmap(self.ln)(z)
        if self.drop is not None:
            z = self.drop(z, key=key, inference=inference)
        return z


# =============================================================================
# [3] Frontier-attention explorer tool (the L4 "disperse" skill)
# =============================================================================
#
# The goal head picks 1 of K compass waypoints off the belief z ALONE. Our
# diagnostics show that from a clustered spawn the reward-driven goal head cannot
# discover dispersal at 32²/10 (coverage stalls ~16%). This module gives the
# explorer an EXPLICIT frontier-seeking mechanism: it reads the agent's own
# observation, measures how much UNCOVERED ground sits in each of the K compass
# SECTORS, and biases the goal logits toward the most informative unexplored
# direction. It is a learned Equinox submodule (attention over the K sectors) but
# its sector FEATURES are hand-derived, normalized fractions — so a model trained
# at one grid size transfers to another.


def _compass_unit_dirs(K: int) -> jax.Array:
    """(K, 2) float32 unit (row, col) directions for the K compass sectors — the
    same ordering as ``controller._COMPASS`` (index 0 = "here", zero vector). The
    8 directional offsets are L2-normalized so only the ANGLE matters (scale-free);
    sector 0 stays the zero vector (no direction — the "stay near me" fallback)."""
    base = _ctrl._COMPASS[:K].astype(jnp.float32)                   # (K,2) e.g. (-1,1)
    norm = jnp.sqrt((base ** 2).sum(-1, keepdims=True))             # (K,1)
    return base / jnp.maximum(norm, 1.0)                           # (K,2) unit (here->0)


def sector_frontier_features(obs_i: jax.Array, K: int, sharp: float = 4.0) -> jax.Array:
    """(K, 2) float32 per-sector frontier features for ONE agent's obs (C,H,W).

    SCALE-INVARIANT by construction — every quantity is a fraction or a unit
    direction; no absolute coordinate or grid-size-dependent magnitude survives:

      frontier(cell)  = 1 - known(cell)                 (ch ``_CH_KNOWN``; uncovered)
      (cr, cc)        = centroid of own_pos              (ch ``_CH_OWN_POS``; the agent)
      u(cell)         = (row-cr, col-cc) / ||·||         (UNIT displacement, scale-free)
      m_k(cell)       = softmax_k( sharp · u(cell)·dir_k )  over the K compass dirs
      feat[k,0]       = Σ_cell m_k·frontier / Σ_cell m_k     (frontier FRACTION in sector k)
      feat[k,1]       = Σ_cell m_k·frontier / (H·W)          (frontier DENSITY toward k)

    ``feat[:,0]`` answers "of the cells lying toward compass-dir k, what fraction is
    unexplored?" and ``feat[:,1]`` "how much of my whole view's frontier sits toward
    k?" — both bounded in [0,1] regardless of H,W or team size. The agent's own cell
    (zero displacement) carries no direction; the soft sector membership lets it fall
    to the "here" sector (dir 0) so it never spuriously votes for a compass heading.
    Pure JAX (vmap/jit-safe)."""
    C, H, W = obs_i.shape
    frontier = 1.0 - obs_i[_CH_KNOWN]                              # (H,W) 1=uncovered
    own = obs_i[_CH_OWN_POS]                                       # (H,W) one-hot

    # Recover the agent's (row, col) as the centroid of its own-position one-hot —
    # exact for a one-hot, and robust if it were ever smoothed. NO absolute coord is
    # exported; only per-cell displacement (a relative, translation-free geometry).
    rows = jnp.arange(H, dtype=jnp.float32)[:, None]              # (H,1)
    cols = jnp.arange(W, dtype=jnp.float32)[None, :]             # (1,W)
    mass = jnp.maximum(own.sum(), 1.0)
    cr = (own * rows).sum() / mass                                # () agent row
    cc = (own * cols).sum() / mass                                # () agent col

    dr = rows - cr                                                # (H,1) row displacement
    dc = cols - cc                                                # (1,W) col displacement
    dr = jnp.broadcast_to(dr, (H, W))
    dc = jnp.broadcast_to(dc, (H, W))
    dist = jnp.sqrt(dr ** 2 + dc ** 2)                            # (H,W) Euclidean radius
    inv = 1.0 / jnp.maximum(dist, 1e-6)
    ur = dr * inv                                                 # (H,W) unit row dir
    uc = dc * inv                                                 # (H,W) unit col dir

    dirs = _compass_unit_dirs(K)                                  # (K,2) unit compass dirs
    # cosine of each cell's direction with each sector direction -> (K,H,W).
    cos = dirs[:, 0][:, None, None] * ur[None] + dirs[:, 1][:, None, None] * uc[None]
    # the "here" sector (dir 0 -> cos==0 everywhere) should win only for cells AT the
    # agent (tiny radius); give it a closeness score so near-cell frontier lands there
    # rather than leaking into an arbitrary compass heading. Directional sectors keep
    # their cosine. score_k(cell): closeness for sector 0, cosine for sectors >=1.
    is_here = (jnp.abs(dirs[:, 0]) + jnp.abs(dirs[:, 1])) < 1e-6  # (K,) True for dir 0
    closeness = jnp.exp(-dist)[None]                              # (1,H,W) in (0,1], 1 at agent
    score = jnp.where(is_here[:, None, None], closeness, cos)     # (K,H,W)

    member = jax.nn.softmax(sharp * score, axis=0)               # (K,H,W) soft sector assign
    fmass = (member * frontier[None]).sum(axis=(1, 2))           # (K,) frontier mass / sector
    smass = member.sum(axis=(1, 2))                              # (K,) cell mass / sector
    frac = fmass / jnp.maximum(smass, 1e-6)                       # (K,) frontier FRACTION
    dens = fmass / float(H * W)                                   # (K,) frontier DENSITY
    return jnp.stack([frac, dens], axis=-1)                       # (K,2) per-sector feats


class FrontierAttn(eqx.Module):
    """Frontier-biased goal-sector attention — the explorer's "disperse" tool.

    Queries the belief ``z`` and keys the K per-sector frontier features, producing
    one additive logit per compass sector that PULLS the goal policy toward the
    sector with the most informative unexplored ground. The combined goal logits are

        goal_logits = goal_head(z) + alpha · frontier_logits

    so PPO still samples a goal from a DISTRIBUTION (the attention biases, never
    argmaxes — the policy keeps training). ``alpha`` is a learned scalar gate
    (softplus, ≥0) so the network can dial the frontier pull up or down per the
    reward signal, starting near a configured value.

    Construction (``F_feat`` = per-sector feature dim, ``d`` = attention dim):
      * ``q``    Linear(W -> d)       query from the belief z
      * ``k``    Linear(F_feat -> d)  key   from each sector's frontier feature

    ``score_k = q(z)·k(feat_k)/√d`` gives a belief-conditioned attention weight per
    sector (softmax over K). The contributed logit MULTIPLIES that learned weight by
    the sector's own frontier FRACTION (``feat[k,0]`` — a non-negative, frontier-
    peaked scalar): ``frontier_logit_k = K · attn_k · frontier_frac_k`` (the ``K``
    restores unit scale since ``Σ attn = 1``). This is high where the belief asks
    "explore here" AND sector k is frontier-rich, and — crucially — it is
    frontier-POSITIVE BY CONSTRUCTION: even at random init (``attn`` ≈ uniform) the
    largest additive logit lands on the most-uncovered sector, an INDUCTIVE BIAS the
    reward then sharpens (via q/k and ``alpha``) rather than having to discover from
    scratch. SIZE-INVARIANT: K is fixed, the frontier fraction is a normalized
    fraction, and the dot-product readout is independent of H, W and team size.
    Pure JAX (vmap/jit-safe)."""
    q: eqx.nn.Linear
    k: eqx.nn.Linear
    log_alpha: jax.Array              # learned gate: alpha = softplus(log_alpha) >= 0
    d: int = eqx.field(static=True)
    sharp: float = eqx.field(static=True)
    F_feat: int = eqx.field(static=True)

    def __init__(self, width: int, *, d: int = 32, F_feat: int = 2,
                 sharp: float = 4.0, alpha_init: float = 1.0, key):
        kq, kk = jax.random.split(key, 2)
        self.q = eqx.nn.Linear(width, d, key=kq)
        self.k = eqx.nn.Linear(F_feat, d, key=kk)
        # invert softplus so alpha starts ≈ alpha_init: softplus(x)=alpha_init.
        a0 = float(max(alpha_init, 1e-4))
        self.log_alpha = jnp.asarray(jnp.log(jnp.expm1(a0)), dtype=jnp.float32)
        self.d = int(d)
        self.sharp = float(sharp)
        self.F_feat = int(F_feat)

    def sector_logits(self, z_i: jax.Array, feats_i: jax.Array) -> jax.Array:
        """(K,) additive goal logits for ONE agent: belief ``z_i`` (W,) attending
        over the per-sector frontier features ``feats_i`` (K, F_feat). The learned
        attention weights the sector's own frontier fraction (``feats_i[:,0]``), so
        the readout is non-negative and peaks at the most-uncovered sector."""
        K = feats_i.shape[0]
        qz = self.q(z_i)                                          # (d,) query
        kf = jax.vmap(self.k)(feats_i)                            # (K,d) sector keys
        scores = (kf @ qz) / jnp.sqrt(float(self.d))             # (K,) attention scores
        attn = jax.nn.softmax(scores, axis=0)                    # (K,) sector weights, Σ=1
        frac = feats_i[:, 0]                                      # (K,) frontier fraction >=0
        return float(K) * attn * frac                            # (K,) frontier logits

    def __call__(self, obs, z, K: int) -> jax.Array:
        """(N, K) additive frontier logits for the team. ``obs`` (N,C,H,W), belief
        ``z`` (N,W). Reads each agent's own obs to build its sector frontier features
        (``sector_frontier_features``), then gates the attention readout by
        ``alpha = softplus(log_alpha)``. Returns the term to ADD to ``goal_head(z)``
        (so ``alpha == 0`` is exactly the unmodified goal policy)."""
        feats = jax.vmap(lambda o: sector_frontier_features(o, K, self.sharp))(obs)  # (N,K,F)
        logits = jax.vmap(self.sector_logits)(z, feats)          # (N,K)
        alpha = jax.nn.softplus(self.log_alpha)                  # () >= 0
        return alpha * logits                                    # (N,K) gated


# =============================================================================
# [4] Compass directional feature (explicit navigation signal for the heads)
# =============================================================================
#
# The CNN local-perception sees only a translation-equivariant patch (after GAP no
# absolute bearing survives); the compass gives every agent an EXPLICIT, scale-free
# sense of "which way is my team" (the GATHER direction) and "which way is fresh
# ground" (the EXPLORE direction), so the role / goal heads can navigate with a
# directional cue beyond the local view. Both directions are DIRECTIONS ONLY — soft
# K-sector distributions over the ``controller._COMPASS`` headings (no distances, no
# absolute coordinates) — so a model trained at one grid size transfers to another
# (Compass axis of agent_architecture.md / I2). When ``compass == 'off'`` the module
# is still BUILT (a stable param surface) but never used, so the belief z and every
# head are byte-identical to the pre-compass actor.


def _agent_unit_dirs(own_i: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """From an agent's own-position one-hot ``own_i`` (H,W) recover, per cell, the
    UNIT displacement from the agent toward that cell — the scale-free geometry the
    soft-sector features key on. Returns ``(ur, uc, dist)`` each (H,W):

      (cr, cc) = centroid of own_i           (the agent's cell; exact for a one-hot)
      (dr, dc) = (row - cr, col - cc)         (per-cell displacement from the agent)
      dist     = ||(dr, dc)||                 (Euclidean radius, in cells)
      (ur, uc) = (dr, dc) / max(dist, eps)    (UNIT direction — only the ANGLE matters)

    No absolute coordinate is exported — only the relative, translation-free
    displacement geometry. Pure JAX (vmap/jit-safe)."""
    H, W = own_i.shape
    rows = jnp.arange(H, dtype=jnp.float32)[:, None]              # (H,1)
    cols = jnp.arange(W, dtype=jnp.float32)[None, :]             # (1,W)
    mass = jnp.maximum(own_i.sum(), 1.0)
    cr = (own_i * rows).sum() / mass                              # () agent row
    cc = (own_i * cols).sum() / mass                              # () agent col
    dr = jnp.broadcast_to(rows - cr, (H, W))                     # (H,W) row displacement
    dc = jnp.broadcast_to(cols - cc, (H, W))                     # (H,W) col displacement
    dist = jnp.sqrt(dr ** 2 + dc ** 2)                           # (H,W) radius
    inv = 1.0 / jnp.maximum(dist, 1e-6)
    return dr * inv, dc * inv, dist                              # (H,W) ur, uc, dist


def _soft_sector_dir(mass_plane: jax.Array, ur: jax.Array, uc: jax.Array,
                     dist: jax.Array, dirs: jax.Array, sharp: float,
                     decay: float = 0.0) -> jax.Array:
    """(K,) soft K-sector one-hot of the DIRECTION toward the mass in ``mass_plane``.

    Each cell with mass votes for the compass sector its unit displacement
    ``(ur,uc)`` aligns with (cosine vs each ``dirs`` heading); the votes are pooled
    into a softmax over the K sectors, giving a normalized "which compass heading
    points at this stuff" distribution that is SCALE-INVARIANT (cosine of unit
    directions + a normalized softmax — no grid-size magnitude survives):

      score_k(cell) = closeness            for the "here" sector (dir 0)
                      cos(u(cell), dir_k)  for the directional sectors (k >= 1)
      w(cell)       = mass(cell) · exp(-decay · dist(cell))   (optional nearer-weighting)
      sect_k        = Σ_cell w·softmax_k(sharp·score_k) / Σ_cell w

    With ``decay > 0`` nearer mass dominates (the EXPLORE direction points at the
    NEAREST fresh ground, not the global frontier centroid); ``decay == 0`` is a plain
    mass centroid direction (the GATHER direction toward in-range teammates). If the
    plane is empty (no mass) the result falls entirely on the "here" sector (index 0),
    i.e. "no direction" — a safe scale-free default. Pure JAX (vmap/jit-safe)."""
    is_here = (jnp.abs(dirs[:, 0]) + jnp.abs(dirs[:, 1])) < 1e-6  # (K,) True for dir 0
    cos = dirs[:, 0][:, None, None] * ur[None] + dirs[:, 1][:, None, None] * uc[None]  # (K,H,W)
    closeness = jnp.exp(-dist)[None]                              # (1,H,W) 1 at the agent
    score = jnp.where(is_here[:, None, None], closeness, cos)     # (K,H,W)
    member = jax.nn.softmax(sharp * score, axis=0)               # (K,H,W) soft sector assign
    weight = mass_plane * jnp.exp(-decay * dist)                  # (H,W) (nearer-weighted) mass
    sect = (member * weight[None]).sum(axis=(1, 2))              # (K,) pooled sector mass
    total = sect.sum()                                           # () total weighted mass
    # empty plane -> put all mass on the "here" sector (no direction); else normalize.
    fallback = is_here.astype(jnp.float32)                       # (K,) one-hot on dir 0
    return jnp.where(total > 1e-6, sect / jnp.maximum(total, 1e-6), fallback)  # (K,)


def compass_features(obs_i: jax.Array, K: int, sharp: float = 4.0,
                     explore_decay: float = 0.5) -> jax.Array:
    """(2, K) float32 directional compass features for ONE agent's obs (C,H,W):

      row 0 — GATHER direction: a soft K-sector one-hot pointing toward the centroid
              of the agent's IN-RANGE TEAMMATES (the ``neighbors`` channel one-hots);
              "which way is my team".
      row 1 — EXPLORE direction: a soft K-sector one-hot pointing toward the NEAREST
              UNCOVERED cell in view (frontier = ``1 - known``, distance-decayed so the
              nearest fresh ground dominates); "which way is fresh ground".

    Both rows are normalized soft sector distributions over ``controller._COMPASS``
    (Σ_k = 1, index 0 = "here"/no-direction) — DIRECTIONS ONLY. SCALE-INVARIANT by
    construction: every quantity is a cosine of unit displacements or a normalized
    softmax, so no absolute coordinate or grid-size magnitude survives and the SAME
    relative layout yields the same features at any H, W or team size. When the agent
    has no in-range teammate (gather) or no frontier in view (explore) that row falls
    to the "here" sector. Pure JAX (vmap/jit-safe)."""
    own = obs_i[_CH_OWN_POS]                                      # (H,W) own one-hot
    neigh = obs_i[_CH_NEIGHBORS]                                  # (H,W) teammate one-hots
    frontier = 1.0 - obs_i[_CH_KNOWN]                            # (H,W) 1 = uncovered
    ur, uc, dist = _agent_unit_dirs(own)                         # (H,W) unit dirs + radius
    dirs = _compass_unit_dirs(K)                                  # (K,2) unit compass headings
    gather = _soft_sector_dir(neigh, ur, uc, dist, dirs, sharp, decay=0.0)        # (K,)
    explore = _soft_sector_dir(frontier, ur, uc, dist, dirs, sharp, decay=explore_decay)  # (K,)
    return jnp.stack([gather, explore], axis=0)                  # (2,K) directions


class Compass(eqx.Module):
    """The compass directional-feature module — an explicit, scale-free navigation
    signal added to the per-agent belief ``z`` before the heads.

    It reads each agent's own obs to build two soft K-sector DIRECTION distributions
    (``compass_features``: GATHER = toward in-range teammates, EXPLORE = toward the
    nearest uncovered cell), flattens them to ``2K`` scalars, and PROJECTS+GATES them
    into the belief width to ADD to ``z``:

        z' = z + beta · proj( [gather(K) , explore(K)] )

    The contribution is gated by a learned scalar ``beta = softplus(log_beta) >= 0``
    (so the network dials the navigation pull per the reward, starting near a
    configured value). Projecting-and-adding (rather than concatenating + widening the
    heads) keeps the head input width — and therefore the WHOLE param surface of the
    goal / role / λ̂₂ / value heads — IDENTICAL whether the compass is on or off; the
    module is ALWAYS built (mirrors ``FrontierAttn`` / the role head) and only its USE
    is gated, so with ``compass == 'off'`` the belief z is byte-identical to the
    pre-compass actor. SIZE-INVARIANT: K is fixed and the features are normalized
    directions, so a model trained @16²/4 transfers up the scale ladder. Pure JAX
    (vmap/jit-safe)."""
    proj: eqx.nn.Linear                # (2K -> W) project the directional features
    log_beta: jax.Array               # learned gate: beta = softplus(log_beta) >= 0
    K: int = eqx.field(static=True)
    width: int = eqx.field(static=True)
    sharp: float = eqx.field(static=True)
    explore_decay: float = eqx.field(static=True)

    def __init__(self, width: int, K: int, *, sharp: float = 4.0,
                 explore_decay: float = 0.5, beta_init: float = 1.0, key):
        self.proj = eqx.nn.Linear(2 * K, width, key=key)
        # invert softplus so beta starts ≈ beta_init: softplus(x)=beta_init.
        b0 = float(max(beta_init, 1e-4))
        self.log_beta = jnp.asarray(jnp.log(jnp.expm1(b0)), dtype=jnp.float32)
        self.K = int(K)
        self.width = int(width)
        self.sharp = float(sharp)
        self.explore_decay = float(explore_decay)

    def __call__(self, obs, z) -> jax.Array:
        """(N, W) the compass term to ADD to the belief ``z`` (N,W). ``obs`` (N,C,H,W).
        Builds each agent's (2,K) directions, flattens to (2K,), projects to W and
        gates by ``beta = softplus(log_beta)``. ``beta == 0`` is exactly z unchanged."""
        feats = jax.vmap(lambda o: compass_features(o, self.K, self.sharp,
                                                    self.explore_decay))(obs)  # (N,2,K)
        flat = feats.reshape(feats.shape[0], -1)                 # (N,2K)
        proj = jax.vmap(self.proj)(flat)                         # (N,W)
        beta = jax.nn.softplus(self.log_beta)                    # () >= 0
        return beta * proj                                       # (N,W) gated term


# =============================================================================
# Actor: backbone + goal / λ̂₂ / value heads (decentralized)
# =============================================================================


class Actor(eqx.Module):
    """Decentralized per-agent actor: LPAC backbone -> belief z_i -> four heads
    (+ the frontier-attention explorer tool + the compass directional feature).

      * ``goal_head``     (W -> K)  L3 goal-pointer logits over candidate waypoints.
      * ``role_head``     (W -> R)  L3 role-picker logits over {explorer, relay}
        (R = ``n_roles``; the Increment-1 labor-division head off the belief).
      * ``frontier_attn`` (the L4 "disperse" skill) biases the goal logits toward
        the most frontier-rich compass sector (``FrontierAttn``).
      * ``compass``       (the directional feature) ADDS a scale-free gather/explore
        navigation term to the belief z BEFORE the heads (``Compass``).
      * ``aux_head``      (W -> 1)  decentralized local-Fiedler λ̂₂ estimate (raw).
      * ``value_head``    (W -> 1)  per-agent baseline (diagnostic / IPPO fallback).

    ``__call__(obs, adj_off, *, key)`` -> ``(goal_logits (N,K), role_logits (N,R),
    value (N,), lambda2_hat (N,), z (N,W))``. The role head, ``frontier_attn`` AND
    ``compass`` are ALWAYS built (cheap, stable param surface) so the parameter tree
    is invariant to the ``role_picker`` / ``explorer_tool`` / ``compass`` knobs; each
    is only *used* when its knob is on. With ``compass == 'off'`` (default) the
    compass term is never added, so the belief z — and therefore EVERY head's output —
    is byte-identical to the pre-compass actor; with ``'on'`` the belief becomes
    ``z + compass(obs, z)`` before all heads (giving them an explicit directional
    cue). With ``explorer_tool == 'goal_head'`` (default) the frontier term is never
    added, so ``goal_logits`` is byte-identical to the pre-tool behaviour; with
    ``'frontier_attn'`` the goal logits become ``goal_head(z) + frontier_attn(obs,z)``.
    The role head is likewise sampled only when ``role_picker == 'expl_relay'``.
    The (post-compass) belief ``z`` is returned so the trainer can compute the degree
    regularizer.

    Recurrence (the ``recurrence`` axis): a per-agent ``gru`` (``eqx.nn.GRUCell``,
    W -> W) is ALWAYS built (stable param surface) but only USED when
    ``recurrence == 'recurrent'``. In that mode each step folds the (post-compass)
    belief ``z`` into a carried hidden state ``h`` — ``h_next = GRUCell(z, h)`` per
    agent (vmap over N) — and EVERY head (goal / role / λ̂₂ / value, incl. the
    frontier/compass tools' belief input) reads ``h_next`` INSTEAD of ``z``, so the
    agent remembers its own trajectory / coverage history across the episode. The
    incoming hidden ``h`` is threaded by the caller (the rollout scan carry; reset to
    zeros at each episode start, and recomputed under the current params along the
    trajectory in the PPO loss). With ``recurrence == 'feedforward'`` (default) the
    GRU is never traced, the heads read ``z`` exactly as before, and ``h_next`` is the
    zero passthrough — so the actor forward is BYTE-IDENTICAL to the pre-recurrence
    actor (the ``gru`` params just sit unused).

    Init note: both the ``compass`` AND the ``gru`` keys are derived via
    ``jax.random.fold_in`` (NOT by widening the ``split``) so the backbone / goal /
    role / frontier / aux / value keys are byte-IDENTICAL to the pre-recurrence actor
    — an actor built with ``compass='off'`` / ``recurrence='feedforward'`` is
    bit-for-bit the same network as before these modules existed.
    """
    backbone: Backbone
    goal_head: eqx.nn.Linear
    role_head: eqx.nn.Linear
    frontier_attn: FrontierAttn
    compass: Compass
    gru: eqx.nn.GRUCell
    aux_head: eqx.nn.Linear
    value_head: eqx.nn.Linear
    K: int = eqx.field(static=True)
    n_roles: int = eqx.field(static=True)
    explorer_tool: str = eqx.field(static=True)
    compass_on: bool = eqx.field(static=True)
    recurrent: bool = eqx.field(static=True)
    width: int = eqx.field(static=True)

    def __init__(self, in_ch: int, K: int, *, backbone_cfg, dropout: float, key,
                 n_roles: int = 2, explorer_tool: str = "goal_head",
                 compass: str = "off", recurrence: str = "feedforward"):
        kb, kg, kr, kf, ka, kv = jax.random.split(key, 6)
        # Derive the compass / gru keys by folding fixed constants into the ORIGINAL
        # key, so the six keys above are unchanged -> compass='off' /
        # recurrence='feedforward' is byte-identical to the pre-module actor
        # (split(key,7+) would have perturbed all six).
        kcomp = jax.random.fold_in(key, 0xC0)
        kgru = jax.random.fold_in(key, 0x60)
        self.backbone = Backbone(
            in_ch, backbone_cfg.width, backbone_cfg.depth, backbone_cfg.mp_rounds,
            backbone_cfg.agg, backbone_cfg.heads, backbone_cfg.norm, dropout, key=kb,
            message_content=getattr(backbone_cfg, "message_content", "learned"),
        )
        W = backbone_cfg.width
        self.goal_head = eqx.nn.Linear(W, K, key=kg)
        self.role_head = eqx.nn.Linear(W, n_roles, key=kr)
        self.frontier_attn = FrontierAttn(W, key=kf)
        self.compass = Compass(W, K, key=kcomp)
        # per-agent recurrent cell over the belief width (W -> W); always built so the
        # param tree is invariant to the recurrence knob, only USED when recurrent.
        self.gru = eqx.nn.GRUCell(W, W, key=kgru)
        self.aux_head = eqx.nn.Linear(W, 1, key=ka)
        self.value_head = eqx.nn.Linear(W, 1, key=kv)
        self.K = int(K)
        self.n_roles = int(n_roles)
        self.explorer_tool = str(explorer_tool)
        self.compass_on = (str(compass) == "on")
        self.recurrent = (str(recurrence) == "recurrent")
        self.width = int(W)

    def init_hidden(self, n: int) -> jax.Array:
        """Zero per-agent hidden state ``(N, W)`` — the episode-start carry for the
        recurrent path (and the inert passthrough returned by the feedforward path)."""
        return jnp.zeros((n, self.width), dtype=jnp.float32)

    def __call__(self, obs, adj_off, *, dist=None, h=None, key=None,
                 inference: bool = False):
        # ``dist`` (N,N) is the normalized sender->receiver distance the non-default
        # backbone message_content modes append to each comm message; the default
        # 'learned' backbone ignores it (so the forward is byte-identical to v0 whether
        # or not a dist is supplied — the caller passes it unconditionally for simplicity).
        z = self.backbone(obs, adj_off, dist=dist, key=key, inference=inference)   # (N,W)
        # Compass directional feature: ADD the gather/explore navigation term to the
        # belief BEFORE any head, so every head (goal / role / λ̂₂ / value) reads the
        # directional cue. `compass_on` is STATIC, so when off this branch never runs
        # and z — hence every head output below — is byte-identical to the pre-compass
        # actor (the `compass` params just sit unused, built for a stable param surface).
        if self.compass_on:
            z = z + self.compass(obs, z)                                # (N,W) z'
        # Recurrence: fold the (post-compass) belief into the carried hidden state and
        # let EVERY head read that hidden instead of z, so the agent remembers its own
        # trajectory across the 100-step episode. `recurrent` is STATIC, so when off
        # this branch is never traced, `feat` stays z, and `h_next` is the zero
        # passthrough -> the whole forward is byte-identical to the pre-recurrence actor.
        n = z.shape[0]
        h_in = self.init_hidden(n) if h is None else h                  # (N,W) carry
        if self.recurrent:
            h_next = jax.vmap(self.gru)(z, h_in)                        # (N,W) per-agent GRU
            feat = h_next                                              # heads read the hidden
        else:
            h_next = self.init_hidden(n)                              # inert zero passthrough
            feat = z                                                  # heads read the belief (v0)
        goal_logits = jax.vmap(self.goal_head)(feat)                   # (N,K)
        # L4 "disperse" tool: add the frontier-attention bias ONLY when selected.
        # `explorer_tool` is a STATIC string, so at "goal_head" this branch never
        # runs and `goal_logits` is byte-identical to the pre-tool actor; the
        # `frontier_attn` params just sit unused (built for a stable param surface).
        if self.explorer_tool == "frontier_attn":
            goal_logits = goal_logits + self.frontier_attn(obs, feat, self.K)  # (N,K)
        role_logits = jax.vmap(self.role_head)(feat)                   # (N,R)
        value = jax.vmap(self.value_head)(feat)[:, 0]                  # (N,)
        lambda2_hat = jax.vmap(self.aux_head)(feat)[:, 0]             # (N,)
        return goal_logits, role_logits, value, lambda2_hat, feat, h_next


# =============================================================================
# Centralized critic (CTDE, training only)
# =============================================================================


class Critic(eqx.Module):
    """Centralized critic over the team ``central_obs`` (Cg,H,W) -> value ()."""
    conv: list
    ln: eqx.nn.LayerNorm | None
    drop: eqx.nn.Dropout | None
    value_head: eqx.nn.Linear

    def __init__(self, in_ch: int, width: int, depth: int, norm: str,
                 dropout: float, *, key):
        kc, kv = jax.random.split(key)
        self.conv = _conv_stack(in_ch, width, depth, kc)
        self.ln = eqx.nn.LayerNorm(width) if norm == "layer" else None
        self.drop = eqx.nn.Dropout(dropout) if dropout > 0 else None
        self.value_head = eqx.nn.Linear(width, 1, key=kv)

    def __call__(self, central_obs, *, key=None, inference: bool = False):
        z = _encode(self.conv, central_obs)                            # (width,)
        if self.ln is not None:
            z = self.ln(z)
        if self.drop is not None:
            z = self.drop(z, key=key, inference=inference)
        return self.value_head(z)[0]                                   # ()
