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
# coordinate is ever fed in, only the displacement geometry it induces).
_CH_KNOWN = 0
_CH_OWN_POS = 1


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


class MPLayer(eqx.Module):
    """One message-passing round over the comm graph.

    Each node updates from (its own feature) + (aggregated neighbour messages).
    The aggregator is configurable and ALWAYS normalized / size-invariant:

      * "mean" — degree-normalized average of neighbour messages.
      * "max"  — elementwise max over neighbours (default; magnitude-invariant).
      * "multihead" — softmax attention over neighbours (``heads`` heads), the
        attention weights sum to 1 so the readout is agent-count-invariant.

    A boolean ``adj`` (N,N, self-loops removed by the caller for neighbour msgs)
    selects who is in range. ``__call__(feats, adj_off)`` -> (N, width).
    """
    msg: eqx.nn.Linear          # neighbour-message transform
    upd: eqx.nn.Linear          # node update from [self || aggregated]
    q: eqx.nn.Linear            # attention query (multihead)
    k: eqx.nn.Linear            # attention key   (multihead)
    agg: str = eqx.field(static=True)
    heads: int = eqx.field(static=True)
    width: int = eqx.field(static=True)

    def __init__(self, width: int, agg: str, heads: int, *, key):
        km, ku, kq, kk = jax.random.split(key, 4)
        self.msg = eqx.nn.Linear(width, width, key=km)
        self.upd = eqx.nn.Linear(2 * width, width, key=ku)
        self.q = eqx.nn.Linear(width, width, key=kq)
        self.k = eqx.nn.Linear(width, width, key=kk)
        self.agg = agg
        self.heads = int(heads)
        self.width = int(width)

    def __call__(self, feats, adj_off):
        # feats (N,W); adj_off (N,N) bool with diagonal already cleared.
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

        cat = jnp.concatenate([feats, agg], axis=-1)                    # (N,2W)
        out = jax.vmap(self.upd)(cat)                                   # (N,W)
        return jax.nn.relu(out)


class Backbone(eqx.Module):
    """LPAC backbone: per-agent CNN -> GAP -> feature, then ``mp_rounds`` of GNN
    message passing over the comm graph -> per-agent belief ``z_i`` (N, width).

    ``__call__(obs, adj_off, *, key)`` with ``obs`` (N,C,H,W) and ``adj_off``
    (N,N) bool (in-range neighbours, diagonal cleared). Optional LayerNorm +
    dropout on the belief. Returns ``z`` (N, width).
    """
    conv: list
    mp: list
    ln: eqx.nn.LayerNorm | None
    drop: eqx.nn.Dropout | None
    width: int = eqx.field(static=True)

    def __init__(self, in_ch: int, width: int, depth: int, mp_rounds: int,
                 agg: str, heads: int, norm: str, dropout: float, *, key):
        kc, kmp = jax.random.split(key)
        self.conv = _conv_stack(in_ch, width, depth, kc)
        mp_keys = jax.random.split(kmp, max(mp_rounds, 1))
        self.mp = [MPLayer(width, agg, heads, key=mp_keys[i]) for i in range(mp_rounds)]
        self.ln = eqx.nn.LayerNorm(width) if norm == "layer" else None
        self.drop = eqx.nn.Dropout(dropout) if dropout > 0 else None
        self.width = int(width)

    def __call__(self, obs, adj_off, *, key=None, inference: bool = False):
        feats = jax.vmap(lambda o: _encode(self.conv, o))(obs)         # (N,W)
        z = feats
        for layer in self.mp:
            z = layer(z, adj_off)                                      # (N,W)
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
# Actor: backbone + goal / λ̂₂ / value heads (decentralized)
# =============================================================================


class Actor(eqx.Module):
    """Decentralized per-agent actor: LPAC backbone -> belief z_i -> four heads
    (+ the frontier-attention explorer tool).

      * ``goal_head``     (W -> K)  L3 goal-pointer logits over candidate waypoints.
      * ``role_head``     (W -> R)  L3 role-picker logits over {explorer, relay}
        (R = ``n_roles``; the Increment-1 labor-division head off the belief).
      * ``frontier_attn`` (the L4 "disperse" skill) biases the goal logits toward
        the most frontier-rich compass sector (``FrontierAttn``).
      * ``aux_head``      (W -> 1)  decentralized local-Fiedler λ̂₂ estimate (raw).
      * ``value_head``    (W -> 1)  per-agent baseline (diagnostic / IPPO fallback).

    ``__call__(obs, adj_off, *, key)`` -> ``(goal_logits (N,K), role_logits (N,R),
    value (N,), lambda2_hat (N,), z (N,W))``. The role head AND ``frontier_attn`` are
    ALWAYS built (cheap, stable param surface) so the parameter tree is invariant to
    the ``role_picker`` / ``explorer_tool`` knobs; each is only *used* when its knob
    is on. With ``explorer_tool == 'goal_head'`` (default) the frontier term is never
    added, so ``goal_logits`` is byte-identical to the pre-tool behaviour; with
    ``'frontier_attn'`` the goal logits become ``goal_head(z) + frontier_attn(obs,z)``.
    The role head is likewise sampled only when ``role_picker == 'expl_relay'``.
    The belief ``z`` is returned so the trainer can compute the degree regularizer.
    """
    backbone: Backbone
    goal_head: eqx.nn.Linear
    role_head: eqx.nn.Linear
    frontier_attn: FrontierAttn
    aux_head: eqx.nn.Linear
    value_head: eqx.nn.Linear
    K: int = eqx.field(static=True)
    n_roles: int = eqx.field(static=True)
    explorer_tool: str = eqx.field(static=True)

    def __init__(self, in_ch: int, K: int, *, backbone_cfg, dropout: float, key,
                 n_roles: int = 2, explorer_tool: str = "goal_head"):
        kb, kg, kr, kf, ka, kv = jax.random.split(key, 6)
        self.backbone = Backbone(
            in_ch, backbone_cfg.width, backbone_cfg.depth, backbone_cfg.mp_rounds,
            backbone_cfg.agg, backbone_cfg.heads, backbone_cfg.norm, dropout, key=kb,
        )
        W = backbone_cfg.width
        self.goal_head = eqx.nn.Linear(W, K, key=kg)
        self.role_head = eqx.nn.Linear(W, n_roles, key=kr)
        self.frontier_attn = FrontierAttn(W, key=kf)
        self.aux_head = eqx.nn.Linear(W, 1, key=ka)
        self.value_head = eqx.nn.Linear(W, 1, key=kv)
        self.K = int(K)
        self.n_roles = int(n_roles)
        self.explorer_tool = str(explorer_tool)

    def __call__(self, obs, adj_off, *, key=None, inference: bool = False):
        z = self.backbone(obs, adj_off, key=key, inference=inference)   # (N,W)
        goal_logits = jax.vmap(self.goal_head)(z)                       # (N,K)
        # L4 "disperse" tool: add the frontier-attention bias ONLY when selected.
        # `explorer_tool` is a STATIC string, so at "goal_head" this branch never
        # runs and `goal_logits` is byte-identical to the pre-tool actor; the
        # `frontier_attn` params just sit unused (built for a stable param surface).
        if self.explorer_tool == "frontier_attn":
            goal_logits = goal_logits + self.frontier_attn(obs, z, self.K)  # (N,K)
        role_logits = jax.vmap(self.role_head)(z)                       # (N,R)
        value = jax.vmap(self.value_head)(z)[:, 0]                      # (N,)
        lambda2_hat = jax.vmap(self.aux_head)(z)[:, 0]                  # (N,)
        return goal_logits, role_logits, value, lambda2_hat, z


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
