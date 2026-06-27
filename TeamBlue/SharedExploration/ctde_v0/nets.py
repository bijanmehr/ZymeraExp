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
# Actor: backbone + goal / λ̂₂ / value heads (decentralized)
# =============================================================================


class Actor(eqx.Module):
    """Decentralized per-agent actor: LPAC backbone -> belief z_i -> three heads.

      * ``goal_head``  (W -> K)  L3 goal-pointer logits over candidate waypoints.
      * ``aux_head``   (W -> 1)  decentralized local-Fiedler λ̂₂ estimate (raw).
      * ``value_head`` (W -> 1)  per-agent baseline (diagnostic / IPPO fallback).

    ``__call__(obs, adj_off, *, key)`` -> ``(goal_logits (N,K), value (N,),
    lambda2_hat (N,), z (N,W))``. The belief ``z`` is returned so the trainer can
    compute the degree regularizer / inspect the KB.
    """
    backbone: Backbone
    goal_head: eqx.nn.Linear
    aux_head: eqx.nn.Linear
    value_head: eqx.nn.Linear
    K: int = eqx.field(static=True)

    def __init__(self, in_ch: int, K: int, *, backbone_cfg, dropout: float, key):
        kb, kg, ka, kv = jax.random.split(key, 4)
        self.backbone = Backbone(
            in_ch, backbone_cfg.width, backbone_cfg.depth, backbone_cfg.mp_rounds,
            backbone_cfg.agg, backbone_cfg.heads, backbone_cfg.norm, dropout, key=kb,
        )
        W = backbone_cfg.width
        self.goal_head = eqx.nn.Linear(W, K, key=kg)
        self.aux_head = eqx.nn.Linear(W, 1, key=ka)
        self.value_head = eqx.nn.Linear(W, 1, key=kv)
        self.K = int(K)

    def __call__(self, obs, adj_off, *, key=None, inference: bool = False):
        z = self.backbone(obs, adj_off, key=key, inference=inference)   # (N,W)
        goal_logits = jax.vmap(self.goal_head)(z)                       # (N,K)
        value = jax.vmap(self.value_head)(z)[:, 0]                      # (N,)
        lambda2_hat = jax.vmap(self.aux_head)(z)[:, 0]                  # (N,)
        return goal_logits, value, lambda2_hat, z


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
