"""Agent-identity augmentation: give each agent a way to distinguish itself.

The connectivity-aware estimators are deliberately permutation-equivariant: identical
local inputs produce identical outputs, so two agents in symmetric positions are
indistinguishable. This module appends per-agent ID features to the node-feature window
tensor `X_node (S,H,N,6)` to test whether breaking that symmetry helps lambda2 estimation.

`augment_with_id(X_node, node_mask, *, id_mode, id_dim=4, seed)` -> (X_node_aug, in_size):

  * "none"   : no-op. Returns X_node unchanged, in_size = 6. (BACKWARD-COMPATIBLE default.)
  * "random" : append `id_dim` random features per agent. The tag is CONSTANT across the
               H window for a given (sample, agent) but drawn independently per agent and
               per sample -- a permutation-equivariant symmetry-breaking tag (no agent has
               a privileged value; only the *contrast* between agents carries signal).
               Padded agents (node_mask False) get 0. in_size = 6 + id_dim.
  * "index"  : append ONE feature = the agent's index along the node axis, normalized to
               [0,1] by N_max (= the node-axis size). This is the RAW-ID variant (a fixed,
               position-dependent label) -- expected to HURT size-transfer because index
               k/N_max means different things at different N. Padded agents get 0.
               in_size = 7 (regardless of id_dim).

The appended features always occupy the LAST axis after the original 6, and padded agents
are zeroed in every mode (padded nodes never enter the masked loss/metrics anyway).
"""
import numpy as np


def augment_with_id(X_node, node_mask, *, id_mode, id_dim=4, seed=0):
    """Append per-agent ID features to X_node (S,H,N,6). Returns (X_node_aug, in_size).

    Args:
        X_node:    float array (S, H, N, 6) -- windowed node features.
        node_mask: bool array (S, N) -- True for real agents, False for padded ones.
        id_mode:   "none" | "random" | "index".
        id_dim:    number of random features for id_mode="random" (ignored otherwise).
        seed:      RNG seed for id_mode="random" (deterministic in this seed).
    """
    X_node = np.asarray(X_node, np.float32)
    if id_mode == "none":
        # No-op; return a copy so callers never accidentally alias the input.
        return X_node.copy(), int(X_node.shape[-1])

    S, H, N, F = X_node.shape
    mask = np.asarray(node_mask, bool)                          # (S, N)
    if mask.shape != (S, N):
        raise ValueError(f"node_mask shape {mask.shape} != (S,N)=({S},{N})")
    mask_shn = np.broadcast_to(mask[:, None, :], (S, H, N))     # (S,H,N)

    if id_mode == "random":
        d = int(id_dim)
        rng = np.random.default_rng(int(seed))
        # one random tag per (sample, agent), then broadcast across the H window so the
        # tag is constant in time for a given (sample, agent).
        tag_sn = rng.standard_normal((S, N, d)).astype(np.float32)        # (S,N,d)
        tag = np.broadcast_to(tag_sn[:, None, :, :], (S, H, N, d)).copy()  # (S,H,N,d)
        tag *= mask_shn[..., None]                              # zero padded agents
        out = np.concatenate([X_node, tag], axis=-1).astype(np.float32)
        return out, int(F + d)

    if id_mode == "index":
        # one feature = node index / N_max in [0,1], constant across S and H.
        idx = (np.arange(N, dtype=np.float32) / float(N))      # (N,)
        feat = np.broadcast_to(idx[None, None, :], (S, H, N)).copy()  # (S,H,N)
        feat *= mask_shn.astype(np.float32)                    # zero padded agents
        out = np.concatenate([X_node, feat[..., None]], axis=-1).astype(np.float32)
        return out, int(F + 1)

    raise ValueError(f"unknown id_mode {id_mode!r} (expected 'none', 'random', or 'index')")
