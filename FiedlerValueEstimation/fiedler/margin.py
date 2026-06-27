"""Connectivity-margin node-feature augmentation: an "about-to-disconnect" signal.

The learned estimators see a BINARY adjacency, which throws away *how close* each comm
edge is to breaking. A boundary edge at ~comm_r barely contributes to algebraic
connectivity (it lowers lambda2), but a binary `1` looks identical to a rock-solid edge
at distance 0. This module restores the discarded soft-edge signal as ONE per-agent node
feature: the agent's MAX in-range-neighbor distance / comm_r -- its *closest-to-breaking*
link ("weakest link"), where ~1 means an edge is on the verge of dropping and 0 means the
agent is isolated (no in-range neighbor).

`augment_with_margin(X_node, X_adj, X_pos, comm_r)` -> (X_node_aug, in_size):

  * Appends exactly ONE feature on the LAST node axis (so it lands after the original F
    and after any agent-ID features), constant across the H window: it is computed from
    the window's LAST step (h = H-1) -- the freshest topology, matching the target
    lambda2 which is also read at the window's last step.
  * Self-loops on the adjacency diagonal are stripped (a self-loop is not a comm link).
  * Isolated agents (no in-range neighbor at the last step) get 0. Padded agents -- which
    `pad_batch` leaves with an all-zero adjacency row -- therefore read as isolated -> 0.
  * in_size = (input feature width) + 1.

The single scalar recovers the soft-weighted-Laplacian signal (weak boundary edges lower
lambda2) that the binary adjacency discards; the companion `messages.content="margin"`
puts the same per-edge `dist/comm_r` margin inside the messages.
"""
import numpy as np


def augment_with_margin(X_node, X_adj, X_pos, comm_r):
    """Append the per-agent connectivity-margin feature to X_node (S,H,N,F).

    Args:
        X_node: float array (S, H, N, F) -- windowed node features.
        X_adj:  bool/float array (S, H, N, N) -- windowed adjacency (self-loops allowed).
        X_pos:  float array (S, H, N, 2) -- windowed integer/float agent positions.
        comm_r: communication range (scalar) used to normalize the edge distances.

    Returns:
        (X_node_aug (S,H,N,F+1) float32, in_size = F+1).
    """
    X_node = np.asarray(X_node, np.float32)
    X_adj = np.asarray(X_adj)
    X_pos = np.asarray(X_pos, np.float32)
    S, H, N, F = X_node.shape
    comm_r = float(comm_r)

    # --- LAST-step topology (freshest; matches the window's lambda2 target) -------------
    adj_last = X_adj[:, H - 1].astype(bool)               # (S, N, N)
    pos_last = X_pos[:, H - 1]                             # (S, N, 2)

    # strip self-loops: a node is never its own comm neighbor
    eye = np.eye(N, dtype=bool)[None, :, :]               # (1, N, N)
    nbr = adj_last & ~eye                                 # (S, N, N) real comm edges

    # pairwise distances at the last step: (S, N, N)
    diff = pos_last[:, :, None, :] - pos_last[:, None, :, :]   # (S, N, N, 2)
    dist = np.sqrt((diff ** 2).sum(-1) + 1e-12)               # (S, N, N)
    margin = dist / comm_r                                     # dist normalized by comm_r

    # max margin over each agent's real neighbors (the closest-to-breaking link).
    # mask non-neighbors with -inf so they never win the max; agents with no neighbor -> 0.
    masked = np.where(nbr, margin, -np.inf)                   # (S, N, N)
    feat_sn = masked.max(axis=2)                             # (S, N)
    feat_sn = np.where(np.isfinite(feat_sn), feat_sn, 0.0).astype(np.float32)  # isolated -> 0

    # broadcast the per-(sample,agent) scalar across the H window (constant in time).
    feat = np.broadcast_to(feat_sn[:, None, :, None], (S, H, N, 1)).astype(np.float32)
    out = np.concatenate([X_node, feat], axis=-1).astype(np.float32)
    return out, int(F + 1)
