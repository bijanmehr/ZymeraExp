"""Dynamic-features node augmentation: the "neighbors moving away / degree crashing" signals.

The static node features ([degree, log-degree, mean/std nbr dist, mean/std nbr degree]) are a
SNAPSHOT -- they say nothing about whether the local neighborhood is *tightening or loosening*.
But lambda2 (global algebraic connectivity) crashes precisely when neighbors recede and degree
drops. This overlay adds the per-agent TEMPORAL TRENDS across the H window -- information the
snapshot lacks:

  1. d_degree : degree[h] - degree[h-1]        (losing neighbors -> negative -> lambda2 falling)
  2. d_nbrdist: mean-nbr-dist[h] - [h-1]        (neighbors receding -> positive)
  3. speed    : ||pos[h] - pos[h-1]|| / comm_r  (own motion magnitude)

Three per-agent features, TIME-VARYING across the window (the first step's diff is 0). The
per-node GRU consumes the sequence, so these explicit derivatives make the "is connectivity
crashing?" trend directly available rather than hoping the GRU re-derives it from raw snapshots.

`augment_with_dynamics(X_node, X_adj, X_pos, comm_r)` -> (X_node_aug (S,H,N,F+3), in_size=F+3).
The original node features are never reordered (overlays only append), so mean-nbr-dist is read
from its fixed column index 2.
"""
import numpy as np

_MEAN_NBR_DIST_COL = 2          # index of mean_nbr_dist in the base node_features vector


def augment_with_dynamics(X_node, X_adj, X_pos, comm_r):
    X_node = np.asarray(X_node, np.float32)
    X_adj = np.asarray(X_adj)
    X_pos = np.asarray(X_pos, np.float32)
    S, H, N, F = X_node.shape
    comm_r = float(comm_r)

    # degree per (S,H,N) from the (binary) adjacency, self-loops stripped
    a = (X_adj > 0).astype(np.float32)
    eye = np.eye(N, dtype=np.float32)[None, None, :, :]
    deg = (a * (1.0 - eye)).sum(-1)                       # (S,H,N)

    d_deg = np.zeros_like(deg)
    d_deg[:, 1:] = deg[:, 1:] - deg[:, :-1]               # rate neighbors gained(+)/lost(-)

    mnd = X_node[..., _MEAN_NBR_DIST_COL]                  # (S,H,N) mean neighbor distance
    d_mnd = np.zeros_like(mnd)
    d_mnd[:, 1:] = mnd[:, 1:] - mnd[:, :-1]               # neighbors approaching(-)/receding(+)

    speed = np.zeros((S, H, N), np.float32)
    step = X_pos[:, 1:] - X_pos[:, :-1]                   # (S,H-1,N,2)
    speed[:, 1:] = np.sqrt((step ** 2).sum(-1) + 1e-12) / comm_r

    feat = np.stack([d_deg, d_mnd, speed], axis=-1).astype(np.float32)    # (S,H,N,3)
    out = np.concatenate([X_node, feat], axis=-1).astype(np.float32)
    return out, int(F + 3)
