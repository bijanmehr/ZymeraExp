"""Signal-strength (path-loss) overlay: a continuous per-edge link-quality weight.

The learned estimators see a BINARY adjacency, which throws away *how strong* each comm
link is. A real radio link does not flip from perfect to gone at exactly comm_r: its
strength falls off smoothly with distance. This module models that with a continuous
per-edge path-loss weight `s in [0,1]` -- ~1 when a neighbor is right on top of you, ->0
as the link stretches to comm range -- and uses it three ways:

  * `signal_weighted_adj`  : replace each comm edge's binary 1.0 with its `s`, so the
    SAME message-passing ops (`messages.aggregate`) become **soft-weighted** automatically
    (mean = (a@m)/deg with a=weights, gcn/gated/... likewise). This is the soft-weighted
    adjacency aligned with the soft Laplacian whose Fiedler value is lambda2 -- a weak
    boundary link contributes little, exactly as it does to the true algebraic connectivity.
  * the same per-edge `s` can ride INSIDE the messages (via `messages.content="margin"`,
    which already carries `dist/comm_r` per edge -- a monotone transform of `s`).
  * `augment_with_signal` : give each agent ONE node-level link-quality feature -- its mean
    in-range-neighbor signal strength -- so the readout sees how strong its links are.

The falloff is a Gaussian path-loss `s = exp(-SHARP * (dist/comm_r)**2)` with `SHARP=3.0`:
exactly 1 at distance 0 and ~0.05 (= e^-3) at distance = comm_r (a link on the verge of
dropping). It is the complementary view of `margin = dist/comm_r` (~1 = about to break):
`s` is ~1 = rock solid, ->0 = about to break.
"""
import jax.numpy as jnp
import numpy as np

# Gaussian path-loss sharpness: s = exp(-SHARP * (dist/comm_r)^2).
# SHARP=3.0 -> s(0)=1, s(comm_r)=e^-3 ~= 0.0498 (a link at comm_r barely carries).
SHARP = 3.0


def signal_strength(dist_over_r):
    """Path-loss link quality `s in [0,1]` from normalized distance `dist/comm_r`.

    `s = exp(-SHARP * x**2)` with `x = dist/comm_r`: ~1 at x=0 (close neighbor), ~0.05 at
    x=1 (= comm range). Smooth, monotone-decreasing, vectorized (pure jnp; any shape).
    """
    x = jnp.asarray(dist_over_r, jnp.float32)
    return jnp.exp(-SHARP * x * x)


def _pairwise_dist(pos):
    """positions (...,N,2) -> pairwise distances (...,N,N), dist[...,i,j] = ||p_i - p_j||."""
    pos = jnp.asarray(pos, jnp.float32)
    diff = pos[..., :, None, :] - pos[..., None, :, :]        # (...,N,N,2)  i - j
    return jnp.sqrt((diff ** 2).sum(-1) + 1e-12)             # (...,N,N)


def signal_weighted_adj(X_adj_bool, X_pos, comm_r):
    """Replace each existing comm edge's 1.0 with its path-loss signal strength.

    Args:
        X_adj_bool: bool/numeric (...,N,N) adjacency (self-loops allowed; preserved).
        X_pos:      (...,N,2) agent positions (same leading shape as X_adj).
        comm_r:     communication range used to normalize edge distances.

    Returns:
        FLOAT (...,N,N) adjacency where edge (i,j) = signal_strength(dist_ij/comm_r) for
        existing edges and 0 for non-edges. The binary SUPPORT is preserved exactly (a
        weight is nonzero iff the bool edge was True) -- existing comm edges sit at
        distance <= comm_r, where `s` is strictly positive. Self-loops are kept as-is
        (they are stripped downstream by `messages._a_hat`, same as the bool case).
    """
    adj = jnp.asarray(X_adj_bool)
    mask = adj.astype(jnp.float32) > 0.0                      # (...,N,N) existing edges
    dist = _pairwise_dist(X_pos)                              # (...,N,N)
    s = signal_strength(dist / float(comm_r))                # (...,N,N) in (0,1]
    weighted = jnp.where(mask, s, 0.0)
    return weighted.astype(jnp.float32)


def augment_with_signal(X_node, X_adj_bool, X_pos, comm_r):
    """Append the per-agent link-quality node feature to X_node (S,H,N,F).

    The appended feature is the agent's MEAN in-range-neighbor signal strength at the
    window's LAST step (matching the lambda2 target, which is read at the last step):
    ~1 when all links are strong, small when links are stretched toward comm_r, and 0 for
    an isolated or padded agent (no in-range neighbor). Self-loops never count.

    Args:
        X_node:      float (S,H,N,F) windowed node features.
        X_adj_bool:  bool/float (S,H,N,N) windowed adjacency (self-loops allowed).
        X_pos:       float (S,H,N,2) windowed agent positions.
        comm_r:      communication range used to normalize edge distances.

    Returns:
        (X_node_aug (S,H,N,F+1) float32, in_size = F+1).
    """
    X_node = np.asarray(X_node, np.float32)
    X_adj = np.asarray(X_adj_bool)
    X_pos = np.asarray(X_pos, np.float32)
    S, H, N, F = X_node.shape
    comm_r = float(comm_r)

    # --- LAST-step topology (freshest; matches the window's lambda2 target) -------------
    adj_last = X_adj[:, H - 1].astype(bool)                  # (S,N,N)
    pos_last = X_pos[:, H - 1]                               # (S,N,2)

    # strip self-loops: a node is never its own comm neighbor
    eye = np.eye(N, dtype=bool)[None, :, :]                  # (1,N,N)
    nbr = adj_last & ~eye                                    # (S,N,N) real comm edges

    # pairwise signal strength at the last step: s = exp(-SHARP * (dist/comm_r)^2)
    diff = pos_last[:, :, None, :] - pos_last[:, None, :, :]      # (S,N,N,2)
    dist = np.sqrt((diff ** 2).sum(-1) + 1e-12)                  # (S,N,N)
    s = np.exp(-SHARP * (dist / comm_r) ** 2)                    # (S,N,N) in (0,1]

    # mean signal strength over each agent's REAL neighbors; isolated agents -> 0.
    deg = nbr.sum(axis=2)                                        # (S,N) neighbor count
    s_sum = np.where(nbr, s, 0.0).sum(axis=2)                    # (S,N) sum over neighbors
    feat_sn = np.where(deg > 0, s_sum / np.maximum(deg, 1), 0.0).astype(np.float32)  # (S,N)

    # broadcast the per-(sample,agent) scalar across the H window (constant in time).
    feat = np.broadcast_to(feat_sn[:, None, :, None], (S, H, N, 1)).astype(np.float32)
    out = np.concatenate([X_node, feat], axis=-1).astype(np.float32)
    return out, int(F + 1)
