"""Window an (E, T+1, ...) dataset into fixed-length-H samples + a train/val split.

Window order is **episode-major**: all windows of episode 0 (start index 0..T+1-H),
then all of episode 1, etc. A window starting at step `s` covers steps `s..s+H-1`;
its target is lambda2 at the LAST step `s+H-1`, so `y == lambda2[:, H-1:].ravel()`.
`make_adj_windows` produces the adjacency windows in the SAME order.
"""
import numpy as np


def _slide(arr, H):
    """(E, T1, *rest) -> (S, H, *rest) with S = E*(T1-H+1), episode-major order.

    For each start s in [0, T1-H], take arr[:, s:s+H] -> (E, H, *rest); stack over s
    on a new axis after the episode axis, then flatten (E, n_windows) -> S.
    """
    E, T1 = arr.shape[0], arr.shape[1]
    n_win = T1 - H + 1
    if n_win < 1:
        raise ValueError(f"H={H} too large for sequence length T+1={T1}")
    # (n_win, E, H, *rest)
    wins = np.stack([arr[:, s:s + H] for s in range(n_win)], axis=0)
    # -> (E, n_win, H, *rest) so flattening (E, n_win) gives episode-major order
    wins = np.moveaxis(wins, 0, 1)
    return wins.reshape((E * n_win, H) + arr.shape[2:])


def make_windows(features, lambda2, H):
    """features (E,T+1,N,6), lambda2 (E,T+1) -> X_node (S,H,N,6) f32, y (S,) f32.

    S = E*(T+1-H+1); target = lambda2 at each window's LAST step.
    """
    features = np.asarray(features, np.float32)
    lambda2 = np.asarray(lambda2, np.float32)
    X_node = _slide(features, H).astype(np.float32)
    # target = last-step lambda2 of every window == lambda2[:, H-1:] (episode-major)
    y = lambda2[:, H - 1:].reshape(-1).astype(np.float32)
    return X_node, y


def make_adj_windows(adjacency, H):
    """adjacency (E,T+1,N,N) bool -> X_adj (S,H,N,N) bool, SAME order as make_windows."""
    adjacency = np.asarray(adjacency, bool)
    return _slide(adjacency, H).astype(bool)


def make_pos_windows(positions, H):
    """positions (E,T+1,N,2) -> X_pos (S,H,N,2) f32, SAME order as make_windows."""
    positions = np.asarray(positions, np.float32)
    return _slide(positions, H).astype(np.float32)


def pad_to(arr, N_max, axis):
    """Zero-pad `arr` along `axis` from its current size up to `N_max` (>= current)."""
    arr = np.asarray(arr)
    cur = arr.shape[axis]
    if cur > N_max:
        raise ValueError(f"axis {axis} size {cur} exceeds N_max {N_max}")
    if cur == N_max:
        return arr
    pad = [(0, 0)] * arr.ndim
    pad[axis] = (0, N_max - cur)
    return np.pad(arr, pad, mode="constant", constant_values=0)


def pad_batch(groups, N_max):
    """Pool a list of single-N window-sets into one padded multi-N batch.

    Each `group` is a dict with:
        X_node (S_g,H,N_g,6), X_adj (S_g,H,N_g,N_g), X_pos (S_g,H,N_g,2), y (S_g,)
    (all with the SAME H and N_g <= N_max). Returns a single dict with everything
    padded to N_max on the node axes and concatenated over groups, plus a boolean
    `node_mask (S,N_max)` that is True only for the real (first N_g) nodes of each row.
    """
    Xn, Xa, Xp, ys, masks = [], [], [], [], []
    for g in groups:
        x_node = np.asarray(g["X_node"], np.float32)
        x_adj = np.asarray(g["X_adj"], bool)
        x_pos = np.asarray(g["X_pos"], np.float32)
        y = np.asarray(g["y"], np.float32)
        S_g, H, N_g = x_node.shape[0], x_node.shape[1], x_node.shape[2]

        Xn.append(pad_to(x_node, N_max, axis=2))                       # (S,H,Nmax,6)
        Xa.append(pad_to(pad_to(x_adj, N_max, axis=2), N_max, axis=3)) # (S,H,Nmax,Nmax)
        Xp.append(pad_to(x_pos, N_max, axis=2))                        # (S,H,Nmax,2)
        ys.append(y)

        m = np.zeros((S_g, N_max), dtype=bool)
        m[:, :N_g] = True
        masks.append(m)

    return {
        "X_node": np.concatenate(Xn, axis=0).astype(np.float32),
        "X_adj": np.concatenate(Xa, axis=0).astype(bool),
        "X_pos": np.concatenate(Xp, axis=0).astype(np.float32),
        "y": np.concatenate(ys, axis=0).astype(np.float32),
        "node_mask": np.concatenate(masks, axis=0).astype(bool),
    }


def train_val_split(n, val_frac=0.2, seed=0):
    """Random permutation split of range(n) -> (train_idx, val_idx) int arrays."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = int(round(n * val_frac))
    val_idx = np.sort(perm[:n_val]).astype(np.int64)
    train_idx = np.sort(perm[n_val:]).astype(np.int64)
    return train_idx, val_idx
