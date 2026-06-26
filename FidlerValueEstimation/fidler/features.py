"""Per-agent local features (6 instantaneous dims) from positions + adjacency."""
import jax.numpy as jnp


def node_features(positions, adj, comm_r):
    """positions (N,2), adj (N,N) bool (diag True). Returns (N,6) float32:
    [degree, log1p(degree), mean_nbr_dist, std_nbr_dist, mean_nbr_deg, std_nbr_deg].
    Distances normalized by comm_r; empty-neighbor stats are 0 (safe)."""
    a = adj.astype(jnp.float32)
    a = a - jnp.diag(jnp.diag(a))                      # strip self-loops
    deg = a.sum(-1)                                    # (N,)

    pos = positions.astype(jnp.float32)
    diff = pos[:, None, :] - pos[None, :, :]           # (N,N,2)
    dist = jnp.sqrt((diff ** 2).sum(-1)) / float(comm_r)  # (N,N) normalized
    cnt = jnp.maximum(deg, 1.0)                        # avoid /0

    mean_d = (a * dist).sum(-1) / cnt
    var_d = (a * (dist - mean_d[:, None]) ** 2).sum(-1) / cnt
    mean_deg = (a * deg[None, :]).sum(-1) / cnt
    var_deg = (a * (deg[None, :] - mean_deg[:, None]) ** 2).sum(-1) / cnt

    has = (deg > 0).astype(jnp.float32)                # zero stats when isolated
    return jnp.stack([
        deg,
        jnp.log1p(deg),
        mean_d * has,
        jnp.sqrt(var_d) * has,
        mean_deg * has,
        jnp.sqrt(var_deg) * has,
    ], axis=-1).astype(jnp.float32)
