"""Ground-truth Fiedler value (algebraic connectivity) from a comm adjacency."""
import jax.numpy as jnp
import zymera.metrics as zmetrics

CONNECTED_TAU = 1e-3  # lambda2 > tau  <=>  connected


def potential_adjacency(positions, comm_r):
    """(N,2) int positions -> (N,N) bool potential comm graph (in-range, diag True)."""
    return zmetrics.adjacency(positions, radius=comm_r)


def _laplacian(adj):
    """Combinatorial Laplacian L = D - A with self-loops stripped."""
    a = adj.astype(jnp.float32)
    a = a - jnp.diag(jnp.diag(a))          # strip self-loops
    deg = a.sum(-1)
    return jnp.diag(deg) - a


def true_lambda2(adj):
    """Second-smallest eigenvalue of the Laplacian; clamped at 0."""
    evals = jnp.linalg.eigvalsh(_laplacian(adj))
    return jnp.maximum(evals[1], 0.0)


def connected_flag(adj, tau: float = CONNECTED_TAU):
    """True iff lambda2 > tau."""
    return true_lambda2(adj) > tau
