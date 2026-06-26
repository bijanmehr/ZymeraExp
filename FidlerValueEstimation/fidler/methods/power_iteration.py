"""Decentralized power-iteration estimate of lambda2 (Yang-style: deflate + diffuse + Rayleigh)."""
import jax
import jax.numpy as jnp
from fidler.fiedler import _laplacian


def estimate(adj, n_rounds: int, eps: float = 0.1, seed: int = 0):
    """adj (N,N) bool. Returns scalar lambda2 estimate after n_rounds local rounds."""
    lap = _laplacian(adj)                                   # (N,N)
    x = jax.random.normal(jax.random.PRNGKey(seed), (adj.shape[0],))

    def body(x, _):
        x = x - jnp.mean(x)                                 # deflate constant (consensus mean)
        x = x - eps * (lap @ x)                             # power-iter on (I - eps L)
        x = x / (jnp.linalg.norm(x) + 1e-12)
        return x, None

    x, _ = jax.lax.scan(body, x, None, length=n_rounds)
    return (x @ lap @ x) / (x @ x + 1e-12)                  # Rayleigh quotient
