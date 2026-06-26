"""Equinox learned lambda2 estimators (size-invariant).

Both models map (x_node (H,N,6), x_adj (H,N,N)) -> (N,) predicted **log** lambda2,
one estimate per agent. They are size-invariant: every learned block acts per-node
(or per-edge via a normalized-mean aggregation `agg=(A_hat@z)/deg`, NEVER sum — sum
scales with N/degree and destroys size-transfer), so the SAME params run at any N.

- GRUEstimator: per-agent GRU over the H-step own-feature sequence (ignores x_adj).
- GCRNEstimator: spatio-temporal; lax.scan over H steps, one message-passing round
  per step (normalized mean aggregation) + a per-node GRU temporal update.
"""
import equinox as eqx
import jax
import jax.numpy as jnp


class GRUEstimator(eqx.Module):
    """Per-agent GRU over its own (H,6) feature history -> log-lambda2. Ignores x_adj."""
    cell: eqx.nn.GRUCell
    head: eqx.nn.MLP
    hidden: int = eqx.field(static=True)

    def __init__(self, in_size: int = 6, hidden: int = 64, *, key):
        kc, kh = jax.random.split(key)
        self.hidden = hidden
        self.cell = eqx.nn.GRUCell(in_size, hidden, key=kc)
        self.head = eqx.nn.MLP(hidden, 1, width_size=hidden, depth=1, key=kh)

    def __call__(self, x_node, x_adj):
        # x_node: (H, N, 6); x_adj ignored.
        H, N, _ = x_node.shape

        def per_agent(seq):  # seq: (H, 6)
            h0 = jnp.zeros((self.hidden,))

            def step(h, x):
                return self.cell(x, h), None

            h, _ = jax.lax.scan(step, h0, seq)
            return self.head(h)[0]  # scalar log-lambda2

        seqs = jnp.transpose(x_node, (1, 0, 2))            # (N, H, 6)
        return jax.vmap(per_agent)(seqs)                   # (N,)


class GCRNEstimator(eqx.Module):
    """Spatio-temporal: per step encode -> 1 MP round (normalized mean agg) -> per-node GRU."""
    encoder: eqx.nn.Linear
    message: eqx.nn.MLP
    cell: eqx.nn.GRUCell
    readout: eqx.nn.MLP
    hidden: int = eqx.field(static=True)

    def __init__(self, in_size: int = 6, hidden: int = 64, *, key):
        ke, km, kc, kr = jax.random.split(key, 4)
        self.hidden = hidden
        self.encoder = eqx.nn.Linear(in_size, hidden, key=ke)
        self.message = eqx.nn.MLP(2 * hidden, hidden, width_size=hidden, depth=1, key=km)
        self.cell = eqx.nn.GRUCell(hidden, hidden, key=kc)
        self.readout = eqx.nn.MLP(hidden, 1, width_size=hidden, depth=1, key=kr)

    def __call__(self, x_node, x_adj):
        # x_node: (H, N, 6); x_adj: (H, N, N) bool/float.
        H, N, _ = x_node.shape
        h0 = jnp.zeros((N, self.hidden))

        def step(h, inp):
            feats, adj = inp                              # (N,6), (N,N)
            z = jax.vmap(self.encoder)(feats)             # (N, hidden)

            a = adj.astype(jnp.float32)
            a = a - jnp.diag(jnp.diag(a))                 # strip self-loops -> A_hat
            deg = a.sum(-1, keepdims=True)                # (N,1)
            agg = (a @ z) / jnp.maximum(deg, 1.0)         # NORMALIZED MEAN (never sum)

            msg = jax.vmap(self.message)(jnp.concatenate([z, agg], axis=-1))  # (N, hidden)
            h_new = jax.vmap(self.cell)(msg, h)           # per-node temporal GRU update
            return h_new, None

        adj_f = x_adj.astype(jnp.float32)
        h, _ = jax.lax.scan(step, h0, (x_node, adj_f))    # scan over H steps
        return jax.vmap(self.readout)(h)[:, 0]            # (N,) log-lambda2
