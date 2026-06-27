import numpy as np
import jax.numpy as jnp
from fiedler import features
from fiedler.fiedler import potential_adjacency


def test_node_features_shape_and_degree_column():
    # 3 agents in a line at cols 0,1,2 on the same row; comm_r=1 => chain 0-1-2
    pos = jnp.asarray([[0, 0], [0, 1], [0, 2]], dtype=jnp.int32)
    adj = potential_adjacency(pos, comm_r=1)
    f = features.node_features(pos, adj, comm_r=1)
    assert f.shape == (3, 6)
    deg = np.asarray(f[:, 0])
    assert deg.tolist() == [1.0, 2.0, 1.0]

def test_isolated_agent_has_zero_degree_and_safe_stats():
    pos = jnp.asarray([[0, 0], [9, 9]], dtype=jnp.int32)  # far apart, comm_r=1 => no edge
    adj = potential_adjacency(pos, comm_r=1)
    f = features.node_features(pos, adj, comm_r=1)
    assert f.shape == (2, 6)
    assert np.all(np.isfinite(np.asarray(f)))
    assert np.asarray(f[:, 0]).tolist() == [0.0, 0.0]
