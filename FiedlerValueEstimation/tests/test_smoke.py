import jax
import jax.numpy as jnp
import zymera


def test_zymera_rollout_exposes_positions_and_comm_graph():
    env = zymera.make("comm-coverage", grid=12, n_agents=4, comm_r=5)
    traj = zymera.rollout(env, zymera.random_policy, n_steps=10, key=jax.random.PRNGKey(0), keep="all")
    pos = traj["world"].body.position          # (T+1, N, 2)
    comm = traj["world"].comm_graph            # (T+1, N, N)
    assert pos.shape == (11, 4, 2)
    assert comm.shape == (11, 4, 4)
    assert pos.dtype == jnp.int32
