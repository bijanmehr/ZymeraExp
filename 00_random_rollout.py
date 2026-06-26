"""Smallest possible experiment: a random policy on comm-coverage, rendered.

Demonstrates the import pattern and the one-way dependency on the lab. Not part
of the lab's test suite."""
import jax

import zymera
from zymera import train, viz   # viz is the opt-in headless-core subpackage


def main() -> None:
    env = zymera.make("comm-coverage", grid=12, n_agents=4)
    report = train.evaluate(env, zymera.random_policy, n_steps=40,
                            n_episodes=8, key=jax.random.PRNGKey(0))
    print("random-policy eval:", report)

    traj = zymera.rollout(env, zymera.random_policy, 40, jax.random.PRNGKey(1),
                          keep="all")
    viz.render_gif(traj["world"], "random_rollout.gif", comm_radius=5)
    print("wrote random_rollout.gif")


if __name__ == "__main__":
    main()
