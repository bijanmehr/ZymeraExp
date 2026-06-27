from dataclasses import dataclass


@dataclass(frozen=True)
class DataCfg:
    n_agents: int = 4
    grid: int = 16
    comm_r: int = 5
    n_obstacles: int = 0
    spawn_radius: int = 2
    n_episodes: int = 8
    n_steps: int = 100
    seed: int = 0
