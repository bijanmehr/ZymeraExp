"""Config for the GROUNDED CTDE v0 agent — full §5-style nested schema.

This mirrors EXPERIMENT_PLAN.md §5 ("every run carries one") and captures EVERY
knob of the grounded TeamBlue agent (agent_architecture.md): the LPAC-style
backbone + GNN-KB aggregator, the multi-level goal-pointer → L1-controller action
stack, the decentralized local-Fiedler λ̂₂ head, the centralized critic, the
mission-safety mechanism, the coverage+connectivity reward weights, and the
regularization. Nothing the agent does is left implicit — a run is fully
documented by its saved config.

The schema is a tree of small frozen dataclasses (``World``, ``Backbone``,
``ActionHead``, ``MissionSafety``, ``Reward``, ``Connectivity``, ``Loss``,
``Trainer``, ``Regularization``) hung off the top-level :class:`CTDEConfig`.
``to_dict`` / :func:`from_dict` round-trip the whole tree to/from JSON so
``train_ctde.py`` can save it next to each run and rebuild it.

Defaults = the v0 slice the task specifies: comm-coverage at 16×16 / 4 agents,
100-step horizon, MAPPO-CTDE, LPAC backbone with max aggregation, a goal-pointer
head over a 9-candidate offset stencil + greedy controller, action-mask
mechanism, coverage(1)+connectivity(2) reward.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any


# =============================================================================
# Leaf config blocks (the §5 tree)
# =============================================================================


@dataclass(frozen=True)
class World:
    """zymera comm-coverage recipe + per-rung overrides (absolute cells)."""
    recipe: str = "comm-coverage"
    grid: int = 16
    n_agents: int = 4
    comm_r: int = 5            # DiskTopology radius; also feeds the true-λ₂ oracle
    sense_r: int = 1
    cover_r: int = 0
    n_obstacles: int = 0
    spawn_radius: int | None = 2   # ClusterSpawn radius; None -> scatter spawn
    horizon: int = 100


@dataclass(frozen=True)
class Backbone:
    """LPAC-style backbone: CNN local-perception -> GAP -> per-agent feature,
    then a GNN message-passing KB over the in-range comm graph.

    * ``type``      — "lpac" (CNN+GNN). Only value for v0.
    * ``depth``     — conv layers in the local-perception CNN.
    * ``width``     — conv channels / latent width = belief dim.
    * ``mp_rounds`` — GNN message-passing rounds over the comm graph (KB fusion).
    * ``norm``      — "layer" (LayerNorm on the belief) | "none".
    * ``agg``       — neighbour aggregator: "mean" | "max" (default) | "multihead".
                      NORMALIZED / size-invariant by construction (never raw-sum).
    * ``heads``     — attention heads when agg == "multihead".
    """
    type: str = "lpac"
    depth: int = 2
    width: int = 64
    mp_rounds: int = 2
    norm: str = "layer"
    agg: str = "max"
    heads: int = 4


@dataclass(frozen=True)
class ActionHead:
    """Multi-level (L3 goal -> L1 controller) action stack — NO direct moves.

    * ``kind``       — "goal_pointer": pick 1 of K candidate relative waypoints.
    * ``K``          — number of candidate offsets in the stencil (9 = center + 8
                       compass dirs; if a different K is given the stencil is the
                       first K of [center, N, E, S, W, NE, SE, SW, NW]).
    * ``stride``     — cells from the agent to each compass candidate (absolute,
                       so the goal geometry is scale-invariant).
    * ``controller`` — "greedy": Chebyshev-descent toward the goal, emitting ONLY
                       env-valid moves (STAY fallback). The sim still sees 1-step
                       moves; the 100-step budget is unchanged.
    """
    kind: str = "goal_pointer"
    K: int = 9
    stride: int = 3
    controller: str = "greedy"


@dataclass(frozen=True)
class MissionSafety:
    """Connectivity / mission-safety enforcement — the swept mechanism axis.

    * ``mechanism`` — "action_mask" (default): forbid goal candidates whose
                      greedy first move would drop true λ₂ below ``min_lambda2``
                      (a hard local guardrail) | "soft_lambda": no masking; a
                      λ·penalty term is added to the reward instead.
    * ``min_lambda2`` — the connectivity floor the action_mask defends.
    """
    mechanism: str = "action_mask"
    min_lambda2: float = 1e-3


@dataclass(frozen=True)
class Reward:
    """Coverage + connectivity reward, composed in the experiment from the env's
    UNWEIGHTED per-term magnitudes (reward engineering stays here).

    Defaults = zymera DEFAULT_TERMS weights (coverage 1 / connectivity 2 /
    collision -4). ``soft_lambda_penalty`` is the λ scale used only when
    ``MissionSafety.mechanism == 'soft_lambda'``.
    """
    kind: str = "extrinsic"
    w_coverage: float = 1.0       # new_coverage (fresh team-covered cells)
    w_connectivity: float = 2.0   # reach_fraction (fraction of others reachable)
    w_collision: float = -4.0     # collision_count (co-located others; penalty)
    soft_lambda_penalty: float = 1.0
    normalized: bool = False      # divide coverage term by free-cell count


@dataclass(frozen=True)
class Connectivity:
    """The locked connectivity metric + agent signal (EXPERIMENT_PLAN §1)."""
    estimator: str = "fiedler_local_poweriter"   # Phase-0 decentralized estimator
    grade_on: str = "true_lambda2"               # non-gameable grader
    threshold: float = 1e-3                       # connectivity-% = steps with λ₂ > τ
    trade_off_lambda: float | None = None         # swept in Phase 2 (soft mechs)
    lambda2_sharp: float = 2.0                    # sigmoid sharpness of the soft graph
    estimator_iters: int = 8                      # power-iteration rounds for λ̂₂


@dataclass(frozen=True)
class Loss:
    """PPO + auxiliary λ₂ supervision knobs."""
    ppo_clip: float = 0.2
    aux_beta: float = 0.1               # weight on the aux λ₂ loss in the total
    aux_loss: str = "mse"               # "mse" | "huber"
    huber_delta: float = 0.1            # delta when aux_loss == "huber"
    vf_coef: float = 0.5


@dataclass(frozen=True)
class Trainer:
    """The optimizer (MAPPO-CTDE)."""
    kind: str = "mappo"
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2                   # mirrors Loss.ppo_clip (kept for §5 parity)
    ppo_epochs: int = 4
    minibatches: int = 4
    max_grad_norm: float = 0.5


@dataclass(frozen=True)
class Regularization:
    """All regularizers, each gated behind a knob.

    * ``degree_reg``    — SizeShiftReg-style weight on the variance of per-node
                          aggregated degree statistics across the batch (guards
                          GNN size-transfer); small default.
    * ``entropy_coef``  — entropy bonus on the goal policy.
    * ``weight_decay``  — AdamW decoupled weight decay.
    * ``dropout``       — dropout rate on the belief (0 = off).
    """
    degree_reg: float = 1e-3
    entropy_coef: float = 0.01
    weight_decay: float = 1e-4
    dropout: float = 0.0


# =============================================================================
# Top-level config
# =============================================================================


@dataclass(frozen=True)
class CTDEConfig:
    world: World = field(default_factory=World)
    backbone: Backbone = field(default_factory=Backbone)
    action_head: ActionHead = field(default_factory=ActionHead)
    mission_safety: MissionSafety = field(default_factory=MissionSafety)
    reward: Reward = field(default_factory=Reward)
    connectivity: Connectivity = field(default_factory=Connectivity)
    loss: Loss = field(default_factory=Loss)
    trainer: Trainer = field(default_factory=Trainer)
    regularization: Regularization = field(default_factory=Regularization)

    # ---- run control --------------------------------------------------------
    scale: str = "16x16/4"            # human label for the rung
    iters: int = 50                  # PPO iterations
    rollouts_per_iter: int = 8       # parallel episodes per iteration (vmap seeds)
    seed: int = 0
    ckpt_path: str | None = None     # if set, save the final actor/critic + meta

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    # ---- convenience accessors (so callers don't reach through the tree) ----
    @property
    def n_actions_goal(self) -> int:
        return self.action_head.K


# --- field name -> leaf type, for from_dict reconstruction --------------------
_LEAF_TYPES: dict[str, type] = {
    "world": World, "backbone": Backbone, "action_head": ActionHead,
    "mission_safety": MissionSafety, "reward": Reward,
    "connectivity": Connectivity, "loss": Loss, "trainer": Trainer,
    "regularization": Regularization,
}


def _build_leaf(cls: type, d: Any):
    if not isinstance(d, dict):
        return d
    names = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in d.items() if k in names})


def from_dict(d: dict) -> CTDEConfig:
    """Rebuild a full config tree from a plain dict (e.g. loaded JSON)."""
    top_scalar = {f.name for f in dataclasses.fields(CTDEConfig)} - set(_LEAF_TYPES)
    kw: dict[str, Any] = {}
    for name, cls in _LEAF_TYPES.items():
        if name in d:
            kw[name] = _build_leaf(cls, d[name])
    for name in top_scalar:
        if name in d:
            kw[name] = d[name]
    return CTDEConfig(**kw)


def flat_schema(cfg: CTDEConfig) -> dict:
    """Flatten the config tree to ``block.field -> value`` (for human listing/logs)."""
    out: dict[str, Any] = {}
    for k, v in cfg.to_dict().items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                out[f"{k}.{kk}"] = vv
        else:
            out[k] = v
    return out
