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
    * ``recurrence``— per-agent temporal memory over the belief (the recurrence axis):
        - "feedforward" (default): the heads read the per-step belief z directly — v0
          behaviour, byte-unchanged. The GRU is still BUILT (stable param surface) but
          never used.
        - "recurrent": an ``eqx.nn.GRUCell`` (W -> W) carries a per-agent hidden state
          ``h`` ACROSS the 100-step episode (``h_t = GRUCell(z_t, h_{t-1})``, reset to
          zeros at episode start); EVERY head reads ``h_t`` instead of ``z_t`` so the
          agent remembers its own trajectory / coverage history (relevant to dispersal
          OVER TIME). The hidden threads through BOTH the rollout (scan carry) and the
          PPO loss (recomputed along each trajectory under the current params, BPTT via
          a per-episode scan; the recurrent path minibatches over EPISODES, keeping each
          100-step sequence intact). Width = ``width`` (the belief dim).
    """
    type: str = "lpac"
    depth: int = 2
    width: int = 64
    mp_rounds: int = 2
    norm: str = "layer"
    agg: str = "max"
    heads: int = 4
    recurrence: str = "feedforward"   # {"feedforward", "recurrent"}


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
    * ``explorer_tool`` — how the EXPLORER picks its goal sector (the L4 "disperse"
                       skill / I2 explorer-tool axis):
        - "goal_head" (default): the goal-pointer logits come from the belief z
          ALONE (``nets.Actor.goal_head``) — v0 behaviour, byte-unchanged.
        - "frontier_attn": a learned frontier-attention module
          (``nets.FrontierAttn``) ADDS a per-sector bias to the goal logits,
          pulling the goal toward the compass sector with the most UNCOVERED ground
          (the agent's own ``known`` channel = frontier). It gives the explorer an
          EXPLICIT frontier-seeking mechanism instead of relying on the reward to
          discover dispersal from a clustered spawn. The module is ALWAYS built (a
          stable param surface) but only contributes under this value; PPO still
          samples the goal (the bias never argmaxes). Only the EXPLORER role uses it
          — relays run the λ̂₂-anchor controller and discard the goal regardless, so
          when ``role_picker == 'off'`` every (explorer) agent uses the tool and
          when it is on relays are unaffected. Scale-invariant: K fixed, the sector
          features are normalized fractions, so a model trained @16²/4 transfers.
    * ``relay_tool`` — which RELAY controller the relay role calls (the I2 relay-tool
                       axis / agent_architecture.md "Relay tool"). Only the relay role
                       uses it; explorers are unaffected, and with ``role_picker ==
                       'off'`` every agent is an explorer so this knob is inert:
        - "lambda2_anchor" (default): the existing ``controller.relay_move`` — each
          relay actively takes the env-valid move that MAXIMIZES its local soft-degree
          (λ̂₂-anchor), i.e. it climbs connectivity every step. v0/I1 behaviour,
          byte-unchanged.
        - "hold": ``controller.relay_hold_move`` — a low-energy STATIC BEACON. The
          relay STAYS put (keeps the bridge from where it stands) UNLESS staying would
          leave it isolated (soft-degree below a floor), in which case it takes the
          single valid move that best re-establishes a neighbour. "Don't wander, just
          hold the connection" vs the anchor's active connectivity-climbing.
    * ``compass`` — append a small SCALE-INVARIANT directional feature to the per-agent
                       belief z BEFORE the heads (the I2 compass feature /
                       agent_architecture.md "compass"):
        - "off" (default): z unchanged -> byte-identical to the pre-compass actor. The
          compass module is STILL built (stable param surface) but never used.
        - "on": ``nets.Compass`` ADDS a gated projection of two soft K-sector
          DIRECTIONS — the GATHER direction (toward the centroid of in-range
          teammates, the ``neighbors`` channel) and the EXPLORE direction (toward the
          nearest uncovered cell, ``1 - known``) — to z, giving every head an explicit
          navigation cue beyond the CNN's local view. Directions only (no distances /
          absolute coords) -> scale-invariant: a model trained @16²/4 transfers.
    """
    kind: str = "goal_pointer"
    K: int = 9
    stride: int = 3
    controller: str = "greedy"
    explorer_tool: str = "goal_head"        # {"goal_head", "frontier_attn"}
    relay_tool: str = "lambda2_anchor"      # {"lambda2_anchor", "hold"}
    compass: str = "off"                     # {"off", "on"}


@dataclass(frozen=True)
class MissionSafety:
    """Connectivity / mission-safety enforcement — the swept mechanism axis.

    * ``mechanism`` — the connectivity-enforcement mechanism (I1b extends the set):
        - "action_mask" (default): forbid goal candidates whose greedy first move
          would drop true λ₂ below ``min_lambda2`` (a hard local guardrail).
        - "soft_lambda": no masking; a FIXED-weight λ·penalty term is added to the
          reward instead (``Reward.soft_lambda_penalty``).
        - "lagrangian": an ADAPTIVE penalty — a dual variable λ ≥ 0 (carried in the
          train state) is dual-ascended on the realized connectivity violation so
          the policy LEARNS to hold the graph (Lagrangian-PPO).
        - "pid_lagrangian": same violation, but λ comes from a PID controller
          (Stooke et al. 2020, "Responsive Safety in RL") for smoother dual
          dynamics (carries integral + prev-error in the train state).
      The two adaptive mechanisms read a CTDE training-time true-λ₂ signal; only
      they activate the dual-variable state (action_mask / soft_lambda are byte-
      unchanged from I1).
    * ``conn_signal`` — the SIGNAL SOURCE the penalty mechanisms read, ORTHOGONAL
      to ``mechanism`` (I1c adds this axis; every mechanism × conn_signal combo is
      valid, action_mask alone ignores it since it masks actions, no penalty):
        - "global_lambda2" (default): the I1b signal — a GLOBAL scalar (true team λ₂
          vs the floor) broadcast IDENTICALLY to all N agents, so no single agent
          knows it is the one stretching the bridge. Default keeps I1b byte-unchanged.
        - "local_edge_margin": a LOCAL, PER-AGENT signal — each agent's own
          soft-degree shortfall (``env_utils.local_edge_margin``), positive only for
          agents drifting toward the edge of comm range (anticipatory, partial-obs-
          native), ≈0 for agents comfortably in the interior. Not broadcast.
    * ``degree_target`` — the per-agent soft-degree floor the "local_edge_margin"
      signal charges the shortfall against (p_i = relu(degree_target − soft_deg_i)).
    * ``min_lambda2`` — the connectivity floor the action_mask defends.
    * ``lambda_init`` — initial dual variable λ for the adaptive mechanisms.
    * ``lambda_lr`` — dual-ascent step size for the "lagrangian" mechanism.
    * ``constraint_threshold`` — the connectivity floor τ the violation
        ``v = relu(τ − mean_rollout(true λ₂))`` is measured against (global_lambda2);
        under local_edge_margin the violation is instead ``v = mean_i p_i``, the
        rollout-mean per-agent margin shortfall. ``None`` -> reuse the locked grading
        threshold ``Connectivity.threshold`` (do NOT invent a second floor); resolve
        via :meth:`CTDEConfig.constraint_threshold`.
    * ``pid_kp`` / ``pid_ki`` / ``pid_kd`` — PID gains for "pid_lagrangian".
    """
    mechanism: str = "action_mask"
    conn_signal: str = "global_lambda2"         # {"global_lambda2","local_edge_margin"}
    degree_target: float = 1.0                  # soft-degree floor for local_edge_margin
    min_lambda2: float = 1e-3
    lambda_init: float = 0.0
    lambda_lr: float = 0.05
    constraint_threshold: float | None = None   # None -> Connectivity.threshold
    pid_kp: float = 1.0
    pid_ki: float = 0.01
    pid_kd: float = 0.1


@dataclass(frozen=True)
class Reward:
    """Coverage + connectivity reward, composed in the experiment from the env's
    UNWEIGHTED per-term magnitudes (reward engineering stays here).

    Defaults = zymera DEFAULT_TERMS weights (coverage 1 / connectivity 2 /
    collision -4). ``soft_lambda_penalty`` is the λ scale used only when
    ``MissionSafety.mechanism == 'soft_lambda'``.

    Connectivity-FLOOR barrier ("Hyper-Singularity") — a STANDALONE, config-knobbed
    reward term (``env_utils.connectivity_barrier``) that COMPOSES with every other
    connectivity mechanism (``conn_signal`` / ``mechanism``); it does NOT replace any.
    A capped one-sided wall on each agent's nearest-neighbour Chebyshev distance:
    EXACTLY 0 inside ``barrier_a`` (silent in the safe zone), an explosive-but-FINITE
    rise as a link nears ``barrier_M`` (the break range), saturating at ``barrier_cap``
    at/past ``barrier_M``. ``barrier_weight`` is the formula's ``k``; at its DEFAULT 0
    the term is OFF / exactly 0 and the composed reward is byte-unchanged.

    * ``barrier_weight`` — k; the term's weight. 0 (default) -> OFF (no-op).
    * ``barrier_a``      — launch point (0 below it). ``None`` -> ``world.comm_r * 0.6``.
    * ``barrier_M``      — the wall / break range. ``None`` -> ``world.comm_r`` (link
                           breaks at comm range). Both resolve via the ``barrier_a`` /
                           ``barrier_M`` accessors on :class:`CTDEConfig` (one source of
                           truth for ``comm_r``, like ``MissionSafety.constraint_threshold``).
    * ``barrier_p``      — explosion power on the ``(M - x)`` denominator.
    * ``barrier_cap``    — the finite "almost-infinity" ceiling the wall saturates at.
    """
    kind: str = "extrinsic"
    w_coverage: float = 1.0       # new_coverage (fresh team-covered cells)
    w_connectivity: float = 2.0   # reach_fraction (fraction of others reachable)
    w_collision: float = -4.0     # collision_count (co-located others; penalty)
    soft_lambda_penalty: float = 1.0
    normalized: bool = False      # divide coverage term by free-cell count
    # --- connectivity-FLOOR barrier (composes with conn_signal/mechanism) ---
    barrier_weight: float = 0.0          # k; 0 -> term OFF / exactly 0 (byte-unchanged)
    barrier_a: float | None = None       # launch point; None -> world.comm_r * 0.6
    barrier_M: float | None = None       # wall / break range; None -> world.comm_r
    barrier_p: float = 2.0               # explosion power
    barrier_cap: float = 50.0            # finite "almost infinity" ceiling


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

    # ---- Increment-1 knobs (additive; defaults reproduce v0 behaviour) -------
    # role_picker: "off" -> homogeneous goal head (v0). "expl_relay" -> a learned
    #   role head off z_i picks {explorer, relay} per agent; explorers run the
    #   frontier/goal behaviour, relays run a local λ̂₂-anchor move that holds the
    #   bridge. The role is a categorical PPO action trained as part of the policy.
    role_picker: str = "off"
    # reward_anti_overlap: "on" -> subtract anti_overlap_weight * same_step_overlap
    #   (teammate-overlapping fresh cells) from the composed reward, rewarding
    #   NON-redundant coverage. "off" -> v0 reward unchanged.
    reward_anti_overlap: str = "off"
    anti_overlap_weight: float = 1.0  # weight on same_step_overlap when on

    # ---- Increment-1b knobs (additive; defaults reproduce I1 behaviour) ------
    # collision_mask: "off" -> the controller may step onto a cell another agent
    #   currently occupies (v0/I1 behaviour). "on" -> a HARD collision mask
    #   (controller.occupied_cell_mask) removes those moves from BOTH L1
    #   controllers before they pick, so agents never step onto an occupied cell;
    #   STAY stays selectable (emitted moves remain env-valid). Exposed as an axis,
    #   NOT hardcoded. (The learned connectivity mechanisms live on
    #   ``mission_safety.mechanism`` — lagrangian / pid_lagrangian.)
    collision_mask: str = "off"

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

    @property
    def constraint_threshold(self) -> float:
        """Connectivity floor τ the Lagrangian mechanisms measure the violation
        against. ``mission_safety.constraint_threshold`` if set, else the locked
        grading threshold ``connectivity.threshold`` (one floor, not a second)."""
        ms = self.mission_safety.constraint_threshold
        return self.connectivity.threshold if ms is None else ms

    @property
    def barrier_a(self) -> float:
        """Resolved barrier launch point a: ``reward.barrier_a`` if set, else
        ``world.comm_r * 0.6`` (one source of truth for ``comm_r``)."""
        a = self.reward.barrier_a
        return float(self.world.comm_r) * 0.6 if a is None else float(a)

    @property
    def barrier_M(self) -> float:
        """Resolved barrier wall / break range M: ``reward.barrier_M`` if set, else
        ``world.comm_r`` (the link breaks at comm range)."""
        m = self.reward.barrier_M
        return float(self.world.comm_r) if m is None else float(m)


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
