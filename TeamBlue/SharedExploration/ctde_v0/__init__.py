"""Grounded CTDE v0 — the TeamBlue shared-exploration agent (multi-level L3/L1).

A from-scratch MAPPO-CTDE trainer on the zymera ``comm-coverage`` task, built to
the TeamBlue architecture (agent_architecture.md / EXPERIMENT_PLAN.md), NOT a
flat actor-critic:

  * ``config``     — the full §5-style nested run config (every knob).
  * ``nets``       — LPAC backbone (CNN -> GAP -> GNN message-passing KB with a
    configurable normalized aggregator) + goal-pointer / λ̂₂ / value heads
    (``Actor``) and the centralized ``Critic`` (CTDE).
  * ``controller`` — the L1 controller: goal stencil, greedy wall-aware move,
    and the action-mask mission-safety mechanism.
  * ``ppo``        — rollout (goal -> controller -> move), GAE on the central
    critic, PPO(goal) + aux-λ₂ + degree-reg + entropy, the trainer.
  * ``env_utils``  — env build, reward composition, true-λ₂ oracle, KB adjacency,
    degree stats, coverage metric.

Entry point: ``train_ctde.py``.
"""
from .config import CTDEConfig, from_dict  # noqa: F401
