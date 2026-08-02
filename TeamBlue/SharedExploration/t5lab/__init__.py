"""t5lab — isolated implementation of the T5 architecture.

New learned code lives HERE (never in ctde_v0). ctde_v0 is imported read-only as the
substrate (env, Backbone, PPO loss, the goal-head → planner → move socket). The T5 comm /
occlusion changes and the MVProp planner are added as *gated* options so the ctde_v0
default stays byte-identical when they are off (matching ctde_v0's own discipline).

Design + params:   ../t5/T5_DECISIONS.md
Experiment plan:   ../t5/T5_EXPERIMENT_PLAN.md
Build plan:        ./PLAN.md
"""
