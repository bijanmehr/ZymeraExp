# Architecture ideas — transformers, KANs, distillation (2026-06-29 design session)

**Status:** design notes. Anchored on transformers; KAN + distillation captured as related ideas.
**Mission token** is PARKED per the user (they have specific missions in mind). **Neighbor-attention**
is running now (`run_slam_attn`, `--agg multihead`). **Attention critic** is the next thing to build.

## 1. Transformers — the unifying view

Every transformer use we want collapses into ONE design: a per-agent attention over a heterogeneous
**token set**, which maps 1:1 onto the project's micro / bridge / macro formalism:

| token | formalism layer | status |
|---|---|---|
| local perception patches | **micro** — `xᵢ` local input | have (CNN over obs) |
| neighbor messages | **bridge** — aggregation over `Nᵢ` | **READY** = `--agg multihead` (GAT) |
| all-agent tokens (critic only) | centralized credit assignment | **BUILD** = attention critic (MAAC) |
| mission token | **macro** — the Dec-POMDP objective | **PARKED** (goal-conditioning) |

So the transformer isn't bolted on — it is an architectural instantiation of micro/bridge/macro, and the
mission token is the macro layer made explicit. This is the bridge from the grid swarm to the LLM-agent
end of the "MARL formalizes autonomous teams across the spectrum" thesis.

### Placements, ranked by near-term value
1. **Attention critic (MAAC — Iqbal & Sha 2019).** Attention over the agent set in the CENTRAL critic.
   Best near-term lever: train-only, so it dodges the scale / inference-budget / decentralization
   constraints that killed every learned-global attempt; and it attacks the actual wall — *credit
   assignment = who-covers-what = flooding*. A third cure for flooding alongside difference rewards +
   Voronoi. → **build first.**
2. **Neighbor-attention (GAT).** Softmax attention over in-range neighbours; replaces mean/max pooling.
   Scale-safe (permutation/count-invariant). Already implemented = `--agg multihead`; A/B running in
   `run_slam_attn` (2×2 agg × sense_walls @32²/10). Honest caveat: often ties mean/max on simple signals.
3. **Mission token.** Small learned embedding (~8–32 d, or one-hot) of *which mission*, attended like an
   LLM system prompt → one architecture serves many missions; enables zero-shot mission transfer. No-op for
   a single mission (constant) → **parked until mission #2.** Keep MINIMAL (one token, not a brain tower —
   already rejected). Interpretability: opaque by default but small + probe-able, or hand-structure named
   axes (gather↔spread · static↔moving-target · cover↔track) to read by construction.

### Do NOT
- Attention in the toolbox **router/selector** — that's the learned selector that collapsed 49→8% at scale.
- Attention over **own trajectory history** — redundant with the belief map (coverage ≈ Markov in the map).

## 2. Kolmogorov–Arnold Networks — SKIP the backbone; narrow MARL niches only

Edge-spline activations suit *smooth low-dim function fitting* + univariate interpretability, NOT high-dim
pixel/graph perception (no equivariance; slower per forward pass — bad for a many-rollout RL loop; our
scale-transfer needs CNN/GNN weight-sharing). Genuinely reasonable KAN-in-MARL niches, all low-dim +
interpretability-driven:
- **Value-decomposition mixer (QMIX/QPLEX) as a monotonic, interpretable KAN** — the mixer combines N
  per-agent Q-values into a joint Q, must be monotone, is credit-assignment-critical; KAN can be monotone
  *and* readable. (Doesn't apply to us — we're actor-critic, not QMIX.)
- **Interpretable constraint / Lagrangian head** — KAN mapping `(λ₂, local degree, dist-to-frontier) →
  connectivity penalty`, readable ("engages sharply below λ₂≈0.3"). *This one applies to our RCPO dual.*
- Actors over **low-dim engineered obs** (MPE/SMAC-like, not our pixel/graph obs).
Verdict: SKIP for the backbone; the interpretable Lagrangian-dual head is the only place worth a try here.

## 3. Distillation — DEPRIORITIZED for the coverage wall (recalibrated 2026-06-29)

**Recalibration (user skepticism — warranted).** Distillation is NOT the coverage-wall lever. The wall is
multi-agent coordination bounded by *decentralized information*: you can't distil information the student
lacks, and the obtainable teachers (single-agent human/scripted-local → local competence not division of
labor; privileged-perception → fixes perception, not the wall; centralized-coordinated → un-imitable from
local obs). A BC warm-start also regresses to the training optimum if that optimum is flooding — distillation
can't fix a misspecified objective, only postpone the regression. Whether *any* teacher could help hinges on
whether 45% is a training failure vs a decentralized info/architecture ceiling — the **scripted-compass
ceiling probe answers that**. Until then: deprioritize for coverage. Keep the *checkpoint* warm-start (#65 —
plain weight-transfer, NOT distillation). Distillation's honest homes = deployment-compression + the
mission-generalist (mission #2). The forms below are retained for those homes, not the wall:
- **(a) Privileged-teacher → partial-obs-student** ("learning-by-cheating," Chen 2020). We already have the
  asymmetry — the central critic / a full-obs teacher / the scripted Voronoi compass-ORACLE sees ground
  truth. Distil into the decentralized actor → the student imitates coordination it can't *discover* from
  scratch (a direct attack on the coverage wall).
- **(b) Mission distillation** — per-mission specialists → one mission-conditioned generalist. The build
  path for the versatile ground-zero agent + the mission token.
- **(c) Human-teleop → behavioral-cloning warm-start** (user's idea, 2026-06-29). Drive ONE agent on its
  PARTIAL obs, record `(obs, action)`, BC → `--init-from` RL fine-tune (kickstarting / RL-from-demos;
  curriculum-adjacent). Injects the systematic-coverage prior RL won't discover; single-agent demo teaches
  the *local* skill (locality), RL learns coordination on top.

**Caveats (c, and (a) generally):** demo on the agent's PARTIAL view, not a god map, or it won't transfer
to a deployable decentralized policy; use BC-init + RL fine-tune (not pure BC) to beat distribution shift;
single-agent demos give local competence, NOT coordination — compose with difference-rewards / attention
critic, don't expect them to break flooding alone.

**Synthesis:** the human teleop and the scripted compass-oracle are interchangeable *teachers* for the same
`teleop/oracle → record → BC → --init-from → RL` pipeline (human = intuition + cross-mission flexibility;
oracle = automatable + infinite demos). Plugs into the `--init-from` machinery already being wired for the
warm-start track. New infra needed: a small teleop+record interface (adapt the report canvas) + a BC
(supervised obs→action) trainer writing the existing checkpoint format.

## References
- MAAC: Iqbal & Sha, *Actor-Attention-Critic for MARL*, ICML 2019. · GAT: Veličković et al., ICLR 2018.
- Distillation: Hinton et al. 2015 · Chen et al., *Learning by Cheating*, CoRL 2020 · Vinyals et al.,
  *AlphaStar*, 2019 (SL-from-demos → RL) · Hester et al., *DQfD*, AAAI 2018.
- KAN: Liu et al., *KAN: Kolmogorov–Arnold Networks*, 2024.
