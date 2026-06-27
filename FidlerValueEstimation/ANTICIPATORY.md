# Anticipatory / Tracking λ₂ Estimator — Design

**Premise.** The static estimator caps at ~0.66 because 2 message-passing rounds can't see a *global*
property from *local* views. The anticipatory angle attacks a different (and more mission-relevant) target:
instead of estimating the current λ₂ cold, each agent **learns a local model of how its neighbors move**,
uses it to **project the comm graph one (or more) steps ahead**, and **forecasts the next λ₂** — exploiting
the fact that λ₂ is *temporally smooth* (next = current + a small, predictable perturbation). The same
neighbor-prediction model is **dual-use**: the residual between predicted and observed neighbor behavior is a
**local anomaly / covert-adversary signal** — the mechanism the Zymera thesis is built on.

This is the *temporal twin* of "more rounds": a continuous tracker accumulates information across timesteps the
way power-iteration accumulates it across rounds.

## Goals (3 measurable deliverables)
1. **Forecast accuracy** — predict λ₂(t+1) more accurately than the static estimate of λ₂(t) (beat 0.66).
2. **Connectivity early-warning** — on episodes that fragment, flag the impending λ₂→0 with usable lead-time
   (recall @ k-steps-ahead, false-alarm rate). The dynamic generalization of the connectivity-margin feature.
3. **Anomaly detection** — against a planted misbehaving agent, the per-neighbor prediction residual separates
   red from blue (ROC-AUC, detection lead-time).

## Architecture (extends `ConfigurableGCRN` → `ForecastGCRN`)
Reuses the encoder + message-passing + per-node GRU. Adds:
- **Neighbor-prediction head** `g_pred`: from per-node state `h_i` and the current edge feature to neighbor `j`
  (`[Δx,Δy,dist]/comm_r`), predict `j`'s **next-step relative position** `Δ̂_{ij}(t+1)`. A learned local
  dynamics model (small MLP over `[h_i, h_j, edge_ij]`). This is "anticipate your neighbors."
- **Two forecast paths** (compared):
  - **(a) implicit** — train the existing `logl2` head to output **λ₂(t+1)** directly (network learns to
    anticipate internally). Simplest baseline.
  - **(b) structured** — roll the predicted relative positions forward → predicted next adjacency
    `Â(t+1) = σ((comm_r − dist̂)/τ)` (soft) → estimate λ₂ on `Â(t+1)`. Interpretable; exposes the prediction
    as an explicit graph forecast and yields the anomaly residual for free.
- **Continuous tracking** — process full episodes sequentially carrying the GRU state (online estimator,
  truncated BPTT), rather than independent H=5 windows. Accumulates info over time.

## Training (multi-task)
`L = L_forecast + λ_p·L_pred + λ_c·L_cflag (+ existing NLL)`, all node/edge masked:
- `L_forecast` = Huber(log λ̂₂(t+1), log λ₂(t+1)).
- `L_pred` = MSE(`Δ̂_{ij}(t+1)`, actual next relative position) over real edges — the auxiliary that *forces*
  the neighbor model, and whose residual becomes the anomaly score.
- `L_cflag` = BCE for the next-step connected flag (now informative once data can disconnect).
Optimizer/schedule unchanged (AdamW + cosine, checkpointed).

## Data (`fidler/datagen.py` additions)
- **Full-episode, next-step-labelled** rollouts (expose t and t+1 graphs + true λ₂(t+1)).
- **Soft regime** (disconnection-allowing): drop the hard connectivity guardrail so episodes *can* fragment —
  required for goals 2 and 3 (the hard-guardrail data never disconnects, so cflag was trivially 1.0).
- **Adversary injection**: 1 red agent following a deviating policy (e.g. anti-cohesion / random-walk / pull a
  cut-vertex out of range); label red vs blue for the anomaly eval.

## Evaluation
1. Forecast accuracy = 1 − median rel-err on λ₂(t+1), vs the static-λ₂(t) baseline (0.66).
2. Early-warning: ROC of "λ₂(t+Δ) < ε" prediction for Δ∈{1,2,4}; report recall @ fixed false-alarm + lead-time.
3. Anomaly: ROC-AUC of per-agent mean prediction residual (red vs blue); detection lead-time before the red
   agent's action actually drops λ₂.

## Implementation plan (files)
- `fidler/anticipate.py` — neighbor-prediction head, soft-adjacency roll-forward, anomaly-residual scorer.
- `fidler/models_eqx.py` — `ForecastGCRN` (prediction head + forecast head + online forward).
- `fidler/datagen.py` — next-step labels, soft regime, adversary injection.
- `fidler/train_eqx.py` — multi-task forecast loss + truncated-BPTT online training mode (reuses checkpointing).
- `fidler/metrics.py` — early-warning ROC + anomaly AUC + lead-time helpers.
- `run_anticipate.py` — launcher (implicit vs structured × tracking on/off). `tests/test_anticipate.py`.

## Ties to the program
- Goal 2 = the connectivity-margin feature made **dynamic** (forecast the break, don't just sense the strain) —
  directly the resilient-swarm "act before you fragment" capability.
- Goal 3 = the **covert-misbehavior detector** (`prediction residual = deviation = red signal`), the governing
  RQ — so this experiment doubles as the substrate for the adversarial line, not just a better λ₂ readout.
