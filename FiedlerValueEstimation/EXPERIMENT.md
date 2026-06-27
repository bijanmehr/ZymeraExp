# Fiedler-Value Estimation — Experiment Design

A **standalone study**: estimate the **global** Fiedler value **λ₂** (algebraic connectivity) of the
comm graph from each agent's **local, partial** view. Cheap and self-contained — the sim gives us **true λ₂
for free** (eigendecomposition of the comm-graph Laplacian), so this is **supervised regression, no RL in the
loop**. Feeds the SuperBlue campaign (the agent's connectivity signal) but stands on its own.

---

## Objective
For each method, find the **minimum history length `H`** (past timesteps of an agent's local observation)
needed to estimate global λ₂ to **≥ 95 % accuracy** — comparing **multiple heuristic *and* learned** methods,
across **team sizes 4–30 agents**, and including **different message / message-passing ideas (incl. learned
messaging)**. Winner = the method + (`H`, message scheme) that clears 95 % with the **fewest historical points
and least bandwidth**, *and* still holds at **30 agents** when trained on smaller teams.

**Accuracy (precise):** `accuracy = 1 − median(|λ̂₂ − λ₂| / λ₂)`; **target ≥ 95 % ⇔ median relative error
≤ 5 %** on held-out episodes. Also report **R²**, the **within-5 % hit-rate**, and **connected/fragmented
classification accuracy** so a single metric can't hide tails.

---

## Ground truth & dataset
- **Label:** the sim eigendecomposes the comm-graph Laplacian → **true λ₂(t)** (oracle; train/eval only).
- **Data:** roll out **diverse policies** (random · disperse · coverage-heuristic · a partial policy) in
  zymera; each step record per-agent **local obs** (own degree, neighbor relative positions/count, received
  messages, recent-contact memory), the **comm adjacency**, and **true λ₂**.
- Build `(H-window → λ₂)` pairs; **split by episode** (no leakage). Also log per-graph **diameter** and
  **mean degree** so results can be sliced by difficulty.

---

## Independent variables
1. **History length `H` ∈ {1, 2, 3, 5, 8, 13, 21}** — the *minimize-this* objective (inference-time window).
2. **Message design** — content × scheme × learned (see its own section).
3. **Team size `N`** — **4 → 30**. Train on **{4, 8, 12, 16, 20}**; test **interpolation {6, 10}** and
   **extrapolation {24, 30}** (zero-shot beyond the max train size, no retraining). **N = 20 additionally
   gets a dedicated 5-fold cross-validation** (rotate held-out 20-agent episodes) for a robust, low-variance
   in-distribution assessment at the upper-mid size.
4. **Graph density** — via comm-radius / world-size, swept **sparse → dense** so λ₂ spans **near-0 (chains /
   bottlenecks) → large (clusters)**; difficulty axis.

---

## Methods (multiple heuristic + multiple learned)

> **Full architecture, hyperparameter, and message-type detail → [`ARCHITECTURES.md`](./ARCHITECTURES.md).**

**No message passing — own-observation-only baselines** *(how far can you get with zero comms?)*
- **Degree-regression** — own degree → λ₂.
- **Temporal net (GRU / TCN)** — own-feature history over `H`.

**With message passing** *(carry the message-design axis below)*
- **Decentralized power-iteration** (Yang) — analytic; message = the consensus value.
- **Random-walk estimator** — analytic; λ₂ from walker mixing.
- **Snapshot GNN** — 1-hop message passing on the current graph, per-node λ̂₂.
- **Recurrent GNN (GCRN)** — message passing **+** recurrence over `H` *(main candidate)*.
- **Unrolled power-iteration net** — `H` learned power-iteration layers (analytic↔learned bridge).

**References (the framing)**
- **Centralized GNN** (sees the whole graph) = **upper bound** — the full-observability ceiling.
- **Degree-only / persistence** = **lower bound**.

The spread own-history → GNN also answers: *how much does using neighbor structure buy over own-degree
history alone?*

---

## Message-design axis  *(the "try different msg ideas, incl. learned" part)*
Applies to the message-passing families. Three sub-axes:

| Sub-axis | Variants |
|---|---|
| **Content** | **C0** value-only · **C1** + degree/count · **C2** + relative geometry · **C3 learned message** (the net encodes what to send — CommNet / DGN style) |
| **Scheme** | **P0** synchronous `k`-rounds/step · **P1** gossip (async pairwise) · **P2** recurrent (1 round/step, carry hidden state) |
| **Aggregation / learned-comms** | mean · **attention over neighbors (TarMAC)** · **learned gating — *when/whether* to send (IC3Net)** → bandwidth |

**Headline message question:** does **learned messaging (C3 / attention / gating)** reach 95 % at **smaller `H`
and/or less bandwidth** than hand-crafted messages? (A better message packs more λ₂-relevant info per round →
fewer historical points needed — directly serving the min-`H` objective.)

Run as a curated sweep: content ladder **C0→C3** under the recurrent scheme first; then schemes
**P0/P1/P2**; then **gating** for the bandwidth Pareto.

---

## Metrics
Per method × `H` × `N` × density: **median rel-error (→accuracy)**, **R²**, **within-5 % hit-rate**,
**connected-flag accuracy** — plus **bandwidth (bytes/agent/step)** and **compute**. So the real choice is
**accuracy-per-historical-point-per-byte**, not raw accuracy.

---

## Conditions
- **Team size:** train {4, 8, 12, 16, 20} → interp {6, 10} + **extrap {24, 30}**; **5-fold CV on N = 20**.
- **Density:** {sparse, medium, dense}.
- **Graph dynamics:** **drifting (swarm moving, primary)** + **frozen-graph control** (isolates staleness).
- **≥ 3 seeds**, held-out episodes for all reported numbers.

---

## Success criteria
1. **≥ 95 % accuracy** (median rel-err ≤ 5 %) on held-out test.
2. **Smallest `H`** achieving it (ideally `H ≤ 3`, the per-step budget) → the chosen estimator + message scheme.
3. **Holds ≥ 95 % extrapolating to {24, 30} agents** when trained on ≤ 20 (size-invariance bar), **and
   ≥ 95 % at N = 20 as the 5-fold CV mean with low cross-fold variance** (in-distribution robustness bar).
4. Report the **partial-observability gap** (best decentralized vs centralized-GNN) and the **bandwidth
   Pareto** (accuracy vs bytes/agent/step).

---

## Folder layout (this study)
```
FiedlerValueEstimation/
  EXPERIMENT.md      # this design
  ARCHITECTURES.md   # full network + hyperparameter + message-type specs
  datagen.py         # rollouts + oracle λ₂ → dataset (per N, density)
  methods/           # one module per estimator + message scheme
  configs/           # per-run yaml: method · H · msg(content/scheme/agg) · N · density · seed
  results/           # accuracy-vs-H curves · min-H table · extrapolation-to-30 · bandwidth Pareto
  JOURNEY.md         # daily log
```

**Build notes:** supervised, no RL; reuse zymera only for rollouts + the oracle λ₂; one config per
(method × H × message × N × density × seed); the whole study is cheap to sweep.
