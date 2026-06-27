# Fiedler-Value Estimation — Findings Log

A dated log of **results, interpretations, and emergent patterns** from the sweeps. Pairs with
`EXPERIMENT.md` / `ARCHITECTURES.md` (design) and the raw `results/*.jsonl` + `experiments/*.json` records.
Newest at the bottom.

**Standing caveat (applies to every result below):** training/eval data is the **hard-connectivity-guardrail
anti-crowding dispersion** regime — a *proxy* for a coverage swarm's comm graphs (always-connected, dispersed),
**not** a real coverage mission and **not** a trained RL agent. Real-mission validation (coverage policy data,
then on-policy fine-tuning) is still owed. Accuracy = `1 − median relative error` on the true λ₂ (oracle
eigendecomposition); target ≥ 0.95.

---

## 2026-06-26 — Slice-2a (single-N, random-policy data)

- First learned estimators (GRU own-history, GCRN) at N=16 on **random-policy** data scored **~0.45** — barely
  above the degree-only floor (0.35), far below power-iteration (~1.0).
- **Diagnosis:** random-policy rollouts are dominated by **disconnected** graphs (λ₂≈0), where the relative-error
  metric explodes. Wrong distribution + wrong regime. → motivated the guardrail data.

## 2026-06-26 — Slice-2b sweeps on guardrail data (running)

Setup: guardrail-dispersion data, `train_N=[4,8,12,16,20]`, hidden=128, n_rounds=2, H=5, 8k steps, 5-fold@20,
zero-shot extrapolation to N∈{24,30}. Four sweeps in parallel (tmux `zymera`/`zid`/`zmar`/`zsig`).

**Aggregation-op sweep (content=value), partial (4/8):**

| op | accuracy | cv20@20 (±std) | →24 | →30 |
|---|---|---|---|---|
| mean | 0.58 | 0.61 ± .039 | .56 | **.00** |
| gcn | 0.64 | 0.63 ± .018 | .46 | **.00** |
| **max** | **0.66** | **0.66 ± .006** | .47 | **.00** |
| sum | 0.64 | 0.63 ± .019 | .52 | **.00** |

**Identity sweep (op=mean), partial (2/6):** `id=none` 0.578 vs `id=random` 0.578 — **identical**.

**Findings / emergent patterns:**
1. **Guardrail data >> random.** Accuracy 0.45 → ~0.58–0.66; `connected_accuracy == accuracy` (no disconnected
   graphs to tank the metric). Validates the guardrail-data decision.
2. **`max` aggregation wins** in-distribution AND is *dramatically* more reliable (cv-std **0.006** vs 0.02–0.04).
   Worth understanding why max is so stable here.
3. **Random-ID tag does nothing** (0.578 vs 0.578). Matches theory exactly: **λ₂ is permutation-invariant**, so
   identity is irrelevant *for estimation*. (Distinguishability matters for *coordination/policy*, not this target.)
   The `index` (raw-ID) arm — predicted to *hurt* extrapolation — is still pending.
4. **`sum` (the ablation expected to break size-invariance) did NOT visibly break** in-distribution — likely
   because the fixed-density guardrail holds node-degree ~constant, muting sum's degree/N scaling problem.
5. **THE WALL: nothing extrapolates to N=30** (`→30 = 0.00` for every config; `→24 ≈ 0.5`). **Confirmed REAL,
   not a bug** — `_extrapolate` builds at the true N (no Nmax truncation) and the GCRN is architecturally
   size-invariant; the smooth slide 24→30 is genuine >100% prediction error. The estimator trained on N≤20 does
   not generalize the λ₂ *scale* to N=30. This is the Round-2 size-transfer wall, made concrete.

**Still mediocre:** best ~0.66 = ~34% median error — far from the 0.95 target. These are untuned baselines.

**Open / next:**
- **Fix the wall:** include N=24/30 *in* the training pool (interpolation not extrapolation) and/or add the
  size-transfer regularizers (SizeShiftReg / subgraph-augmentation) that aren't implemented yet.
- **margin** (dist-to-comm-range) and **signal+ID** (soft path-loss weighting × identity) sweeps — running.
- **Better aggregators** (PNA etc.) — lit search in progress.
- Real-mission data (coverage policy / on-policy) — still owed.

## 2026-06-26 (cont.) — `index`-ID result + aggregator literature

- **`id=index` (raw normalized agent index) HELPED in-distribution: 0.63 vs 0.58** for none/random, and was
  more reliable (cv-std 0.016 vs 0.035). This *contradicts* the simple "ID is irrelevant" story — but it did
  **NOT fix extrapolation** (→24 0.55, →30 0.00). Interpretation: a positional/index feature gives the
  per-node readout a useful within-distribution symmetry-breaker (helps *fitting*), even though λ₂ itself is
  permutation-invariant — but it doesn't generalize past the training N. (The `random` tag still = no help.)
- **Better-aggregator literature (search).** Standout: **PNA — Principal Neighbourhood Aggregation**
  (arXiv:2004.05718, NeurIPS'20): single aggregators are *provably insufficient* to distinguish neighbourhoods
  in continuous space → **combine multiple aggregators (mean/max/min/std) × degree-scalers**. Plus **learnable**
  aggregators: SoftmaxAgg (learnable temperature → mean↔min↔max), PowerMean, GenAgg (generalised f-mean,
  arXiv:2306.13826). And directly on our N=30 wall: **"Learning to Pool in GNNs for Extrapolation"
  (arXiv:2106.06210)** — the aggregator choice *drives* size-extrapolation. → next aggregator arms: **PNA**
  (prime candidate — we currently pick ONE aggregator; PNA says use them together) + Softmax/PowerMean.
