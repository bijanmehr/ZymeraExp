# Fiedler-Value Estimation — Results & Parameters

Consolidated, parameter-complete record of every estimator run in this study. Pairs with the dated
narrative log in `FINDINGS.md` and the design docs `EXPERIMENT.md` / `ARCHITECTURES.md`. Raw records live
in `results/sweep_*.jsonl` (one JSON line per config, full echoed config + metrics).

**Goal of the study.** Estimate a team's **global algebraic connectivity λ₂ (the Fiedler value)** from each
agent's **local, partial view** of the time-varying comm graph, to ≥ **0.95 accuracy** with the fewest
historical steps. Target ops: teams of **N = 4–30**.

**Standing caveat (applies to every number here).** Training/eval data is the **hard-connectivity-guardrail
anti-crowding dispersion** regime — a *proxy* for a coverage swarm's comm graphs (always-connected,
dispersed), **not** a real coverage mission and **not** an RL-trained policy. Real-mission validation
(coverage-policy data, then on-policy fine-tuning) is still owed.

---

## 1. Experimental setup (shared by all 27 runs)

**Model — `ConfigurableGCRN`** (Equinox/JAX): per-node encoder → `n_rounds` of message passing → per-node
GRU over an `H`-step history window → 3 readout heads. The varied axes (below) only change the
message-passing round; everything else is fixed.

| group | parameter | value |
|---|---|---|
| **architecture** | hidden width | **128** |
| | message-passing rounds `n_rounds` | **2** |
| | attention heads (attn/multihead ops) | **4** |
| | history window `H` (timesteps per sample) | **5** |
| | readout heads | `logl2` (λ₂, Huber on log), `cflag` (connected, BCE), `logsig` (uncertainty, NLL) |
| **training** | optimizer | AdamW, `lr=3e-4` cosine-decayed, `weight_decay=1e-4`, global-norm clip 1.0 |
| | max steps | **8000** (early-stop, `patience=15` evals) |
| | early-stop metric | val **median relative error** in linear λ₂ over connected-window real nodes |
| | batch | 128 · `val_frac` 0.2 · `dropedge` 0.1 · agreement weight `agree_w` 0.05 |
| | loss | `Huber(logλ₂) + 0.3·BCE(cflag) + 0.1·NLL(logsig) + 0.05·node-variance`, all node-masked |
| **data (guardrail)** | regime | hard-connectivity-guardrail anti-crowding dispersion (always-connected) |
| | training sizes `train_N` | **[4, 8, 12, 16, 20]** (padded to N_max=20, node-masked) |
| | per-N grid | `grid = max(8, round(√(N/0.04)))` → density ≈ 0.04 agents/cell, fixed across N |
| | comm radius | 5 · obstacles 0 · spawn radius 2 |
| | rollouts | **5 episodes × 150 steps** per N, windowed at H=5 |
| **evaluation** | in-distribution | fresh held-out episodes at `eval_N=[6,10,20]` (separate seed) |
| | cross-validation | **5-fold** at **N=20** (`cv20_mean ± cv20_std`) |
| | extrapolation | **zero-shot** at `N ∈ {24, 30}`, no retrain (`→24`, `→30`) |
| **metric** | accuracy | **1 − median relative error** on linear λ₂ (oracle = `eigvalsh` of soft-weighted Laplacian) |

**Oracle label.** True λ₂ = 2nd-smallest eigenvalue of the soft-weighted graph Laplacian (`fidler/fiedler.py`),
computed by exact eigendecomposition; the network never sees it, only per-agent local features + messages.

**Eval note.** `predict_configurable` evaluates with `jax.lax.map` (per-sample), not `jax.vmap` — a GPU XLA
layout miscompile hit `vmap` over single-head attention at N=20 (`Reshape 8x1024 -> 8x2x32x32`); `lax.map` is
numerically identical and dodges it. **Hardware:** 1× RTX PRO 6000 Blackwell (96 GB) + 24 cores; per-config
wall ≈ 1000–2000 s (`wall` column). Run on commit history through `8a299b7`.

**Reading the columns.** `acc` = in-distribution accuracy · `cv20 ±std` = 5-fold mean ± fold std at N=20 (std =
reliability) · `→24`,`→30` = zero-shot extrapolation accuracy · `valerr` = final val median-rel-err ·
`steps` = early-stop step (of 8000). `cflag` is omitted: it is **1.00 in every run** (guardrail data is always
connected, so the connected/not classifier is trivial here).

**Noise floor.** The `value / id=none` baseline was (re)run in three separate sweeps and landed at
**0.580 / 0.578 / 0.572** — so accuracy differences **below ~0.01 are within replication noise** (RNG-stream
differences in the augmentation pipeline), not real effects.

---

## 2. Results — all 27 runs

### 2.1 Aggregation operator (content = value, id = none) — `sweep_ops_value`
The 8 message-aggregation ops at a fixed message content.

| op | acc | cv20 ± std | →24 | →30 | valerr | steps |
|---|---|---|---|---|---|---|
| mean | 0.580 | 0.609 ± .039 | 0.559 | 0.00 | 0.432 | 1200 |
| gcn | 0.635 | 0.628 ± .018 | 0.458 | 0.00 | 0.294 | 6550 |
| **max** | **0.656** | 0.663 ± **.0055** | 0.471 | 0.00 | 0.279 | 3000 |
| sum | 0.642 | 0.628 ± .019 | 0.518 | 0.00 | 0.285 | 6750 |
| attention (1-head) | 0.557 | 0.620 ± .023 | 0.387 | 0.00 | 0.428 | 1250 |
| **multihead (4-head)** | **0.659** | **0.670** ± .018 | **0.556** | 0.00 | 0.247 | 6550 |
| gated | 0.637 | 0.651 ± .010 | 0.431 | 0.00 | 0.270 | 7150 |
| laplacian | 0.639 | 0.648 ± .017 | 0.503 | 0.00 | 0.290 | 4500 |

### 2.2 Agent identity (content = value) — `sweep_ids`
ID feature appended per node: `none` / `random` (per-episode permutation-equivariant tag) / `index` (raw
normalized agent index).

| op | id | acc | cv20 ± std | →24 | →30 |
|---|---|---|---|---|---|
| mean | none | 0.578 | 0.628 ± .036 | 0.568 | 0.00 |
| mean | random | 0.578 | 0.632 ± .035 | 0.531 | 0.00 |
| mean | **index** | **0.631** | 0.646 ± .016 | 0.549 | 0.00 |
| attention | none | 0.590 | 0.599 ± .043 | 0.351 | 0.00 |
| attention | random | 0.600 | 0.565 ± .047 | 0.333 | 0.00 |
| attention | **index** | **0.627** | 0.636 ± .020 | 0.493 | 0.00 |

### 2.3 Message content / connectivity-margin (id = none) — `sweep_margin`
Message content: `value` (raw z_j) / `margin` (z_j + dist/comm_r, the link-fragility margin) / `geom`
(z_j + full edge geometry [dx,dy,dist]/comm_r).

| op | content | acc | cv20 ± std | →24 | →30 |
|---|---|---|---|---|---|
| mean | value | 0.572 | 0.611 ± .039 | 0.481 | 0.00 |
| mean | **margin** | 0.641 | 0.655 ± **.0097** | 0.542 | 0.00 |
| mean | **geom** | 0.643 | 0.653 ± .011 | 0.485 | 0.00 |
| attention | value | 0.562 | 0.592 ± .046 | 0.422 | 0.00 |
| attention | margin | 0.589 | 0.618 ± .024 | 0.250 | 0.00 |
| attention | geom | 0.595 | 0.598 ± .037 | 0.311 | 0.00 |

### 2.4 Signal strength × identity (op = max) — `sweep_signal`
`signal=on` adds a continuous path-loss link weight `exp(-3·(dist/comm_r)²)` → soft-weighted aggregation + a
per-node mean-neighbor-signal feature.

| op | id | signal | acc | cv20 ± std | →24 | →30 |
|---|---|---|---|---|---|---|
| max | none | off | 0.659 | 0.665 ± .013 | 0.474 | 0.00 |
| max | random | off | 0.637 | 0.649 ± .012 | 0.505 | 0.00 |
| max | none | **on** | 0.661 | **0.676** ± .018 | 0.471 | 0.00 |
| max | random | on | 0.656 | 0.671 ± .029 | 0.476 | 0.00 |

### 2.5 Learned content (id = none) — `sweep_learned_{mean,max,multihead}`
`learned` = each neighbor message is `MLP(z_j)` (trainable transform, no new input info).

| op | content | acc | cv20 ± std | →24 | →30 | vs. value (acc / cv20 / →24) |
|---|---|---|---|---|---|---|
| mean | learned | 0.641 | 0.654 ± .0096 | 0.547 | 0.00 | 0.580 / 0.609 / 0.56 → **+0.06**, far tighter |
| max | learned | 0.654 | **0.680** ± .015 | **0.568** | 0.00 | 0.656 / 0.663 / 0.47 → acc tie, **best cv20**, big extrap gain |
| multihead | learned | 0.653 | 0.662 ± .0062 | 0.571 | 0.00 | 0.659 / 0.670 / 0.56 → acc ~tie |

---

## 3. Leaderboards (what actually won)

- **Best in-distribution accuracy:** `max+signal` **0.661** ≈ `multihead+value` 0.659 ≈ `max+value` 0.656/0.659.
- **Best CV (the reliability of the 0.95-target metric):** `max+learned` **0.680** > `max+signal` 0.676 >
  `multihead+value` 0.670.
- **Most reliable (lowest fold std):** `max+value` **±0.0055** < `multihead+learned` ±0.0062 <
  `mean+learned` ±0.0096 ≈ `mean+margin` ±0.0097.
- **Best extrapolation →24:** `multihead+learned` **0.571** ≈ `max+learned` 0.568 ≈ `mean+id_none` 0.568 >
  `multihead+value` 0.556.
- **Best single all-round config so far:** **`max + learned`** (acc 0.654, top cv20 0.680, top-tier →24 0.568).

---

## 4. Conclusions

1. **Two co-best aggregators: `max` and `multihead` (~0.66).** `max` is the most *reliable* (fold std .0055);
   `multihead` has the best raw accuracy + extrapolation. **Single-head `attention` is the worst (0.557)** and
   extrapolates worst — it is the only attention variant in the id/margin sweeps, which earlier made attention
   look bad in general; the 4-head op is actually top-tier. `gcn/sum/gated/laplacian` cluster 0.635–0.642.
2. **Edge-distance content helps:** `margin`, `geom`, `signal` all lift `mean` 0.57→0.64; `signal` gives `max`
   the best CV (0.676). `learned` — despite adding *no new input* — also lifts `mean` (+0.06) and gives `max`
   the best CV (0.680) and best →24 (0.568); so it is **not** dominated (the original prior was wrong).
3. **`index` ID helps both ops (+~0.05); `random` does nothing.** λ₂ is permutation-invariant, so `index` acts
   as a positional *fitting* aid, not an identity signal.
4. **In-distribution ceiling ≈ 0.66 — structural.** Across all 27 message-design configs nothing exceeds ~0.66
   (≈ 34% median error). Aggregator/content/id choice moves *reliability and extrapolation* far more than it
   moves the accuracy ceiling. → message-design stacking alone is unlikely to reach the 0.95 target.
5. **N = 30 wall: `→30 = 0.00` for all 27 configs** (`→24 ≈ 0.33–0.57). The model trained on N ≤ 20 cannot
   calibrate the λ₂ *scale* at N=30. This is a **training-distribution** problem — needs larger N in training
   (also the most likely lever past the 0.66 ceiling) or an explicit size-transfer regularizer, **not** a
   message-design tweak.
6. **`cflag` head is trivial here** (1.00 everywhere) because guardrail data is always connected — uninformative
   until data includes disconnection events.

**Implied next levers (outside the message-design space):** train on larger N (24/30 in-pool) ·
size-transfer regularization · more / richer data · target reformulation. The message-design grid
(`{max,multihead,…,PNA} × {value,learned,geom,margin,signal} × {none,index}`) remains worth running to settle
whether the good ingredients *stack*, but should be paired with a training-distribution experiment.
