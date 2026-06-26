# Fiedler Estimation — Networks, Hyperparameters & Message Types (full spec)

Companion to `EXPERIMENT.md`. Every estimator's architecture, every hyperparameter, and every message type —
documented for exact reproducibility. **Framework: JAX — Equinox (modules) + optax (training)**, both already
in the zymera venv (no PyTorch, no new deps). The learned modules stay native with the JAX datagen and plug
straight into the JAX policy. Our comm graphs are **small & dense** (N ≤ 30, `(N,N)` adjacency), so GNN
message-passing is plain `jnp` matmuls (`Â @ h`) — **no jraph / PyG needed**. `chex` for shape asserts.

---

## 0. Shared conventions

### 0.1 Per-agent node features `f_i ∈ ℝ^F`  (F = 8)
All locally computable each step; standardized (z-score) using train-split statistics.

| # | feature | definition |
|---|---|---|
| 1 | `deg_norm` | own degree `dᵢ / (N_known − 1)` (or `dᵢ / 8` if roster unknown) |
| 2 | `deg_log` | `log(1 + dᵢ)` (raw neighbor count, compressed) |
| 3 | `mean_nbr_dist` | mean distance to in-range neighbors (÷ comm_radius) |
| 4 | `std_nbr_dist` | std of those distances |
| 5 | `mean_nbr_deg` | mean degree of 1-hop neighbors (needs 1 exchange) |
| 6 | `std_nbr_deg` | std of neighbor degrees |
| 7 | `reach_frac` | `|distinct IDs seen in last W=H steps| / N` (the fragment signal) |
| 8 | `steps_since_new` | steps since a *new* ID entered range (÷ H) |

### 0.2 Per-neighbor / edge features `e_{ij} ∈ ℝ^5`  (message-passing methods)
`[Δx_ij, Δy_ij` (÷ comm_radius)`, dist_ij` (÷ comm_radius)`, deg_j_norm, reciprocal_flag]`.

### 0.3 Targets (3 heads)
- **regression:** `y = log(λ₂ + 1e-6)` — log handles the dynamic range and emphasizes the near-0 regime.
- **connected flag:** `c = 1[λ₂ > τ]`, `τ = 1e-3`.
- **uncertainty:** predict `log σ` (heteroscedastic).

### 0.4 Loss
`L = Huber_δ=1(ŷ, y)  +  0.3·BCE(ĉ, c)  +  0.1·[ ½·exp(−2 logσ)·(ŷ−y)² + logσ ]`
Report **accuracy in linear λ₂ space**: `1 − median(|e^ŷ − λ₂| / λ₂)`.

### 0.5 Optimization defaults (overridden per method in §4)
`AdamW` · lr `3e-4` · weight-decay `1e-5` · **cosine schedule, 5 % warmup** · grad-clip global-norm `1.0` ·
batch `128` · ≤ `200` epochs · **early-stop** patience `15` on val median-rel-error · `≥3` seeds ·
embedding/hidden dim **D = 128** · activation **GELU** · **LayerNorm** + residuals in all blocks.

---

## 1. Heuristic / non-learned methods

### 1.1 Degree-regression  *(floor-ish learned baseline)*
- **Input:** `[dᵢ, mean dᵢ over H, std]` (3 scalars).
- **Net:** MLP `3 → 32 → 32 → {ŷ, ĉ}`; GELU.
- **HP:** lr `1e-3`, batch `256`, epochs `60`. No message passing.

### 1.2 Temporal net — own-history (two variants)
Own features `f_i` sequence, shape `(H, 8)`. **No messages** (own-obs only — the "zero-comms" ceiling test).
- **GRU:** 2 layers, hidden `64`; last hidden → MLP `64 → 32 → {ŷ, logσ, ĉ}`. dropout `0.1`.
- **TCN:** 3 dilated Conv1d (dilations 1-2-4, channels `64`, kernel 3, causal) → global-avg-pool → head.
- **HP:** lr `3e-4`, batch `128`, epochs `120`.

### 1.3 Decentralized power-iteration (Yang)  *(analytic — no training)*
- **State per node:** scalar `xᵢ` (Fiedler-vector estimate) + running mean estimate `μ̂ᵢ`.
- **Per round** (`rounds = H`): mean-consensus `μ̂ᵢ ← μ̂ᵢ + γ Σ_{j∈Nᵢ}(μ̂_j − μ̂ᵢ)`; deflate `xᵢ ← xᵢ − μ̂ᵢ`;
  diffuse `xᵢ ← xᵢ − ε·(Lx)ᵢ`, `(Lx)ᵢ = Σ_{j∈Nᵢ}(xᵢ − x_j)`; renormalize via consensus.
- **Readout:** Rayleigh quotient `λ̂₂ = (xᵀLx)/(xᵀx)` via sum-consensus.
- **Params (grid-searched, not learned):** `ε ∈ {0.05, 0.1, 0.2}`, consensus weight `γ = 0.5/d_max`,
  inner-consensus steps `{1 (amortized), 3}`. **Message = C0 (value `x_j`)**.

### 1.4 Random-walk estimator  *(analytic)*
- One walker per component; `walk_len = H`; estimate λ₂ from the lag-1 autocorrelation / return-rate of the
  visited-node value sequence. **Params:** restart prob `0`, estimator EMA `0.9`.

### 1.5 Degree-only / persistence  *(lower bound)*
- `λ̂₂ =` train-fit monotone curve of mean-degree, **0 rounds**. (Reference floor.)

---

## 2. Learned graph methods

### 2.1 Snapshot GNN  *(spatial only)*
- **Encoder:** `f_i → Linear(8→128)`.
- **k message-passing layers** (`k ∈ {1,2,3}`, default 2), each:
  `msg_{j→i} = MLP([h_i, h_j, e_{ij}] : 261→128)`; aggregate (see §3.3); `h_i ← LN(h_i + GELU(MLP(agg)))`.
- **Readout:** per node `MLP(128 → 64 → {ŷ, logσ, ĉ})`.
- **HP:** lr `3e-4`, batch `64` graphs, epochs `120`, heads `4` (if attention). Snapshot only ⇒ needs more `k`.

### 2.2 Recurrent GNN — **GCRN (main candidate)**  *(spatio-temporal)*
For each of `H` steps: run `k = 2` spatial message-passing rounds (as §2.1) → node embedding `z_i^t`; then a
**graph-conditioned GRU** updates per-node memory: `h_i^t = GRU(z_i^t, h_i^{t-1})`, hidden `128`. After `H`
steps, readout from `h_i^H`.
- **Dims:** node-enc `8→128`; edge-enc `5→128`; MP rounds `k=2`; GRU hidden `128`; heads `4`.
- **Heads:** `ŷ` (`128→64→1`), `logσ` (`128→1`), `ĉ` (`128→1`).
- **Message:** configurable via §3 (C0–C3 × scheme × aggregation) — this is where the message-design sweep lives.
- **HP:** lr `3e-4`, batch `64` sequences, epochs `150`, weight-decay `1e-5`, grad-clip `1.0`, cosine+warmup,
  **BPTT over H** (truncate at `H≤21`).

### 2.3 Unrolled power-iteration net  *(analytic ↔ learned bridge)*
`H` layers, each a **learned** power-iteration step: `x ← x − εₗ·(Lx) + MLP_corr,ₗ(x, agg_nbr)`, with **per-layer
learned `εₗ`** and a tiny shared correction MLP (`128→64→1` on pooled state); learned mean-deflation;
**learned Rayleigh readout**. Strong spectral prior, few params (~`H` scalars + 2 small MLPs).
- **HP:** lr `1e-3`, batch `64`, epochs `100`.

### 2.4 Centralized GNN  *(privileged upper bound)*
Same blocks as §2.1 but message passing runs over the **full** graph with `k = graph_diameter` (or 10) rounds,
**global mean-pool → one λ̂₂**. Not deployable; defines the **full-observability ceiling**.

---

## 3. Message types (full spec) — the message-design axis

A message from `j` to `i` is `m_{j→i} ∈ ℝ^{d_msg}`. Three orthogonal sub-axes.

### 3.1 Content (what's in the message)
| id | message vector `m_{j→i}` | `d_msg` |
|----|--------------------------|---------|
| **C0** value-only | `[x_j]` (consensus/estimate scalar) | 1 |
| **C1** + structure | `[x_j, deg_j, n_seen_j]` | 3 |
| **C2** + geometry | `[x_j, deg_j, Δx_ij, Δy_ij, dist_ij]` | 5 |
| **C3** **learned** | `m_{j→i} = MLP_enc(h_j) ` (or `MLP_enc([h_j, e_{ij}])`) — net decides what to send | **16** (knob: 8/16/32) |

### 3.2 Passing scheme (how messages flow)
| id | scheme | detail |
|----|--------|--------|
| **P0** synchronous | all agents exchange simultaneously, **`k` rounds/step** (`k ∈ {1,2,3}`) |
| **P1** gossip | random pairwise / asynchronous activation; one exchange per active pair |
| **P2** recurrent | **1 round/step**, state carried in the GRU across `H` (the GCRN default) |

### 3.3 Aggregation (how `i` combines incoming messages)
| id | aggregator | detail |
|----|-----------|--------|
| **mean** | `agg_i = mean_j m_{j→i}` | permutation-invariant, cheapest |
| **attention (TarMAC)** | `α_ij = softmax_j(q_i·k_j/√d)`, `agg_i = Σ_j α_ij v_j` | 4 heads, q/k/v dim 32 |
| **gating (IC3Net)** | learned gate `g_j = σ(MLP(h_j)) ∈ [0,1]` decides **whether `j` broadcasts**; train with a bandwidth penalty `λ_bw·Σ g_j` (`λ_bw ∈ {0, 1e-3, 1e-2}`) | drives the bandwidth Pareto |

### 3.4 Bandwidth accounting
`bytes/agent/step = d_msg × (avg #neighbors broadcast-to) × 4 (float32) × (rounds/step)`. Logged for every run
so accuracy is plotted against **bytes/agent/step** (the message Pareto). C3-learned + gating is expected to
dominate this frontier.

**Headline message question:** does **C3 / attention / gating** reach 95 % at **smaller `H` and/or fewer
bytes** than **C0–C2**? Sweep order: content **C0→C3** under P2; then schemes **P0/P1/P2**; then gating `λ_bw`.

---

## 4. Consolidated hyperparameter table

| Method | params | lr | batch | epochs | hidden | MP `k` | seq `H` | notes |
|---|---|---|---|---|---|---|---|---|
| Degree-reg | ~1k | 1e-3 | 256 | 60 | 32 | — | agg over H | own-obs |
| GRU | ~50k | 3e-4 | 128 | 120 | 64 | — | H | own-obs |
| TCN | ~60k | 3e-4 | 128 | 120 | 64 | — | H | own-obs, causal |
| Power-iter (Yang) | 0 (ε,γ) | — | — | — | — | rounds=H | — | analytic, C0 |
| Random-walk | 0 | — | — | — | — | — | walk=H | analytic |
| Snapshot-GNN | ~300k | 3e-4 | 64 | 120 | 128 | {1,2,3} | 1 | spatial only |
| **GCRN (main)** | ~500k | 3e-4 | 64 | 150 | 128 | 2 | H | spatio-temporal, BPTT |
| Unrolled-PI | ~20k | 1e-3 | 64 | 100 | 64 | =H | H | spectral prior |
| Centralized-GNN | ~300k | 3e-4 | 64 | 120 | 128 | =diam | 1 | upper bound |

Shared (all learned): AdamW, wd 1e-5, cosine+5%-warmup, grad-clip 1.0, GELU, LayerNorm, dropout 0.1,
early-stop patience 15, ≥3 seeds. Sweep knobs: `H ∈ {1,2,3,5,8,13,21}`, message {C0–C3}×{P0–P2}×{mean/attn/gate},
**network-size (§7)**, **regularization (§6)**. The full cross-product is large → run **OFAT** (sweep one axis
at a time from the default), matching the campaign's staged philosophy.

---

## 5. Training & assessment protocol

- **Multi-size training pool:** episodes from `N ∈ {4, 8, 12, 16, 20}`, mixed per batch; densities {sparse,
  medium, dense}; drifting-graph (primary) + frozen control.
- **Standardize** inputs with train-split stats; predict `log λ₂`.
- **In-distribution robustness — 5-fold CV on N = 20:** partition the 20-agent episodes into 5 folds; for each
  fold, train on the full pool **minus that fold's 20-episodes** and evaluate on the held-out fold → report
  **mean ± std accuracy** at N = 20. (Robust, low-variance number at the size of interest.)
- **Generalization tests (zero-shot, no retrain):** interpolation `N ∈ {6, 10}`; **extrapolation `N ∈ {24,
  30}`** (beyond the max train size 20).
- **Per-run config:** `method · H · message(content/scheme/agg/λ_bw) · net_size(width/k/layers) · reg · N · density · seed`.
- **Report:** accuracy-vs-`H` curves · **min-`H`-for-95 %** table · **5-fold-CV@20** (mean±std) ·
  extrapolation-to-{24,30} drop · **bandwidth Pareto** · partial-observability gap (best decentralized vs
  centralized-GNN).
- **Pass bar:** ≥ 95 % (median rel-err ≤ 5 %) — at min-`H`, holding on 5-fold@20 and out to 30.

---

## 6. Regularization

For this study regularization is the **primary lever for size-extrapolation (train ≤ 20 → 30)** and a tool for
**min-`H` parsimony** — not just overfitting control (Round-2: GNNs converge to bad minima that overfit small
graphs and break under degree/size shift). **Swept axis:** `reg ∈ {off, generic, +size-transfer, +domain}`.

**6.1 Architectural (by design — highest leverage)**
- **Normalized aggregation, NEVER sum** — mean / attention / degree-normalized. Sum-aggregation magnitudes
  scale with N and degree → destroys extrapolation. *(Locks §3.3 to mean/attn; sum is not an option.)*
- Residuals + LayerNorm + shallow `k=2` (over-smoothing control); jumping-knowledge if deeper.
- Optional **spectral-norm / Lipschitz cap** on weights — stability for the spectral target.

**6.2 Size-transfer (the → 30 make-or-break)**
- **Subgraph-sampling augmentation** — train on random subgraphs / coarsenings of larger graphs.
- **SizeShiftReg** (arXiv:2207.07888) — penalize graph vs size-shifted-version discrepancy.
- **DropEdge** — drop comm edges in training (doubles as comm-dropout realism).

**6.3 Domain-specific (free, novel)**
- **Node-agreement penalty `λ_agree · Var_i(λ̂₂ᵢ)`** — all agents estimate the *same* global λ₂, so penalize
  disagreement across nodes. Self-supervised, stabilizing — the *learned* analog of the dropped runtime
  consensus. `λ_agree ∈ {0, 0.01, 0.1}`.

**6.4 Generic** — weight-decay (sweep `1e-5…1e-3`), dropout `0.1`, early-stop (patience 15), **SWA**,
label-smoothing on the connected-flag, Huber (robust loss).

**6.5 Temporal (GRU / GCRN over H)** — variational/recurrent dropout (shared mask across t), zoneout,
truncated BPTT.

**6.6 min-`H` tie-in** — optional **L1 / information-bottleneck on temporal attention** to sparsify which of
the `H` steps are used → *finds* the smallest sufficient history.

**Defaults-on (cheap, high-value):** normalized aggregation · DropEdge · subgraph augmentation ·
node-agreement penalty · weight-decay · early-stop. Heavy ones (SizeShiftReg, Lipschitz) = ablation knobs.

---

## 7. Network-size sweep

A **second minimization axis alongside `H`:** find the **smallest network (fewest params)** that clears 95 % at
min-`H` **and** holds extrapolating to 30. The interesting trade-off — **more capacity helps in-distribution
fit but can HURT extrapolation** (overfits small-graph training) — so the extrapolation-vs-params curve may
**peak then drop**.

**7.1 Size knobs**
| knob | values | applies to |
|---|---|---|
| width `D` (hidden / embed) | **{32, 64, 128, 256}** | all learned |
| MP rounds `k` | {1, 2, 3} | GNN / GCRN |
| recurrent layers | {1, 2, 3} | GRU / TCN / GCRN |
| readout depth | {1, 2} | all |
| message dim `d_msg` | {8, 16, 32} | C3 learned |
| attention heads | {2, 4, 8} | attention agg |

Default: `D=128, k=2, layers=1–2, d_msg=16, heads=4`.

**7.2 Strategy (OFAT, lean)** — primary: **width ladder `D ∈ {32,64,128,256}`** at default depth; secondary:
depth (`k` / recurrent layers) at the best width. Report everything **against parameter count + inference
FLOPs** (the x-axis), not raw config.

**7.3 Headline questions**
- Is there a **capacity sweet spot for extrapolation-to-30** (does big overfit)?
- **Smallest net** hitting 95 % @ min-`H` @ extrapolate-30?
- Does **width trade off against `H`** (wider net ↔ less history needed)?

**7.4 Interaction with §6** — bigger nets lean harder on the size-transfer regularizers. Run the width ladder
with defaults-on regularization, then a targeted **size × reg** cell at the largest width to confirm reg
rescues big-net extrapolation.
