# Near-Term Experiments (T1–T5) — the current running suite

**Scope.** The near-term blue-team experiments built **2026-07-21**, run via `run_serial32.py`
(heavy 32² units serialized on the single 96 GB card, light 16²/24² units packed in parallel).
Separate from the older Phase-structured SuperBlue campaign in `EXPERIMENT_PLAN.md`; this doc is
the plan for *these* runs. Program goals (blue-team only, adversary set aside): **reasoning,
generalization, behavioral reproducibility, skill transferability.**

**Runners.** `ctde_v0/run_roles_t2.py` (T2) · `ctde_v0/run_lag_t3.py` (T3) ·
`ctde_v0/run_critic_t4.py` (T4) · `ctde_v0/probe_frontier_align.py` + `probe_message_ablation.py`
(T1) · `ctde_v0/run_serial32.py` (launch: `zheavy` + `zlight` tiers).

**Fixed spec (identical on every training run unless noted).** `--explorer-tool frontier_attn`
· `--collision-mask on` (movement collisions only — **never** a connectivity mask) ·
`--mechanism lagrangian --conn-signal global_lambda2 --constraint-threshold 0.5` (θ=0.5 λ₂ floor,
folded into reward, never observed) · `--role-picker expl_relay` · `--w-coverage 3` · the full
**7-channel SLAM occupancy belief** (`--sense-walls --sense-free --boundary`: free/occupied/unknown
+ occ_frontier + boundary) · `--horizon 100` · `--cover-r 0`. **Grade: %-of-optimal** at cover_r=0
(the god-view Voronoi budget caps ~72 % raw coverage there, so absolute % is misleading).

---

## The suite at a glance

| ID | Question it answers | Scale(s) | Arms × seeds | Status (2026-07-21) |
|---|---|---|---|---|
| **T1** | *mechanism probes* — does the frontier-attention read real frontiers? do inter-agent messages change behavior? | 16², 32² (on trained role model) | 2 probes × 2 scales | **DONE** (results below) |
| **T2** | does the hand-coded explorer/relay **role split** beat the flat homogeneous explorer? | open ladder 16²/4, 24²/6, 32²/10 | flat vs role × 3 seeds = 18 | **running** (o16/o24 done; o32 in `zheavy`) |
| **T3** | can the learned-Lagrangian dual **hold a global-λ₂ target** θ∈{0.5,0.7} without collapsing coverage — soft vs lagrangian vs PID? *(the "show the Lagrangian works" experiment)* | 32²/10 | soft_th5 · lag_th5 · lag_th7 · pid_th5 · pid_th7 × 3 seeds = 15 | **running** (all in `zheavy`; just started) |
| **T4** | does a **count-invariant critic** (DeepSets mean/attn) transfer across team size where a fixed-N conv critic degrades? | 16²/4→24²/6→32²/10 from-scratch + zero-shot eval | conv · setpool · setattn × 3 seeds = 27 + eval | **running** (16²/24² done; 32² in `zheavy`) |
| **T5** | is the 32² frontier-attention failure a **learning-signal** problem or a **representation** problem — and is *attention over the whole KB* worth it? | 32²/10 | {sector8, kbattn} × {shared, difference} × 3 = 12 | **PROPOSED (2026-07-21)** — see below |

**Memory constraint (why heavy/light tiers).** One RTX PRO 6000 (96 GB VRAM). A Lagrangian
32²/10 training reserves ~50 GB (measured; the driver is **cuDNN convolution-autotuner scratch**,
not activations — an unconfirmed `--xla_gpu_autotune_level=0` may cut it to ~5 GB and retire the
serialization). Two heavy units OOM, so `run_serial32.py` runs 32² units one-at-a-time.

---

## T1 — mechanism probes  *(DONE; seed 0, trained role model)*

Both probes load a finished checkpoint and inspect one internal computation over ~120 frozen states.

**Findings.**
- **Inter-agent messages are load-bearing** (both scales). Cutting the comm graph moves each agent's
  belief (rel. change **0.65–0.77**) and flips its chosen goal on **44–50 %** of steps. The team
  genuinely acts on neighbors' shared map/features — more so at 32² than 16².
- **The frontier-attention reads frontiers at 16² but collapses at 32².** It points at the truly
  most-uncovered compass sector **48.3 % vs 12.5 % chance (3.9× chance) at 16²**, but **12.7 % ≈ chance
  (1.0×) at 32²**. Even at 16² it overrides the greedy goal only **4.2 %** of the time — a minor rider
  on the hand-coded `frac` heuristic, not the driver.

**Caveats.** Seed 0 only. The frontier probe tests *alignment-with-frontier* only — it does **not**
reveal what the attention tracks *instead* at 32² (a follow-up correlation probe against
neighbor-density / walls / `frac` / boundary is proposed but not run). Re-run across seed 1/2 once
their 32² role checkpoints land.

---

## T5 — Explorer representation × Credit signal  *(NEW, proposed 2026-07-21)*

**Motivation.** T1 shows the 32² frontier-attention is at chance. Two competing causes, with
**opposite testable predictions** — a factorial separates them and simultaneously tests the
"attention over the whole KB" idea head-to-head against the cheaper fix.

- **H_signal:** the learning signal is too weak/diffuse at 10 agents — the shared team reward
  spreads credit thin, and the hand-coded `frac` heuristic shadows the learner (near-zero gradient
  pressure). The 8-sector representation is fine. *Evidence for it:* the **same** 8-sector
  representation works at 16²/4 and dies at 32²/10 — what changed is the learning conditions
  (4→10 agents), not the representation.
- **H_repr:** the 8-sector hand-reduction is a genuine capacity bottleneck (it crushes a 1,024-cell
  map into 8 hand-chosen directional scalars, and gets *coarser* with scale — 8 bins over 4× the
  area). Attention over the KB is needed. *Evidence for it:* 8 bins do lose effective resolution at
  32².

Both factors plausibly degrade with scale, so they are **confounded** in the current runs — hence
the factorial.

**Design — 2×2 factorial @ 32²/10, 3 seeds (12 heavy runs).** Everything else = the fixed spec.

| | credit = **shared** (current, diffuse) | credit = **difference** (exact submodular D_i, already built — sharp per-agent) |
|---|---|---|
| explorer = **sector8** (current 8-bin frontier-attn) | baseline (the T1 failure) | signal-fix on the old representation |
| explorer = **kbattn** (patch multi-head attention over the KB — *the new idea*) | new representation, weak signal | new representation, sharp signal |

**Metrics — mechanism AND outcome, logged across iterations (so undertraining can't masquerade as a
null):**
1. **Frontier-alignment (× chance)** — the T1 diagnostic, computed for *both* explorers (for kbattn:
   does the attention mass concentrate on true-frontier patches vs chance).
2. **Coverage %** (%-of-optimal) and **connectivity** (steps with λ₂ > τ; mean λ₂).

**Pre-registered interpretation** (fixed *before* running — this is what makes it not hand-wavy):

| Observation | Conclusion |
|---|---|
| `difference` rescues `sector8` (alignment chance→~4×, coverage↑) | **signal-limited** → fix credit; KB-attention unnecessary |
| `kbattn` beats `sector8` at matched credit | **representation-limited** → the KB-attention idea is validated |
| `kbattn` wins *only* under `difference` (interaction) | **both needed** — capacity is inert without signal |
| nothing moves | it's elsewhere (task hardness / substrate) → re-diagnose, don't build more |

**To build:** the `kbattn` explorer (see KB-attention note below) + an alignment metric for it.

**Cost.** 12 heavy 32² runs. Serial (current mode) ~10–12 h; with the autotuner fix confirmed
(dense parallel) ~3 h — a third reason to confirm that flag.

**Open decisions (before build).**
- Add a 5th/6th arm `sector8_nofrac` (heuristic crutch removed) to **isolate shadowing** from
  diffuse-credit? (+6 runs, cleaner attribution.)
- Patch size for kbattn: **4×4 → 64 tokens at 32²** (default; balances resolution vs O(tokens²) cost).
- Sequencing: after the current serial run, or confirm the autotuner fix first to run it in ~3 h.

---

## NOTE — Attention over the whole KB  *(the richer explorer idea; proposed, UNBUILT)*

The `kbattn` arm of T5 **is** this idea; T5 is its first controlled test. Recorded here in full
because it also doubles as the substrate for a future L3 reasoner.

**What runs today (for contrast).** The `frontier_attn` explorer (`nets.py:388`) does **not** attend
over the KB. It applies a fixed geometric formula (`sector_frontier_features`) that reduces the
belief (7, H, W) to a **(K=8, 2)** table — per compass sector, the *hand-computed* fraction of cells
that are frontier — then a small learned softmax weights those **8 numbers** to bias which of 8
compass waypoints the agent heads to. The learned part **never sees the map**; it sees 16 pre-digested
scalars, and can only ever say "head more toward direction k." It cannot attend to a wall gap, a
specific region, or a teammate-discovered frontier — those are discarded in the reduction.

**The proposed mechanism.** Tokenize the belief map (P patches, each a learned embedding of a local
region with all 7 channels intact — **not** per-cell, to bound the O(tokens²) cost and memory, which
matters given the 50 GB VRAM fight), form learned Q/K/V, and apply **multi-head attention over the map
tokens**: weights = softmax(QKᵀ/√d), out = weights·V. Unlike the 8-sector head, the weights are
**learned and data-dependent** over spatial locations, and there is **no fixed direction-binning** — a
token is the same size at 16² and 32², so it does not lose resolution with scale.

**Why it might help.**
- **Scale-invariance** — the direct structural fix for the T1 resolution-loss-with-scale.
- **Expressiveness** — can represent decisions the 8-sector head physically cannot.
- **Reasoner substrate** — attention over the KB is the mechanism a future deliberative L3 would use to
  *read* the map; the 8-scalar gate never could.

**Why it is NOT an automatic win (the caution T5 is designed to check).**
- It fixes **representation**, but T1's failure is plausibly dominated by **signal** (diffuse credit +
  heuristic shadowing) — a higher-capacity module with the *same* weak signal tends to learn spurious
  features or collapse (arguably what the 8-sector head already did). **High capacity + thin gradient
  is a known failure mode.** → *Prove it's a representation bottleneck (T5) before paying for the
  upgrade.*
- **Cost/memory:** attention over map tokens is O(tokens²); patch-based control is mandatory, and we
  just fought a 50 GB VRAM wall.
- **Sample efficiency:** more parameters need more signal/iterations; may not converge at the current
  1,500–2,000-iter budget.
- **Discards a working prior:** the hand-coded reduction *encodes* "go toward unexplored." A blank
  attention must relearn that from a weak reward — so **initialize `kbattn` as a residual on the
  frontier prior** rather than a cold start (keeps the T5 comparison about the *added* capacity).

**Open questions.** Patch size / per-cell vs patch; frontier-prior residual vs from-scratch; training
budget; whether the same module then serves as the L3 reasoner's read head.

**Status.** Unbuilt. Gated on T5's outcome (build only if T5 shows representation is the bottleneck).
Relates to the deferred L3 reasoner in `L3_REASONER_whitepaper.html` / `COGNITION_DESIGN.md`.
