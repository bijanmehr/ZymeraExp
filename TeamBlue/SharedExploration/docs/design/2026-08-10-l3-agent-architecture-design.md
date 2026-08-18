# L3 Agent Architecture — Design Specification

**Status:** draft for review · **Date:** 2026-08-10 · **Scope:** design only (build is GPU-gated)
**Depends on:** `2026-08-10-zymera2-world-design.md` (the world/bridge contract this agent runs against)
**Evidence base:** four-scout L3 literature verdict (`lit-l3-goal-representation-verdict` memory; SAM ·
ANS/MAANS · ARiADNE/MARVEL · Burgard/Corah/Grimsman · Nachum/Dabney/TempoRL · UPDeT/SF-GPI), plus the
2026-08-09/10 code verification of the current stack.

---

## 1 · Verdict being implemented

The three-clock direction (goal head → planner → tracker) is confirmed as the field-standard shape.
Three components are replaced, each indicted by direct published evidence:

| Component | Indictment | Replacement |
|---|---|---|
| 9-way stride-3 stencil | SAM ablation: small steering-command sets lose ~4× to map-endpoint heads; MAANS: spatial-categorical > point regression | Goal representation ablation arms G0/G1/G2 (§4) |
| Per-step goal resampling | Unsupported anywhere; SAM shows commitment-wavering; Nachum: commitment *is* hierarchy's benefit | Commit-and-hold with event-triggered release (§5) |
| No goal coordination | Refuted locally (flood proof) and by 20 years of MRS | Claim-round + intention visibility (§6) |

**Fixed design criteria (2026-08-10):** (1) SLM-portable L3 contract — "K proposed candidates +
teammates' claims → choose & commit"; (2) emergent roles, label-free delivered-flow credit; (3)
mission-generality demos = SharedExploration · TetherRelay · Cornering (Sense · Organize · Act — the
three families); (4) the G×C×X ablation is the evidence instrument. Parked to the red phase: potential
games, EBM detector (with the conditional-baseline objection on record).

## 2 · The agent as one closed module

Everything cognitive lives agent-side (per the env/agent boundary ruling), carried in policy state,
consuming only the harness-assembled observation {sensing readouts · mail · delivered-adjacency row ·
declared statics}:

```
AgentState (the policy carry, per agent):
  belief:  KB grids (known-ness · known-wall · optional extensions)   ← kb.fold(obs, belief)
  goal:    committed cell (2,) · age · active flag                     ← commitment machinery (§5)
  claims:  teammates' committed goals as last-heard                    ← from mail
  h:       optional recurrent hidden (off by default)

policy(obs, AgentState, key) -> (action, outbox, AgentState')
```

- **`kb.py` (zymera_lab):** the reference belief — fold = monotone union of sensed patch ∪ mail
  content (bit-compatible with v1 semantics); outbox = authored from own belief (truthful for blue;
  the red lie seam exists from day one); extension slots (staleness stamps, entity beliefs) as data.
- **Claims transport (staged, per 2026-08-10 decision):** claim semantics defined once — (committed
  goal, age), visible within delivered comm range. v1 transport = claims ride the outbox payload
  through the zymera2 bridge (they are just payload fields — no special mechanism); the bridge's
  delay/dropout/occlusion apply to claims automatically. This supersedes the earlier "pipeline plane"
  workaround: with zymera2's opaque payloads, claims-in-mail is the *natural* v1 path.
- The whole module (fold → propose → plan → score → claim) is the SLM-swap unit and the unit red
  compromises. No dormant legacy heads: the v2 actor is built fresh (roles/selector/flock/compass do
  not carry over; the ablation controls are configs of the new head, not resurrected modules).

## 3 · Backbone

CNN over belief planes → 2-round message passing over the **delivered** adjacency row (fixing the
potential-vs-delivered leak) → per-agent feature map **and** pooled context `z`. Two changes from v1:

1. **The goal head reads pre-pooling spatial features** (position survives; the pooled `z` remains for
   comms context) — the verified position-blindness fix.
2. **The planner field from own position enters as an input plane** — the geodesic
   distance-from-me-through-known-walls reference (wall-aware, size-invariant), computed by the same
   L2 operator the agent already runs. L2 becomes part of L3's eyes.

Count/size invariance doctrine unchanged: convolutional heads, per-edge/per-node ops, no shape-bound
parameters; Kinetix-tier static caps from the zymera2 contract.

## 4 · Goal representation — the G axis

One head architecture, three candidate sources (G1 is the dense limit of G2; the ablation is config):

- **G0 — stencil (control):** the 9 fixed offsets, stride 3. Kept as the published baseline arm.
- **G1 — egocentric coarse spatial-softmax:** fully-convolutional logits over every stride-3 cell of
  the believed map (fixed `⌈H/3⌉×⌈W/3⌉` grid under the static caps, arena/known-masked). Backed by
  SAM (dense > steering) and MAANS (spatial-categorical > regression).
- **G2 — top-K proposed candidates + shared scorer:** a mission-supplied **proposer** (pure function
  of the belief) emits K candidate cells + features; a masked pointer/attention head scores them.
  Features per candidate: egocentric offset · frontier mass · **L2 field value** (reachability) ·
  visited flag · connectivity-risk proxy · **claimed-by-teammate flag**. K fixed (≈12), padded+masked.
  - Proposers as data (the mission-generality mechanism): coverage = frontier-cluster centroids ·
    tether = cut-path relay slots · cornering = containment posts. Backbone/scorer mission-blind.
- Mission identity enters ONLY via proposer + reward terms (task layer); no task embeddings (they
  degenerate to one-hot across discrete missions), no SF factorization (breaks on k-disjoint rewards).

## 5 · Commitment — the C axis

- **C0 — per-step (control).**
- **C1 — commit-and-hold:** goal + age in AgentState; L3 consulted only on release. Release set:
  **reached** (within 1 cell) · **unreachable** (own L2 field at goal ≈ 0 after belief update) ·
  **invalidated** (the belief no longer justifies it — e.g., a merge shows the target frontier
  covered; computed by the *proposer's* justification predicate, mission-general) · **timeout**
  (age > K_max, K_max ≈ 10–25 per ANS/FuN-range evidence). Event-triggered per agent, never
  synchronized (ACE). Learned termination = deferred dial, not v1.
- Training consequence: PPO's policy-gradient term is masked to decision steps (fewer, denser
  decisions — the credit-density gain the commitment literature promises).

## 6 · Coordination — the X axis

- **X0 — none (control).**
- **X1 — decode-time claim-round:** at decision points, agents in a component choose sequentially in
  ID order over the shared candidate view; later agents see earlier claims (claimed-candidate mask or
  feature). Deterministic symmetry-breaker (an order is exactly what the flood proof demands); ½
  approximation heritage (sequential submodular assignment); Grimsman's bound-vs-independent-groups
  gives the adversary-analysis handle. Bounded rounds, JAX-shape-safe.
- **X2 — intention visibility:** claims-in-mail rendered into a "claimed cells" belief plane (Spatial
  Intention Maps precedent) — the learned counterpart; count-invariant; later the covert red's
  natural lying surface (a claim is a per-edge KL-boundable message).
- X1 and X2 share the same claim state; they differ in *who* enforces complementarity (procedure vs
  learning).

## 7 · Credit

Unchanged adoption: exact submodular difference reward for coverage (per-agent advantage path exists);
**delivered-flow marginal credit** (label-free, pays by function/position) arrives with
TetherRelay/delivered-coverage — the emergent-roles-compatible form. Diagnostic: the per-agent credit
distribution is the **role-emergence order parameter** (bimodality ⇒ division) and the load-bearing
map for later red placement (H1). HAPPO sequential-update axis = MAPPO session (task #7), orthogonal.

## 8 · Instrumentation (build first)

Per-agent recording as first-class run artifacts (the 2026-08-10 request): per step per agent
{position, goal, commit age, claim conflicts, dᵢ credit, delivered-flow share, soft degree, component
id}; per run: credit-distribution figures (the order parameter), per-agent traces, claim/conflict
stats, belief-vs-ledger divergence — wired into the report panels + the isometric belief-overlay view.
Prerequisite for P1/P2 and the ablation; costs no training.

## 9 · The experiment program

1. **Probes (CPU, frozen v1 checkpoints, before any build):** P1 deploy-then-release (allocation
   share of the L3 gap) · P2 decode-time claim-round over frozen goal logits (does coupling close
   it). Parameters presented for sign-off before running.
2. **The G×C×X ablation** (GPU-gated): {G0,G1,G2} × {C0,C1} × {X0,X1,X2}, backbone/L2/L1/PPO fixed;
   metrics = %-of-BFS-oracle coverage · conn_real · redundancy floor · zero-shot unseen maps ·
   multi-seed IQM/CIs (rliable). Primary bets per evidence: C1 and G1; X arms answer division-of-labor
   and create the externalized mechanisms the resilience study needs. The stencil-vs-dense-vs-candidate
   head-to-head on unseen maps does not exist in the literature — the ablation is itself citable.
3. **Mission-generality demo:** port the winning config to TetherRelay then Cornering by swapping
   proposer + terms only; success = no architecture change required.

## 10 · Paper-integrity constraints

Leakage suite T1–T5 green on every run; agent inputs = harness-assembled observation only; declared
assumptions (pose, frame, arena, sight) cited from the zymera2 assumptions page; run manifests with
map hashes + package versions; one-command figure regeneration. The v1 leak paths (λ₂-oracle mask,
god-view fallback, potential-adjacency) are structurally absent in the v2 path.

## 11 · Open items

K_max default and stride-vs-grid resolution for G1 (pick at implementation with a small sweep) ·
claim-round round-count under fragmentation (per-component ordering detail) · learned termination
(deferred dial) · HAPPO axis (MAPPO session) · SLM head swap (future phase; the contract here is the
enabler, not the work).
