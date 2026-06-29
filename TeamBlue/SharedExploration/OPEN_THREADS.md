# SuperBlue — Open Threads & Idea Map

**A living index of everything in flight, so nothing gets lost.** This is the *board*, not the
detail — deep dives live in `STRATEGY.md` (verdicts + citations), `EXPERIMENT_PLAN.md` (designs),
`CAMPAIGN_REVIEW.md` (what we've actually run + the honest numbers), `JOURNEY.md` (the daily log),
`IDEAS.md` (raw idea dump). When a thread here gets resolved, move it to VALIDATED and point at the doc.

_Last gathered: 2026-06-28._

---

## ★ THE DESTINATION — why any of this matters
**Covert-adversary resilience.** How does *stealthy, internal* micro-level misbehavior by one (or a
few) agents propagate through the team and surface as macro **mission failure**? The classical methods
have no defense; that's the whole point. Everything below is substrate for this. We win on the
**resilience axis**, not the raw-coverage race. (Governing RQ — see root `CLAUDE.md` + the formalism.)

**The headline target:** ≥90% coverage **AND** ≥90% connectivity *simultaneously*, at **32×32 / 10
agents**, within a **100-step** budget — graded on the **real** connectivity bar (λ₂ > 0.5), not the
trivial 1e-3 floor. **Connectivity side now CLOSED** — the mechanism shootout reached ~85% real-connectivity
with **role + learned-Lagrangian (RCPO)**; **coverage (~45% @32²) is the entire remaining gap.** Direction →
`LOCALITY_DESIGN.md`: scale-invariance through locality (Voronoi cell + in-cell frontier + difference rewards).
The learned mode-selector is **dead at scale** (ES collapses 49→25→8% up the ladder). The contribution reframes
to the **formalisation** — MARL as a model of *autonomous teams across a spectrum* (multi-robot→swarm→LLM agents),
the learning's real job being **generalisable orchestration ("when to use what")**; resilience is one stratum.

---

## ⚙ NOW — the selector core, and the A/B batch results (2026-06-28)
The fixed spec (locked, never drifts): **comm_r = 5 everywhere · density-pinned ladder
16²/4→24²/6→32²/10 · hard collision-mask · soft/learned connectivity · 100 steps · 3 seeds**.

**A/B batch — RAN (balthar, partial 23/39; full writeup `CAMPAIGN_REVIEW.md` §6):**
- **Every arm beats base** on coverage; the **diversity arms (B-fork, B-dico) lead, most at scale**
  (32²/10: B-dico **61%** s1 · B-fork 50 · armA 47 · base 41). **C0 confirmed — specialization emerges.**
- **The DTE tail COLLAPSED at 32² (8%, 100% conn = huddle)** → **the CTDE central critic is load-bearing
  at scale** (→ keep CTDE for the executor; only the selector goes to ES).
- **Connectivity 90–100% (strict λ₂>0.5) everywhere** → coverage-at-scale is the wall, not connectivity.
- Caveat: 32² is single-seed-per-arm → cross-arm 32² ranking preliminary; the 16² (3-seed) numbers are solid.

**The selector core — BUILT (89 tests green), being integrated:**
- the learned **selector over {disperse, flock, hold}** (hierarchical skill+offset policy) + the **flock**
  skill (scripted + learned) + the **free-market congestion** price + per-agent coverage metric. Gated;
  selector-off byte-unchanged. The **ES coexistence** trainer (`es.py`, OpenAI-ES/CEM + MERL) is built too.
- → **next:** wire ES↔selector (`actor.selector_head`), run the **2×2 flock×congestion** sweep + the
  **fixed-world N-sweep**, try task-grounded individuation (graph-position role, diversity-as-a-loss).

→ tracked as tasks #49–#62. Design: `COGNITION_DESIGN.md`.

**Obstacle + crowded reruns — RAN (balthar; 2026-06-28 cont., `JOURNEY.md`):**
- **Obstacle batch FINISHED (50/54).** **Connectivity barrier REFUTED even in corridors** (r24 rooms:
  barrier-ON sacrifices half the coverage for ~+10 pts conn that barrier-OFF's 88–98% didn't need).
  **`einfo` reward-hacks (~1%), `ebump` wins, role ≥ base ≥ sel** — confirmed at 32² too.
- **Connectivity-safe crowded terrains BUILT** (`ctde_v0/terrains.py`: ConnectedClutter · Pillars ·
  MixedCluttRooms · RandomCrowded; BFS flood-fill *constructs* a single connected free component, JAX-traceable;
  10 tests, suite 105 green). `--terrain clutter/pillars/mixed/crowded_mix`. Committed `0a80fed` (NOT pushed).
- **Crowded curriculum batch RAN (10/12).** **Training-on-crowded helps at 24² (+11–13 hard maps) but WASHES
  at 32² (±3)** → weak scale-transfer of the harder regime (the lit-review open Q, answered "weakly").
  **role > base on crowded** (32²: pillars +5.7, heavy clutter +3.8).
- **Interactive gallery** (`report/index.html`, `ctde_v0/make_report.py`) — **41 runs, 7 categories**,
  manifest-driven (categories + descriptions + render-time world-override); crowded zero-shot vs TRAINED
  before/after. → **keep refreshed as new checkpoints land.**

---

## ✓ VALIDATED — verdicts in, don't re-litigate (see STRATEGY.md)
- **Connectivity = a CONSTRAINT, not a reward penalty (barrier → learned-Lagrangian).** Empirically the
  exp-barrier penalty HURTS coverage in open AND corridors; the lit review (Tessler RCPO ICLR'19) explains why
  — a *fixed* penalty coefficient is brittle and can't hold across the 16²→32² scale ladder. **Fix = RCPO/CPO
  with an auto-tuned multiplier** (Li 2022 = CPO on a λ₂<0 cost). Don't re-try fixed-weight barriers. (#4/#9.)
- **LPAC-style weight-shared GNN spine = the size-invariance enabler — keep it.** Scale-transfer comes from
  shared filter taps + permutation-equivariance (LPAC, VGAI N=50→75), NOT depth/capacity. Caveat: every
  verified coverage backbone is connectivity-blind → that's exactly our gap, and **the conjunction**
  (connectivity-constrained + scale-invariant + hierarchical-role + adversarial-resilience) is the defensible
  novelty (no single paper does all four). Closest threat: LPAC follow-on *"Constrained Learning for
  Decentralized Multi-Objective Coverage Control"* — read directly. (#2/#5/#6.)
- **Shared map / blackboard / pheromone** — classical baseline that, in its honest decentralized form,
  *is* our GNN belief idealized; superseded as a *policy* by learned GNN/LPAC; the literal blackboard
  smuggles in global comms. **Value to us = baseline + clean red-team attack surface** (poison the
  field = Sense attack; the gossip staleness = the covert-withholding channel). Don't adopt as the method.
- **Market auction (SLAM + A* + frontier bidding)** — textbook Zlot–Stentz; same smuggled-global-comms
  catch as the blackboard; `D_max` = the soft cohesion-magnet again (dominated by our hard constraint).
  **Value = scripted baseline + covert-attack surface** (lie in your bids). Build deliberately, not rushed.
- **The 4-tier "brain" tower (the old L4)** — anthropomorphic over-engineering; short-circuits emergence.
  **Cap at 2 learned levels.** Connectivity = a hard constraint, not a brain level. Roles/phases EMERGE.
- **ES as a trainer *swap*** — it's neuroevolution; strips the gradient pipeline ("how is this MARL
  anymore"). **Demote to a diagnostic / coexisting idea, not the backbone.** (open Q below).

---

## ⏭ NEXT — queued deliberate builds (after tonight)
- **Non-MARL baseline panel** (this is what "compare to ALL competitors" actually means — a bounded
  ~6: greedy frontier · auction · CVT/Lloyd · potential-field · pheromone · + LPAC ref + clairvoyant
  oracle ceiling). Same env + same scorecard. Be *competitive* nominal, *decisively best* adversarial.
  → task #56.
- **Scripted auction + shared-map** as both baselines AND red-team attack surfaces (members of the panel).
- **2-level cognition architecture — DESIGNED 2026-06-28 → `COGNITION_DESIGN.md`.** Capabilities (rich
  always-on substrate: GNN belief + λ₂ aux head + A\*) vs a small behavioral **menu {disperse · flock ·
  hold}** the selector toggles; the selector is a *general unnamed-mode* picker (roles emerge); uniqueness
  lives in **individuated ES-evolved selectors over shared skills**; **anti-collapse = 3 forces** (SND/role_div
  diversity pressure · individuation · the **free-market congestion price** — learned/local, not a scripted
  auction); **ES evolves selectors + gradient trains skills/perception, side by side** (MERL/feudal). The
  gather/disperse rhythm is an emergent *prediction to test*, not hardcoded. → **experiment design is next.**

---

## ❓ OPEN QUESTIONS — parked, not yet answered
- **Coexisting ES beside the gradient trainer** — can an ES policy warm-start CTDE (do they look alike)?
  Does an ES find at 16² generalize up the ladder? How would ES juggle coverage *and* connectivity?
- **Gather/disperse rhythm in TIME** (the surviving half of the ex-L4 idea) — trade coverage↔connectivity
  across *phases* (disperse → stabilize → contract) rather than every step. Learned or scripted phase control?
- **Does the warm-start curriculum survive the fixed spec?** (the old 52% was comm_r-confounded.)
- **Is the coverage plateau the agent or the 100-step budget?** (h-sweep says budget matters, but the
  disperse skill @h100 beats the baseline @h400 — dispersal efficiency dominates.)

---

## ☰ THE STRATEGY MENU — the 11 axes from the deep-research (verdicts in STRATEGY.md)
energy/battery tokens · learnable comms (GNN messages) · frontier-attention *(the proven disperse lever,
~32% honest)* · CBF/Lagrangian constraints · physics-inspired / LPAC · network fundamentals
(size-invariance) · multi-phase mission cycle · per-role tools/skills · reward engineering / intrinsic
motivation · graph embeddings / Fiedler · diffusion models. ← the palette every new experiment draws from.
