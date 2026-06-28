# SuperBlue — Open Threads & Idea Map

**A living index of everything in flight, so nothing gets lost.** This is the *board*, not the
detail — deep dives live in `STRATEGY.md` (verdicts + citations), `EXPERIMENT_PLAN.md` (designs),
`CAMPAIGN_REVIEW.md` (what we've actually run + the honest numbers), `JOURNEY.md` (the daily log),
`IDEAS.md` (raw idea dump). When a thread here gets resolved, move it to VALIDATED and point at the doc.

_Last gathered: 2026-06-27._

---

## ★ THE DESTINATION — why any of this matters
**Covert-adversary resilience.** How does *stealthy, internal* micro-level misbehavior by one (or a
few) agents propagate through the team and surface as macro **mission failure**? The classical methods
have no defense; that's the whole point. Everything below is substrate for this. We win on the
**resilience axis**, not the raw-coverage race. (Governing RQ — see root `CLAUDE.md` + the formalism.)

**The headline target:** ≥90% coverage **AND** ≥90% connectivity *simultaneously*, at **32×32 / 10
agents**, within a **100-step** budget — graded on the **real** connectivity bar (λ₂ > 0.5), not the
trivial 1e-3 floor. Still open.

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

---

## ✓ VALIDATED — verdicts in, don't re-litigate (see STRATEGY.md)
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
