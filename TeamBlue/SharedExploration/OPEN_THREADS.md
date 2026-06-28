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

## ⚙ NOW — building tonight (overnight balthar batch)
The fixed spec (locked, never drifts): **comm_r = 5 everywhere · density-pinned ladder
10²/2→16²/4→24²/6→32²/10 · hard collision-mask on · connectivity soft/learned · 3 seeds**.

- **Yardstick metrics** — real conn bar (λ₂>0.5) + behavioural diversity (SND) + role-distinctness
  (role_div). ✅ *built (increment 1)* — these are also the common scorecard for the baseline panel.
- **Arm A** — curriculum + randomness + DTE tail: climb the ladder warm-starting each rung, inject
  σ-noise to break symmetry, then switch the centralized critic → decentralized (does dropping the
  central crutch unlock specialization?).
- **Arm B-fork** — 2 groups (explorer/relay) with *separate* params on a shared backbone+critic;
  specialization emerges + is measured (between-group role_div).
- **Arm B-dico** — one shared policy + a small per-agent residual sized to a target behavioural
  diversity (DiCo-style); diversity *without* fully splitting.

→ tracked as tasks #49–#55. Deploy is the **last** step (one clean push, GPU-smoke per arm, then tmux).

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
- **2-level cognition architecture — make it concrete + expandable.** L1 (control) is just plumbing →
  we effectively have 2 levels. **Rename the role-picker to something general and let it evolve.** This
  is your explicit next deep-dive after the batch.

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
