# Zymera — Open TODO

Snapshot of the open work as of **2026-08-09**. Research-backlog detail lives in `IDEAS.md`; this is the
tracked, checkable list. (18 tasks completed this arc — the workspace reorg, the planner validation, the GPU
backup, the doc/site consolidation — are done and pushed.)

## ⏸ Deferred — paused by choice
- [ ] **Refresh the whitepaper** — `../literature/`… actually `docs/planning/experiment-plan.html`'s
      "Findings so far" is a Jul-21 snapshot (T1 only); update it with the T2–T5 results. The HTML/report
      reorg is done — this is all that's left of it. *(#29)*
- [ ] **Process the staged `literature/` folder** — distill the 4 lit docs (`STRATEGY.md` +
      `asymmetry_litreview` / `positioning_portfolio` / `slm_marl_survey_brief`) into memory + the site's
      related-work, then archive. Plan in `docs/literature/README.md`. *(#30)*

## 🚫 Blocked — needs new GPU access (balthar + gaius were wiped)
- [ ] **Connectivity signal** — actor critic-only vs a distilled λ₂/Fiedler estimator: does the actor need an
      explicit λ̂₂, or does the CTDE critic + loss suffice? *(#5)*
- [ ] **Critic ablation** — setpool vs setattn (was mid-run at the wipe; T4 said set-based > conv, attn ≈ pool). *(#9)*
- [ ] **Train full pipeline** — all connectivity systems + central critic + mvprop on open 32²; grade
      connectivity-first. *(#19)*

## 🔬 Research backlog — also captured in `IDEAS.md`
- [ ] **Adversarially verify the KB audit** (workflow) — stress-test the linear-probe claims. *(#4)*
- [ ] **Brainstorm MAPPO details** (dedicated session). *(#7)*
- [ ] **Cornering / hazard-cordon mission** — the obligate-collaboration flagship (~10 pursuers vs 6 evaders,
      multi-agent); reward shape makes every agent pivotal. *(#8)*
- [ ] **Occlusion-aware comms** — walls attenuate comm range (soft `d_eff = d + c·k`); today occlusion is OFF
      (through-wall comms), which overstates connectivity. See `wall-rf-occlusion-comms` memory. *(#15)*
- [ ] **Fusion-poisoning mission** — Boeing standards-based-teaming; bias the local→shared belief fusion
      (the C-seam attack). *(#21)*

---

### The bigger picture (the two prongs — see `IDEAS.md`)
- **Prong 1 — the adversarial / resilience thesis** (barely started): covert misbehavior → mission failure,
  minimum m-of-n, propagation, local anomaly detection, the stealth–damage frontier, resilience metrics.
- **Prong 2 — team design** (the substrate, ~mature): the L3 goal head is the open lever; distillation-recipe
  upgrade; connectivity-weight Pareto sweep; reproducibility/transfer.

*Pairs with `IDEAS.md` (the full idea registry), `T5_STATUS.md` (current build state), and
`../journal/JOURNEY.md` (the narrative).*
