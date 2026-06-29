# Locality Design — scale-invariance through divide-and-conquer

**Status:** Proposed 2026-06-28 (post-ES-collapse). Supersedes the *learned-selector* path of
`COGNITION_DESIGN.md`. Empirical basis: `CAMPAIGN_REVIEW.md` (synthesis) + `JOURNEY.md`. Connectivity
mechanism pending the running shootout (`run_conn_shootout.py`). This is the **design + falsification test**,
not code.

---

## 1. Why this design (what the data forced)

Six campaign findings, one conclusion (full numbers in `CAMPAIGN_REVIEW.md`):

1. **Connectivity is free** — 90–100% potential connectivity everywhere (clustered start + soft penalty).
   **Coverage-at-scale is the wall**: hardcoded role 77% @16² → ~57% @32².
2. **Imposed structure beats learned structure, always.** The learned mode-selector *collapses* with scale —
   ES coverage **49 → 25 → 8%** at 16/24/32; removing the central critic (DTE) → 2%.
3. **Diversity helps only when task-grounded** (B-fork/B-dico beat base; the forced *congestion price* hurt).
4. **Reward shaping is fragile** — info-gain *reward-hacks* to ~1%, the connectivity barrier *hurts* coverage;
   only the direct coverage up-weight robustly helped.
5. **Scale-transfer carries weights, not the skill** — crowded training helps +13 @24² but *washes* @32².
6. **Adding agents floods, it doesn't divide** — redundancy climbs 3.7→7.9; no method partitions space.

→ The coverage wall is a **spatial-partitioning / credit-assignment** problem, not a connectivity one, and
**anything global, or learned-to-select, fails at scale.**

---

## 2. The principle: scale-invariance through locality

**A big world is just many small worlds.** An agent should only ever solve **its own local cell**, never
"the N×N grid." If every skill is *local and bounded*, it is **scale-invariant by construction** — a larger
world means *more agents*, each doing the identical small job. This is exactly why the global coverage skill
didn't transfer (finding 5) and why the learned selector collapsed (finding 2): both reason globally. The fix
is to make the **unit of work the local cell**, and let the world recurse into cells.

---

## 3. The design — a parallel local toolbox

Every agent carries the **same** tools, always on, chosen by **cheap hardcoded context** — never a learned
selector (that is the exact thing that collapsed at scale).

**① Local "where-to-go" skill (the core).** Fuse three classical multi-robot primitives into one heading:
- **Voronoi cell** — the region this agent is closest to among its neighbours (CVT / Lloyd). Defines *whose
  job* a patch is → no two agents own the same ground.
- **In-cell frontier** — the covered/uncovered boundary *inside the agent's own cell* (frontier exploration).
  Defines *where the edge* is.
- **Compass** — the heading toward the chosen in-cell frontier (reuses the existing compass capability).

  Output, each step: *head toward the best unexplored edge of **your** region.* Partition + frontier + heading,
  all local. Built like the compass mechanism (a substrate capability, not a menu item).

**② Difference rewards (own your cell).** Credit each agent only for coverage **no teammate also provided**
(`D_i = G − G_{-i}`). Selfishly, each agent then goes where the others aren't → **division of labour emerges,
not imposed.** The principled cure for finding (6); replaces the failed congestion price.

**③ Event-staleness exploration bonus.** Pay a bonus **at the moment of discovery**, scaled by the agent's
dry-spell: `bonus = base · g(steps_since_last_find)`, multi-fold for long droughts. Rewards the *act* of
finding new ground after a drought (pushes agents off exhausted patches). Because it pays on the **event**,
not the **state**, it can't be farmed by loitering (the info-gain failure mode); the 100-step budget further
punishes stalling. Start `g` linear-to-mild to avoid sandbagging (sitting on easy cells to grow the multiplier).

**④ Per-agent explore temperature (heterogeneity = "creativity").** One scalar per agent setting its
explore/exploit balance: *hot* agents wander far for distant pockets (risk / creativity), *cold* agents
thoroughly finish nearby frontiers. Assigned or evolved (QD). Task-grounded diversity, unlike the congestion
price. Pairs with ③ — hot agents tolerate longer droughts.

**⑤ Parallel toolbox + context selection.** All tools available **at once** (not a cascade); selection is a
**cheap rule on local state** — e.g. *"frontier in my cell → seek it; cell empty / I'm a cut-vertex → hold as
relay."* No learned selector head. Hardcoded-context over a parallel toolbox is the only thing the data says
survives at scale.

**⑥ Recursion (the scale story).** The agent's job is "cover my cell." As the world grows, more agents → more
cells → the **same** per-agent job. The big world *is* a tiling of small worlds; the skill is identical at
every scale. The operational form of §2 and the direct answer to the transfer wall (finding 5).

---

## 4. The connectivity layer — DECIDED (shootout 8/8, 2026-06-28)

Connectivity stays a **constraint, not a reward term**: a **learned-Lagrangian** dual (RCPO) that auto-tunes
its multiplier to the *smallest* value holding λ₂ ≥ τ — scale-robust, unlike the brittle fixed barrier (which
hurt coverage and didn't transfer).

**`run_conn_shootout.py` verdict (32²/10, role/base × 4 mechanisms):**

| mechanism (with roles) | coverage | real-connectivity (λ₂>0.5) |
|---|---|---|
| soft λ₂ penalty | 44.7% | 81.2% |
| **learned-Lagrangian (RCPO)** | 42.4% | **84.7%** |
| PID-Lagrangian | 39.8% | 83.0% |
| hard action-mask | 39.9% | 63.1% |

**Decision: role split + learned-Lagrangian dual.** It holds the highest real-connectivity with competitive
coverage and *adapts* across scale (soft edges coverage but is a fixed-weight penalty). The dual **needs the
role structure** — `base_lag` collapsed to 62% connectivity without it. Hard-mask underperforms (keeps a
*connected* but weak/stretched chain, low λ₂).

**The headline the shootout settles:** *connectivity is essentially solved (~85%); **coverage (~45%) is the
entire remaining gap to 90/90.*** That gap is exactly what the local skill (§3) attacks — the connectivity
mechanism is no longer the limiter.

---

## 5. Falsification test (how we'd know it works — or doesn't)

- **The redundancy curve must invert.** Today adding agents *raises* redundancy (3.7→7.9 = flooding). Success
  = redundancy → 1 as agents divide. **This is the primary metric — not raw coverage.**
- **Scale-invariance.** Train at 16², deploy *frozen* at 24²/32²; coverage % must **hold**, not collapse like
  the global skills. If it drops with scale, locality failed.
- **90/90 at 32²/10** on the strict bar (λ₂>0.5) within the 100-step budget — the headline target.
- **Honest kill criteria:** if a Voronoi + frontier + difference-reward agent *still* floods (redundancy stays
  high) or *still* drops with scale, the locality hypothesis is **wrong** and we revisit — don't rationalise.

---

## 6. What this supersedes / keeps

- **Supersedes:** the learned mode-selector (Level-2 of `COGNITION_DESIGN.md`; collapses to 8% @32²); the
  congestion price (→ difference rewards); the fixed connectivity barrier (→ learned-Lagrangian).
- **Keeps:** hardcoded explorer/relay *as a special case of the ⑤ context rule*; the central critic
  (load-bearing — the DTE collapse); the LPAC backbone (weight-transfer, scale-invariant params).
- **Parked (re-think later):** ES — proven not competitive as a policy optimiser; its only live future is as a
  **QD engine** for the ④ explore-temperature / a behaviour repertoire.

---

## 7. References

- **Coverage / partition:** Cortés, Martínez, Karataş & Bullo, *Coverage Control for Mobile Sensing Networks*
  (CVT / Lloyd), IEEE T-RA 2004 · LPAC (Gama et al., 2024).
- **Frontier exploration:** Yamauchi, *A Frontier-Based Approach for Autonomous Exploration*, 1997 · Burgard et
  al., *Coordinated Multi-Robot Exploration*, IEEE T-RO 2005.
- **Difference rewards:** Wolpert & Tumer, *Optimal Payoff Functions for Members of Collectives*, 2001 ·
  Devlin, Yliniemi, Kudenko & Tumer, *Potential-Based Difference Rewards for MARL*, AAMAS 2014.
- **Learned-Lagrangian:** Tessler et al., *RCPO*, ICLR 2019 · Stooke et al., *PID-Lagrangian*, ICML 2020 ·
  Achiam et al., *CPO*, ICML 2017.
- **Quality-Diversity:** Mouret & Clune, *Illuminating search spaces (MAP-Elites)*, 2015.
