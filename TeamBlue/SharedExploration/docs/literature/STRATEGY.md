# SuperBlue — Consolidated Strategy Verdict (literature-grounded)

**Written 2026-06-27**, after a 7-search literature review of the directions we'd been considering
(ES, QD, evolve-then-finetune, curriculum/scale, hierarchical RL, role-based MARL, swarm-flat-vs-cognitive).
This is the "what the evidence says to actually do." It supersedes the in-flight strategy chatter.

## The one-sentence verdict
**Most of the elaborate ideas we got excited about — a 4-tier strategy→role→skill→action brain, switching
the whole optimizer to ES, using QD as the trainer — are over-reach the literature undercuts. The
better-evidenced plan is a near-flat shared MARL policy with the goal-head we already built, fixed by
*targeted exploration* and an *emergent-diversity* signal, under a *connectivity constraint*, on a
*fixed-spec density-pinned curriculum* — with ES/QD demoted to a frontier-mapping diagnostic.**

## The reframe that unifies everything
The huddle is a **hard-exploration / deceptive-optimum problem in a homogeneous, over-shared policy.** Two
independent searches converged on this:
- **Hierarchy's benefit IS exploration, not structure** (Nachum et al. 2019, arXiv:1909.10618 — a flat agent
  matched HRL once it had temporally-extended exploration + multi-step returns; the structural inductive bias
  contributed little beyond that).
- **ES doesn't escape deception either** ("ES beats PG on deception" is folklore; plain ES collapses to the
  same degenerate optima — the "always one action" collapse Salimans et al. 2017 (arXiv:1703.03864) report,
  and the deception failure that Lehman & Stanley 2011 (*Evol. Comput.* 19(2)) introduced novelty search to
  escape). The real lever is **directed novelty/diversity exploration, which works on PPO too** (NS-ES /
  NSRA-ES, Conti et al. 2018, arXiv:1712.06560 — the exploration gain comes from the novelty pressure, not
  from ES per se).

So both big moves we considered (the tower, the ES swap) are *expensive ways to buy exploration.* Buy it cheap.

## What the evidence says to DO (in priority order)

1. **Build a STRONG flat shared-policy MARL baseline FIRST.** Temporally-extended exploration + n-step
   returns. Per Nachum et al. 2019 (arXiv:1909.10618) this alone may close most of the gap, and it is the
   benchmark every added piece must beat. *Don't add structure until it wins a fair fight against this.* (The
   "flat, well-tuned PPO is a strong baseline" finding is the same one MAPPO reports — Yu et al. 2021,
   arXiv:2103.01955.)

2. **Keep the goal/region action head** (frontier-attention — we already have it; 16→32%). The ES search
   independently calls a goal-selection head **"plausibly higher-leverage than ES."** This is our proven lever,
   and it is the *one* temporal-abstraction the HRL evidence cleanly supports (Nachum et al. 2019,
   arXiv:1909.10618).

3. **Fix the homogeneity / "entropy-made-it-worse" pathology with an emergent-diversity signal — not roles, not
   a tower.** In shared-policy MARL, more exploration *breaks cooperation* (representations entangle — "How
   Exploration Breaks Cooperation in Shared-Policy MARL", arXiv:2601.05509). The better-evidenced fix is a
   **CDS-style mutual-information diversity regularizer** (agent-identity ↔ trajectory) on a near-flat shared
   net, which lets the explorer/relay split *emerge* (Li et al. 2021, CDS, arXiv:2106.02195) — vs hand-building
   a role layer. (MAPPO-flat matches RODE on 10/14 SMAC maps — Yu et al. 2021, arXiv:2103.01955; roles in
   ROMA / RODE are themselves *learned latents*, not imposed — Wang et al. 2020, arXiv:2003.08039; Wang et al.
   2021, arXiv:2010.01523.)

4. **Connectivity is a CONSTRAINT, not an objective or a brain level.** Maximize coverage **subject to
   connectivity ≥ a real bar** (constrained-RL / Lagrangian / hard mask — cf. "Constrained Learning for
   Coverage", arXiv:2409.11311, and "RL Connectivity Maintenance", arXiv:2109.08536) — beats scalarizing (our
   own prior: hard guardrail +20 pts; QD search agrees). Keep the hard mask as a safety envelope.

5. **Reframe-cheap exploration fixes worth trying alongside #1–#3:** break/group parameter-sharing (the
   documented cause of #3 — arXiv:2601.05509); **Go-Explore-style archived exploration** (remember frontier
   states → return → explore; beat novelty-ES on hard exploration, "almost tailor-made" for coverage —
   Ecoffet et al. 2021, *Nature* 590) with a **coverage-footprint behavior descriptor**; potential-based
   dispersion shaping.

6. **Cap any structure at 2 levels, learned not scripted.** A learned role/goal selector over a low-level
   executor (manager/worker — the only depth that works at scale: FeUdal (Vezhnevets et al. 2017,
   arXiv:1703.01161) / HIRO (Nachum et al. 2018, arXiv:1805.08296) / Director (Hafner et al. 2022,
   arXiv:2206.04114) / SOL ("Scalable Option Learning" 2025, arXiv:2509.00338) / FMH (Ahilan & Dayan 2019,
   arXiv:1901.08492) are all 2-level). **No 3rd level** without HAC-style on-task evidence that 3 > 2 > flat
   beyond seed noise (Levy et al. 2019, arXiv:1712.00948 — HAC *does* learn 3 levels, but only with hindsight
   and on tasks where the depth demonstrably pays). For *homogeneous* coverage the required role-diversity is
   **low**, so even one explicit role level may be inert — **measure role-diversity** (action/trajectory/
   contribution; Hu et al. 2022, arXiv:2207.05683) on the flat baseline first; if low, skip the layer.

7. **Curriculum (small→large) is sound** (LPAC — Agarwal et al. 2025, IEEE T-RO, arXiv:2401.04855; EPC —
   Long et al. 2020, arXiv:2003.10423) **but FIX three things:** (a) connectivity must **bind at every rung
   incl. the small ones** (else "small doesn't contain the hard problem"); (b) **pin density** (agents/area)
   across rungs; (c) the **100-step budget must bind at every rung** (or scale with the world) — and add
   **multi-scale fitness** (don't train-once-small-then-transfer; the small/sparse end is the weak corner — cf.
   DyMA-CL, Wang et al. 2020, arXiv:1909.02790; learning-progress curricula, arXiv:2205.10016).

## Where ES / QD still belong (demoted, not deleted)
- **QD/MAP-Elites with (coverage, connectivity) descriptors = the FRONTIER-MAPPER & deception-escape *diagnostic*** on a compact controller — well-precedented (MAP-Elites, Mouret & Clune 2015, arXiv:1504.04909; QD as a field, Pugh et al. 2016, *Front. Robotics & AI*; and directly the **multi-function-swarm coverage+connectivity descriptors** of Engebråten et al. 2020, arXiv:2007.08656). **Not** the trainer for the deep GNN (QD+GNN-at-scale is unexplored; PGA-MAP-Elites (Nilsson & Cully 2021, GECCO) and DCRL/DCG-MAP-Elites (Faldor et al. 2024, arXiv:2401.08632) scale QD to deep *single-agent* nets, and Mix-ME (arXiv:2311.01829) extends it to MARL, but none to a size-invariant GNN coverage policy). Produces the cov↔conn frontier as a deliverable (MOME, Pierrot et al. 2022, arXiv:2202.03057; CVT-MAP-Elites for high-dim descriptor spaces, Vassiliades et al. 2018, arXiv:1610.05729 — on-theme for the formalism's Pareto/brittleness frontiers).
- **MERL-style hybrid** (evolve the sparse *team* objective + gradient the dense *per-agent* reward, **interleaved** with periodic policy injection — NOT a one-shot weight handoff) is the proven multi-agent way to combine them, *if* the flat-baseline + diversity-signal path stalls (MERL, Majumdar et al. 2020, arXiv:1906.07315; the single-agent lineage: ERL, Khadka & Tumer 2018, arXiv:1805.07917; CEM-RL, Pourchot & Sigaud 2019, arXiv:1810.01222; PDERL, Bodnar et al. 2020, arXiv:1906.09807; survey: Sigaud 2022, arXiv:2203.14009). Gotchas: small PPO log-std init, critic warm-up, don't recombine net weights raw (PDERL, arXiv:1906.09807 — naive crossover of net weights is destructive; the genetic operators must be made representation-aware).

## The honest meta-point
We kept reaching for big architecture (a cognitive tower) and big method swaps (ES). The literature's repeated
message — across hierarchy, roles, ES, and QD — is the same: **attack the specific failure (exploration,
over-sharing, the action head) with specific, minimal, *learned* tools; earn every added level/component against
a strong flat baseline.** The elegant-tower instinct is exactly the trap the skeptical literature names.

## Immediate next experiments (the executable plan)
1. **Honest baseline:** flat shared MARL + goal-head + temporally-extended exploration, at **fixed comm_r** with a
   **real binding connectivity constraint**, density-pinned. Get the true coverage/connectivity numbers.
2. **Measure role-diversity** on it. If low → no role layer.
3. **Add the CDS-style diversity regularizer**; see if explorer/relay emerges and coverage climbs.
4. Only then weigh: a single learned role/selector level · Go-Explore archived exploration · the MERL hybrid · QD
   for the frontier picture.

## The capstone (7th search: swarm-flat-rules / bitter-lesson / anthropomorphism)
Three independent traditions converge on: **a hand-designed strategy→role→skill→action brain is anthropomorphic
over-engineering.**
- **Swarm intelligence's founding result — you don't need cognitive layers for sophisticated collective
  behavior.** Boids = flocking from 3 flat local rules (Reynolds 1987, SIGGRAPH). **Couzin gets
  swarm→torus→parallel-group *phase transitions* — exactly the "mode/phase" an L4 strategy layer is meant to
  provide — from tuning ONE parameter over identical flat rules** (Couzin et al. 2002, *J. Theor. Biol.*). So
  even the gather/disperse *phases* are the kind of thing that EMERGES; they shouldn't be a hand-built
  mode-head.
- **Bitter lesson:** "building in how we think we think does not work in the long run" (Sutton 2019,
  incompleteideas.net). A strategy→role→skill→action stack is almost a definitional instance of it; the
  contemporary MARL echo is **"Flattening Hierarchies" (SAW, arXiv:2505.14975)**, which shows a flattened
  policy can recover hierarchical performance.
- **Swarm-robotics measured the cost of imposed structure: overfitting / reality-gap brittleness** (Birattari
  et al. 2021, PMC8285396 — automatic *off-line* swarm design overfits the simulation; restricted design space
  beat expressive neural; "more expressive ≠ more capable"). And the *learned-flat* alternative is competitive
  on the very task we care about: a decentralized coverage GNN beats Lloyd's (arXiv:2109.15278).

**THE argument specific to *your* program:** a hand-designed hierarchy **short-circuits the very phenomenon you
exist to study.** If you legislate the roles and phases, you can no longer ask *whether* they emerge, *how* they
propagate micro→macro, or *how an adversary perturbs them.* Flat-policy + emergence keeps the macro structure a
**measured outcome** — more honest for a resiliency study, and it gives the red team a *real emergent target* to
attack instead of a scaffold you installed. This is the strongest reason of all to not build the tower.

**The bias-about-the-WORLD vs bias-about-COGNITION distinction (the rule for what to keep):** impose structure
only where it encodes a true *task/world* invariant (like a CNN's translation-equivariance), never a guess about
how a mind should think. That permits exactly three impositions:
1. **Connectivity = a hard constraint / action-mask safety shell** (world-invariant; must hold w.p. 1; our own +20-pt evidence; cf. "RL Connectivity Maintenance", arXiv:2109.08536). KEEP.
2. **A coarse spatial goal/region action head** (task-geometry bias; the *one* temporal-abstraction level the HRL evidence supports — Nachum et al. 2019, arXiv:1909.10618; fixes our 1-step-move ceiling). KEEP — this is the single concession to "hierarchy," and it's 2 levels.
3. **Roles as a LEARNED, regularized-to-be-readable latent** (ROMA-style MI — Wang et al. 2020, arXiv:2003.08039), measured as an outcome — NOT a hand-authored explorer/relay layer. LEARN.
**DROP entirely: the explicit "strategy" top level and the discrete "skill" library** — the most anthropomorphic,
least-evidenced layers; phases and skills are what *emerge* (Couzin et al. 2002, *J. Theor. Biol.*; Reynolds
1987, SIGGRAPH). The contrast case to remember: option/skill libraries are themselves *discovered*, not
authored, in the HRL literature (Option-Critic, Bacon et al. 2017, arXiv:1609.05140; h-DQN, Kulkarni et al.
2016, arXiv:1604.06057) — and even there the structural payoff over a flat baseline is contested (Nachum et al.
2019, arXiv:1909.10618).

## ⮕ The decisive falsification test (run this BEFORE adding any structure)
Train the **flat-ish system** — goal head + GNN backbone + a learned role latent, inside the hard connectivity
shell, under a **delivered-coverage** objective (so a reason to gather *exists*) — at the **fixed honest spec**.
Then check whether **labor-division and the disperse↔gather rhythm EMERGE** (redundancy curve bending toward 1,
MI(role;behavior) rising — measured per Hu et al. 2022 (arXiv:2207.05683); a visible breathe-out/breathe-in).
- **If they emerge → the 4-level brain is *refuted as unnecessary*** (and you have a clean emergence result for
  the resiliency program — the Couzin-style phase transition reproduced in a learned policy, Couzin et al. 2002).
- **If they DON'T emerge even with the right objective → *that* is your real, earned evidence that a thicker
  scaffold is warranted** — and you'll know exactly *which* layer to add and *why* (the HAC discipline:
  earn each level on-task — Levy et al. 2019, arXiv:1712.00948), instead of importing the whole human
  hierarchy on faith.

This is the cheapest possible way to settle the entire "tower vs. emergence" question with data.

---

## References

Grouped by topic. arXiv ids verified; where author names were unavailable the entry is cited by short title +
year + arXiv id. All claims in this document draw only on the sources below.

### ES / exploration / deception
- Salimans, Ho, Chen, Sidor & Sutskever 2017. *Evolution Strategies as a Scalable Alternative to Reinforcement Learning.* arXiv:1703.03864.
- Such, Madhavan, Conti, Lehman, Stanley & Clune 2017. *Deep Neuroevolution: Genetic Algorithms Are a Competitive Alternative for Training Deep Neural Networks for RL.* arXiv:1712.06567.
- Lehman & Stanley 2011. *Abandoning Objectives: Evolution Through the Search for Novelty Alone.* Evolutionary Computation 19(2).
- Conti, Madhavan, Petroski Such, Lehman, Stanley & Clune 2018. *Improving Exploration in Evolution Strategies for Deep RL via a Population of Novelty-Seeking Agents (NS-ES / NSR-ES / NSRA-ES).* NeurIPS. arXiv:1712.06560.
- Pagliuca & Nolfi 2022. *The Importance of Selecting the Right Algorithm: Qualitative Differences Between Evolutionary Strategies and Reinforcement Learning.* arXiv:2205.07592.
- Ecoffet, Huizinga, Lehman, Stanley & Clune 2021. *First Return, Then Explore (Go-Explore).* Nature 590.
- *How Exploration Breaks Cooperation in Shared-Policy Multi-Agent RL.* 2026. arXiv:2601.05509.

### Evolve + gradient hybrids
- Khadka & Tumer 2018. *Evolution-Guided Policy Gradient in Reinforcement Learning (ERL).* NeurIPS. arXiv:1805.07917.
- Pourchot & Sigaud 2019. *CEM-RL: Combining Evolutionary and Gradient-Based Methods for Policy Search.* ICLR. arXiv:1810.01222.
- Bodnar, Day & Lió 2020. *Proximal Distilled Evolutionary Reinforcement Learning (PDERL).* AAAI. arXiv:1906.09807.
- *Collaborative Evolutionary Reinforcement Learning (CERL).* 2019. ICML. arXiv:1905.00976.
- *ERL-Re²: Efficient Evolutionary Reinforcement Learning with Shared State Representation and Individual Policy Representation.* 2023. ICML. arXiv:2210.17375.
- Majumdar, Khadka, Miret, McAleer & Tumer 2020. *Evolutionary Reinforcement Learning for Sample-Efficient Multiagent Coordination (MERL).* ICML. arXiv:1906.07315.
- *RACE: Improve Multi-Agent Reinforcement Learning with Representation Asymmetry and Collaborative Evolution.* 2023. ICML.
- Sigaud 2022. *Combining Evolution and Deep Reinforcement Learning for Policy Search: a Survey.* ACM TELO. arXiv:2203.14009.
- *Guided Evolutionary Strategies / Global-Local Policy Search (ES → PPO warm-start).* 2020. arXiv:2010.06718.
- *Warm-Start Actor-Critic.* 2023. ICML. arXiv:2306.11271.

### Quality-Diversity
- Mouret & Clune 2015. *Illuminating Search Spaces by Mapping Elites (MAP-Elites).* arXiv:1504.04909.
- Pugh, Soros & Stanley 2016. *Quality Diversity: A New Frontier for Evolutionary Computation.* Frontiers in Robotics and AI.
- Engebråten, Moen, Yakimenko & Glette 2020. *Multi-Function Swarms (coverage + connectivity behavior descriptors).* arXiv:2007.08656.
- Pierrot, Macé, Chalumeau, et al. 2022. *Multi-Objective Quality-Diversity Optimization (MOME).* arXiv:2202.03057.
- Nilsson & Cully 2021. *Policy Gradient Assisted MAP-Elites (PGA-MAP-Elites).* GECCO.
- Faldor, Chalumeau, Flageat & Cully 2024. *Synergizing Quality-Diversity with Descriptor-Conditioned Reinforcement Learning (DCRL / DCG-MAP-Elites).* arXiv:2401.08632.
- Vassiliades, Chatzilygeroudis & Mouret 2018. *Using Centroidal Voronoi Tessellations to Scale Up the Multidimensional Archive of Phenotypic Elites Algorithm (CVT-MAP-Elites).* arXiv:1610.05729.
- Cully, Clune, Tarapore & Mouret 2015. *Robots That Can Adapt Like Animals.* Nature.
- *Mix-ME: Quality-Diversity for Multi-Agent Learning.* 2023. arXiv:2311.01829.

### Curriculum / scale-transfer
- Agarwal, Tolstaya, Gosrich, et al. 2025. *Learnable Perception–Action–Communication (LPAC) for Multi-Robot Coverage.* IEEE Transactions on Robotics. arXiv:2401.04855.
- Tolstaya, Gama, Paulos, et al. 2021. *Multi-Robot Coverage and Exploration using Spatial Graph Neural Networks.* IROS. arXiv:2011.01119.
- Long, Zhou, Gupta, et al. 2020. *Evolutionary Population Curriculum for Scaling Multi-Agent RL (EPC).* ICLR. arXiv:2003.10423.
- Wang, Han, Wang, et al. 2020. *From Few to More: Large-Scale Dynamic Multiagent Curriculum Learning (DyMA-CL).* arXiv:1909.02790.
- *Scaling Swarm Coordination with Graph Neural Networks — How Far Can We Go?* 2025. MDPI AI 6(11):282.
- *Learning Progress Driven Multi-Agent Curriculum.* 2022. arXiv:2205.10016.

### Hierarchical RL
- Nachum, Tang, Lu, Gu, Lee & Levine 2019. *Why Does Hierarchy (Sometimes) Work So Well in Reinforcement Learning?* arXiv:1909.10618.
- Levy, Konidaris, Platt & Saenko 2019. *Learning Multi-Level Hierarchies with Hindsight (HAC).* ICLR. arXiv:1712.00948.
- Vezhnevets, Osindero, Schaul, et al. 2017. *FeUdal Networks for Hierarchical Reinforcement Learning.* ICML. arXiv:1703.01161.
- Nachum, Gu, Lee & Levine 2018. *Data-Efficient Hierarchical Reinforcement Learning (HIRO).* NeurIPS. arXiv:1805.08296.
- Hafner, Lee, Fischer & Abbeel 2022. *Deep Hierarchical Planning from Pixels (Director).* NeurIPS. arXiv:2206.04114.
- Bacon, Harb & Precup 2017. *The Option-Critic Architecture.* AAAI. arXiv:1609.05140.
- Zhang & Whiteson 2019. *DAC: The Double Actor-Critic Architecture for Learning Options.* NeurIPS.
- *Scalable Option Learning in High-Throughput Environments (SOL).* 2025. arXiv:2509.00338.
- *On Credit Assignment in Hierarchical Reinforcement Learning.* 2022. arXiv:2203.03292.
- *Hierarchical Reinforcement Learning: A Survey.* MDPI 4(1):9.

### Role-based MARL
- Yu, Velu, Vinitsky, et al. 2021. *The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games (MAPPO).* arXiv:2103.01955.
- Hu, Zhao, Hu, et al. 2021. *RIIT: Rethinking the Importance of Implementation Tricks in Multi-Agent RL.* arXiv:2102.03479.
- Ellis, Cook, Moalla, et al. 2023. *SMACv2: An Improved Benchmark for Cooperative Multi-Agent RL.* NeurIPS. arXiv:2212.07489.
- Li, Wang, Yang, et al. 2021. *Celebrating Diversity in Shared Multi-Agent RL (CDS).* NeurIPS. arXiv:2106.02195.
- Wang, Ren, Liu, et al. 2020. *ROMA: Multi-Agent Reinforcement Learning with Emergent Roles.* ICML. arXiv:2003.08039.
- Wang, Gupta, Mahajan, et al. 2021. *RODE: Learning Roles to Decompose Multi-Agent Tasks.* ICLR. arXiv:2010.01523.
- Hu, Xu, Zhang & Zhang 2022. *Policy Diagnosis via Measuring Role Diversity in Cooperative Multi-Agent RL.* ICML. arXiv:2207.05683.
- Ahilan & Dayan 2019. *Feudal Multi-Agent Hierarchies for Cooperative Reinforcement Learning (FMH).* arXiv:1901.08492.

### Swarm / bitter-lesson / emergence
- Reynolds 1987. *Flocks, Herds and Schools: A Distributed Behavioral Model (boids).* SIGGRAPH.
- Couzin, Krause, James, Ruxton & Franks 2002. *Collective Memory and Spatial Sorting in Animal Groups.* Journal of Theoretical Biology.
- Sutton 2019. *The Bitter Lesson.* incompleteideas.net.
- Birattari, Ligot, Hasselmann, et al. 2021. *Automatic Off-Line Design of Robot Swarms: A Manifesto (overfitting / reality-gap).* PMC8285396.
- *Decentralized Control of Multi-Robot Coverage with Graph Neural Networks (beats Lloyd's).* 2021. arXiv:2109.15278.
- *Constrained Learning for Decentralized Multi-Objective Coverage Control.* 2024. arXiv:2409.11311.
- *Flattening Hierarchies with Policy Bootstrapping (SAW).* 2025. arXiv:2505.14975.
- Kulkarni, Narasimhan, Saeedi & Tenenbaum 2016. *Hierarchical Deep Reinforcement Learning (h-DQN).* arXiv:1604.06057.
- *Reinforcement Learning for Connectivity Maintenance of Multi-Robot Systems.* 2021. arXiv:2109.08536.
- Lazaridou & Baroni 2020. *Emergent Multi-Agent Communication in the Deep Learning Era (survey).* arXiv:2006.02419.
