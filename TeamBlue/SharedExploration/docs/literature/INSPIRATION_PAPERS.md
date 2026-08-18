# Inspiration papers — what shaped the design & architecture

**Living document (started 2026-08-09).** Every paper that materially shaped a design or architecture
decision in Zymera, organized by the component/decision it influenced, with *what we took* from each.
One line per paper; the full verdicts live in the memory `lit-*` corpus and the scout transcripts.
**Rule:** add a paper here when it changes a decision, not when it's merely read. Entries without an
arXiv/DOI are marked *(no id in our notes)* — verify before citing in a paper.

Companions: `site/related-work.html` (published positioning) · `site/inspirations.html`
(cross-disciplinary analogies) · `site/assets/sources/marl_taxonomy/` (70-entry general MARL/agents
taxonomy) · `STRATEGY.md` (~70-paper method bibliography, staged).

## 1 · Backbone — perception, comms, belief (the LPAC-KB stack)

| Paper | id | What we took |
|---|---|---|
| **LPAC** — Agarwal, Kumar, Ribeiro, T-RO 2025 | 2401.04855 (+ precursor 2109.15278) | THE backbone: CNN local map → K-hop GNN → MLP, imitation of a clairvoyant expert, zero-shot size transfer. Cited as substrate, not novelty; K1 mandatory baseline |
| Constrained-LPAC | 2409.11311 | Primal–dual constraints on the same backbone (constraint = coverage cost, *not* connectivity — our delta) |
| Tolstaya et al. — aggregation GNNs | 1903.10527, 2011.01119 | K-hop GNN imitation of centralized experts; the imitation→decentralization bridge |
| Gupta, Egorov, Kochenderfer 2017 | *(no id in our notes)* | Parameter sharing + agent indication — the shared-policy substrate we build on (and whose flood we then fight) |
| Deep Sets — Zaheer et al., NeurIPS 2017 | 1703.06114 | Count-invariant set pooling → the T4 count-invariant critic (setpool ≈ setattn) |
| GIN / 1-WL bound — Xu et al., ICLR 2019; Abboud et al. 2021 | 1810.00826; RNI *(id in asymmetry review)* | Why the shared GNN provably cannot split symmetric agents (expressiveness gap); RNI = randomness lifts it |
| MARVEL — IROS 2025 | 2502.20217 | Frontier + graph-attention, permutation-equivariant scale invariance @2/4/8 — external proof of our T1/backbone direction |
| Patil, Malegaonkar, Christensen 2026 ("Diamond Attention") | 2605.06825 | Flood-as-theorem: provable redundancy in parameter-shared policies; structured protocol-space masking |
| AttentionNeuron — Tang & Ha | 2109.02869 | Permutation-invariant corruption-robust aggregation (flagged cheap edit; doubles into the adversarial prong) |

**Our GCRN belief** (recurrent message-passing graph belief, size-invariant zero-shot) is our own build in
the recurrent-GNN class — no single source paper in our notes.

## 2 · Planner (L2) — value propagation

| Paper | id | What we took |
|---|---|---|
| **VIN** — Tamar et al., NeurIPS 2016 | 1602.02867 | Differentiable value iteration as a network layer — the class our L2 lives in |
| **MVProp / VProp** — Nardelli et al. 2019 | *(no id in our notes)* | The lightweight value-propagation planner we deploy (passability × max-prop flood); wins the per-step budget |
| GPPN — Lee et al., ICML 2018 · Highway/DT-VIN (depth fixes) · MSP | *(no ids in our notes)* | The planner-arm matrix; empirically mvprop ≈ gppn ≈ highway ≫ msp (all approximate the same wavefront) |
| Guez et al., ICML 2019 | *(no id in our notes)* | Reactive nets DO plan at large depth/data — frames the 25%-vs-79% gap as budget-relative, not architectural |
| Yamauchi 1997 — frontier exploration | *(classical)* | The frontier concept; `occ_frontier` channel is Yamauchi on the gossiped occupancy belief |

Design rule distilled from this block (memory `lit-multiL-arch-verdict`): **learn the planner only where
the cost field is unknown/dynamic/soft** — for known/static/uniform fields the deterministic wavefront
dominates (every learned variant approximates the exact BFS operator); occlusion `d_eff`/moving-lead/red
costs are exactly the soft case that justifies distilled MVProp.

## 2b · Goal level (L3) — added 2026-08-10 after the four-scout L3 check

| Paper | id | What we took |
|---|---|---|
| **Spatial Action Maps** — Wu et al., RSS 2020 | 2004.09141 | The indictment: dense map-endpoint heads 9.9–18.3 vs 18-way steering commands 1.2–4.1 (our stencil's shape); their short-commitment ablation = "wavering between endpoint targets" → commit-and-hold |
| **Active Neural SLAM** — Chaplot et al., ICLR 2020 | 2004.05155 | The canonical 2-level: goal = point on the built map, replanned every 25 steps, planner beneath — external validation of the T5 shape + the cadence anchor |
| SemExp (NeurIPS 2020) · PONI (CVPR 2022) | *(see verdict memory)* | The map-goal family's drift toward frontier-restricted potentials — candidates defined by content, not geometry |
| MAANS — Yu et al., ECCV 2022 · ACE — AAMAS 2023 | 2110.05734 · 2301.03398 | Spatial-categorical goal logits > continuous point regression (their worst variant); async event-triggered replanning beats rigid sync. Caveat: centralized, perfect comms — does not touch our seam |
| ARiADNE — Cao et al., ICRA 2023 (+ RA-L 2024) | 2301.11575 · 2403.10833 | The deflation: its action is a *local hop* on a free-space graph — candidate-attention per se unproven vs a stencil; the real gains = free-space-conforming candidates + frontier-utility features + attention context |
| NeuralCoMapping — CVPR 2022 | 2203.16319 | Robots↔frontier bipartite matching via GNN — the learned-assignment precedent that generalizes across scenes and robot counts |
| **Spatial Intention Maps** — Wu et al., ICRA 2021 | *(IEEE 9561359)* | Goal-as-a-map-channel in teammates' observations → the X2 coordination arm, and the ideal covert attack surface (lying about intent is a KL-boundable message) |
| IS — Kim, Park, Sung, ICLR 2021 | *(OpenReview)* | Intention sharing = attention-compressed imagined trajectories — the message-content upgrade path |
| Burgard et al., T-RO 2005 · Zlot & Stentz, ICRA 2002 | *(classical)* | The classical de-confliction anchors: utility-discount greedy (no bound) and market auctions — even crude coordination beats none |
| DGNN-GA — Goarin et al., RA-L 2024 | *(RA-L 9(5))* | Decentralized GNN goal assignment that wins under restricted comms — the learned-assignment line that fits our constraints (Sinkhorn/OT precedent: none found) |
| Nachum et al. 2019 · Dabney et al., ICLR 2021 · TempoRL, ICML 2021 | 1909.10618 · 2006.01782 · 2106.05262 | Commitment is the active ingredient (HRL's benefit ≈ temporally-extended exploration; repeated/learned action durations beat per-step) → the commit-K dial is evidence-backed |
| Hu et al., IEEE TVT 2020 — Voronoi + DRL | *(TVT 69(12))* | **Evaluated & declined for us**: partition gives exclusion by construction, but cells degenerate when soft connectivity keeps the swarm clustered |
| UPDeT — ICLR 2021 · ODIS — ICLR 2023 | 2101.08001 · *(OpenReview)* | Cross-config transfer works within one mission semantics; new semantics = finetune → mission identity belongs at the goal layer as data, backbone stays mission-blind |
| SF/GPI — Barreto NeurIPS 2017 · USFA ICLR 2019 | 1606.05312 · 1812.07626 | **Evaluated & declined for mission transfer**: linear-in-features reward assumption breaks on k-disjoint-path objectives; no multi-agent precedent |

## 3 · Credit, reward & the game underneath

| Paper | id | What we took |
|---|---|---|
| **Wolpert & Tumer — COIN / Wonderful-Life Utility** | cs/9908014 | THE difference reward `d_i = G − G_{−i}`; also the delivered-flow relay-credit precedent (packet routing) |
| Castellini et al. — Difference Rewards PG (Dr.Reinforce) | 2012.11258 | Exact difference-reward policy gradient — measurement and training unify on the exact `d_i` |
| SubMAPG (2026) · "Exact Is Easier" (2026) | 2605.13269 · 2603.06859 | Exact submodular difference reward on multi-robot coverage beats shared reward; exact > learned |
| SQDDPG — Wang et al., AAAI 2020 · SHAQ — NeurIPS 2022 | 1907.05707 · 2105.15013 | Sampled Shapley credit (COMA credit↔truth r≈0.13 vs Shapley 0.32); Shapley reserved for relay *synergies* (non-submodular) |
| COMA — Foerster et al., AAAI 2018 | 1705.08926 | **Evaluated & demoted**: per-agent counterfactual too sparse for delayed team-summed coverage; kept as the attention-critic diagnostic (`contribution.py`) |
| DVE — Li et al., ICML 2022 | *(no id in our notes)* | Confirms COMA's counterfactual advantage is high-variance (structural, not tuning) |
| Zhou et al., IROS 2020 — resilient submodular coverage | *(no id in our notes)* | remove-k + re-run as the resilience curve; efficient=brittle vs flooding=robust trade-off |
| Amir, Bettini, Prorok 2025 — "When is diversity rewarded?" | 2506.09434 | Theorem: submodular/subadditive rewards → homogeneous is optimal — our "more diversity ≠ more coverage" made formal |
| **Arslan, Marden, Shamma 2007** · Marden–Arslan–Shamma, IEEE SMC-B 2009 · Marden — state-based, Automatica 2012 | DOI 10.1115/1.2766722 | Marginal-contribution utilities ⇒ **exact potential game, Φ = welfare** (NOT in Wolpert–Tumer); coverage/consensus as potential games |
| Blume, GEB 1993 · Marden–Shamma, GEB 2012 | *(journal refs)* | Log-linear learning: Gibbs stationary `p ∝ exp(Φ/τ)`, stochastically stable = argmax Φ — the nominal null model |
| Vetta, FOCS 2002 — valid-utility games | *(no arXiv)* | PoA ≥ ½ for submodular welfare + marginal utilities; diff-reward team's welfare optimum is a pure NE |
| Marden & Wierman, OR 2013 · Gopalakrishnan et al., EC'13/MOR'14 | 1402.3610 | Utility-design constraints: diff-reward aligned but not budget-balanced; **budget-balance + PNE ⇔ Shapley family** |
| DSAC — Zhang et al., AAAI 2022 · LToS — NeurIPS 2022 | LToS 2112.08702 | Dual/shadow-price-as-learned-reward and learned reward-sharing over the graph — the delivered-flow relay-credit route |

## 4 · Trainers & heterogeneity

| Paper | id | What we took |
|---|---|---|
| MAPPO — Yu et al., NeurIPS 2021 | 2103.01955 | The CTDE PPO trainer family our `ctde_v0` implements |
| **HAPPO/HATRPO** — Kuba et al., ICLR 2022 | 2109.11251 | Advantage decomposition + sequential updates; **Prop. 1** = shared policies exponentially suboptimal at division-of-labor (the "trap of homogeneity" — informal name, cite the Prop.) — the proof-backed fix for *initiating* division |
| HARL — Zhong et al., JMLR 2024 · HASAC, ICLR 2024 | 2304.09870 · 2306.10715 | The consolidated heterogeneous-agent family; HASAC = MaxEnt bridge to the EBM lens |
| Fu et al., ICML 2022 | 2206.07505 | Provable shared-policy failure on multi-modal reward landscapes; individual-policy PG converges |
| SePS — Christianos et al., ICML 2021 · HyperMARL · Kaleidoscope, NeurIPS 2024 | 2102.07475 · 2412.04233 · 2410.08540 | The shared-yet-diverse middle ground (selective sharing / agent-conditioned hypernets / learnable masks) — reconciles heterogeneity with our count-invariance |
| Soft-Q — Haarnoja et al., ICML 2017 · FOP — ICML 2021 | 1702.08165 | π ∝ exp(Q/α): policies are EBMs; factorized MaxEnt in MARL |
| RODE · ROMA · CDS · Option-Critic · FuN · FMH · ALMA | 2010.01523 · 2003.08039 · 2106.02195 · 1609.05140 · 1703.01161 · 1901.08492 · 2205.14205 | The role/hierarchy canon we **evaluated and mostly declined**: RODE's own ablation shows the win is the effect-structured action space, not the role selector → our goal-region head, no discrete role head; FuN → directional/relative goals; FMH = goal-assignment over comms = our covert L3 surface |
| Hu et al. 2022 | 2207.05683 | Team size ≠ role diversity — role advantage does not grow with N |
| G2N (NeurIPS 2018) · ERL-Re² (ICLR 2023) | *(no ids in our notes)* | The ES-selector + gradient-executor coexistence pattern (disjoint params, shared return, timescale separation) |
| OpenAI-ES — Salimans et al. 2017 | 1703.03864 | The ES trainer used for the role-switcher/gate line (zymera_env era) |
| MAT (sequence-model MARL) · Sable, ICML 2025 | MAT *(see marl_taxonomy)* · 2410.01706 | Sequential decoding via advantage decomposition at scale — the trainer frontier we track, not yet adopt |

## 5 · Symmetry-breaking & division of labor

| Paper | id | What we took |
|---|---|---|
| **Angluin 1980** | *(classical)* | Impossibility: anonymous networks cannot break symmetry deterministically → unique IDs / randomness are justified primitives |
| DARP — Kapoutsis et al. 2017 | *(no id in our notes)* | Region partition for the deploy-then-release wrapper (the Tier-1 division mechanism) |
| Cortés, Martínez, Karatas, Bullo 2004 | *(classical)* | Lloyd / centroidal-Voronoi coverage — the canonical partition dynamics (+ the CVT expert LPAC imitates) |
| CBBA — Choi, Brunet, How 2009 | *(classical)* | Consensus-based auction — the market route to conflict-free region assignment |
| Fisher–Nemhauser–Wolsey 1978 · Corah & Michael (DSGA) · Grimsman et al. | 2107.08550 · 1807.10639 | Sequential-greedy submodular maximization (1−1/e) — the theory behind claim-round / sequential-greedy goal arbitration |
| Dispersion of mobile robots — Augustine & Moses; Kshemkalyani et al. | 1805.12242 · 2008.09379 | Graph-dispersion primitives (Tier-1 separated seeds without a snapshot) |

## 6 · Connectivity, topology & relays

| Paper | id | What we took |
|---|---|---|
| Zavlanos & Pappas, T-RO 2007/08 | *(classical)* | Connectivity-preserving potentials — soft pressure lets bridge behavior fall out of the gradient |
| Yang, Freeman, Lynch — decentralized λ₂ estimation (+ Sabattini, Ji–Egerstedt, Kim–Mesbahi) | *(classical set)* | The distributed Fiedler-estimation canon — and why we replaced online eigensolving with a local degree/component check + learned belief |
| Li et al., ICRA 2022 — CPO connectivity | 2109.08536 | λ₂ as CMDP cost holds 71–77% connectivity where unconstrained collapses — the constraint-vs-reward evidence |
| learning-connectivity — Mox et al. | 2112.07663 | CNN imitates an SDP maximizing λ₂ — one-shot relay placement (relays *assigned*, our emergence delta) |
| MWIoD flow stack — Mox · Calvo-Fullana | 2002.03026 · 2306.08737 | Infrastructure-on-demand routing SOCP; **dual prices = per-edge contribution signal** (delivered-flow credit route) |
| k-CMCS — Luo & Sycara, IROS 2019 | *(no id in our notes)* | k-node-connectivity CBF-QP; edge critical iff ≥ k+1 node-disjoint paths (max-flow test) — the TetherRelay `k` metric |
| Tokekar FCR/FBR | 2404.03834 · 2011.00685 | Connectivity restoration/recovery (minmax move) — the recovery axis |
| W-MSR — LeBlanc et al., JSAC 2013 | *(classical)* | Resilient consensus + (2f+1)-robustness — the overt-Byzantine canon our *covert* thesis is positioned against |
| Katifori & Corson, PRL 2010 | *(journal ref)* | Optimizing transport nets against damage FORCES loops — redundancy-by-design theorem behind the looped-maze precondition |
| IR2 — IROS 2024 | 2409.04730 | Learned intermittent-connectivity ("grace") — the connectivity↔explore trade-off is learnable |

## 7 · Missions, benchmarks & time-varying graphs

| Paper | id | What we took |
|---|---|---|
| VMAS — Bettini et al. | 2207.03530 | The vectorized benchmark family (same lab as LPAC/adversarial_comms — composable); mirrored, not forked |
| GLAS — Rivière et al. | 2002.11807 | Amortize a centralized optimum into local policies with safety — the imitation pattern |
| DNB — Galliera et al. | 2404.01551 | Dynamic network bridging (nearest MARL to tether) — single-path, no redundancy → extending to `k` is our gap |
| SUB-PLAY — CCS 2024 | 2402.03741 | Partial-obs red/blue sweep harness = the ΔJ(k)/stealth-sweep pattern (external adversary — adapt, not adopt) |
| MAGEC | 2403.13093 | Resilient patrolling under overt attrition/comms-loss — the fault (not covert) baseline family |
| Hüttenrauch et al. | 1807.06613 | Δ-disk local obs + mean-embedding swarm control — the classic pursuit/rendezvous substrate |
| Casteigts et al. — TVG framework | 1012.0009 | Time-varying-graph formalism: journeys, connected-over-time classes — formalizes no-hard-mask (delivered coverage = a journey exists) |
| Mertzios et al. — temporal flows · Kempe–Kleinberg–Kumar | *(journal refs)* | Out-disjoint journeys = temporal redundancy (temporal Menger fails ⇒ use flows); intermittent-delivery metrics |
| Williams & Musolesi 2016 · Zhao–Ammar–Zegura (message ferry) | *(journal refs)* | Temporal robustness R^λ(f) = the temporalized break budget; ferry-route objective |
| GAT-MARL lunar DTN | 2510.20436 | Store/forward/deliver unified as edge-choice incl. self-loop; size-invariant per-neighbor features |
| ACHORD (SubT forward-comms) | *(no id in our notes)* | The documented "SNR-lie" — the real-world anchor for the tether covert link-liar |

## 8 · Adversarial & detection (the thesis side)

| Paper | id | What we took |
|---|---|---|
| **adversarial_comms** — Blumenkamp & Prorok, CoRL 2020 | 2008.02616 | THE covert-insider bullseye: one agent learns deceptive GNN messages, +128% vs the team — the π̂ᵢ split made real, on coverage |
| Standen, Kim, Szabo — SoK adversarial MARL | 2301.04299 | The one authoritative survey (APOSG threat model); confirms no covert-internal-within-team survey exists |
| BLAST | 2501.01593 | k=1 covert backdoor → team failure — nearest minimum-compromise evidence |
| Wolfpack/WALL | 2502.02844 | Influential-position attacks (external) — the position-amplification H1 neighbor |
| SA-MDP / SA-PPO | 2003.08938 | Smoothness regularization at the belief→goal head — the stealth-bounded-message defense primitive |
| BARDec-POMDP | 2305.12872 | Byzantine type-belief at the aggregation point — borrowed for the detector side |
| Kazari, Shereen, Dán — IJCAI 2023 · ECAI 2025 | 2508.15764 | Decentralized per-agent normality scores + CUSUM (on actions) — nearest detector to RQ4 |
| Mitchell, Blumenkamp, Prorok | 2012.00508 | GP inter-message consistency → confidence-weighted filtering in GNN comms |
| AME — ICLR 2023 | *(OpenReview)* | Certified robustness via ablated message ensembles |
| BARD-MARL (2026) | 2606.20701 | Byzantine detection on learned comms at scale — the near-threat forcing our decentralization-mandatory hook |
| SHARP (2026) | 2602.08335 | Enabler credit + credit-filters-bad-agents in LLM-MAS — the encroaching neighbor to credit-as-detector |
| **Liu et al., NeurIPS 2020 — energy OOD** | 2010.03759 | The energy score: one forward pass, no sampling — the RQ4 detector primitive |
| Song & Kingma | 2101.03288 | EBM cost asymmetry: MCMC lives in training and is avoidable; scoring is cheap |
| GNNSafe — ICLR 2023 (+ bounded energies 2504.13429) | *(OpenReview)* | Per-node graph energies + energy belief propagation — size-agnostic by construction |
| Yehudai et al., ICML 2021 | *(PMLR)* | Size-generalization warning: scores drift when local structure shifts — the caveat on cross-size detection |
| Borowski–Marden CDC 2015 · Canty et al. · Paarporn et al. · Brown et al. | 1909.02671 · 1906.01142 · 1711.00609 | Adversaries inside game-theoretic learning (all overt/external) — the cluster our covert Gibbs-shift corner extends |

## 9 · Bio / physics / structural inspirations

| Paper | id | What we took |
|---|---|---|
| Tovar & Renaud — HCA topopt, J.Mech.Des. 2006 | *(journal ref)* | Decentralized load-path condensation from a local rule — same object as Physarum |
| Tero et al., Science 2010 — Physarum | *(journal ref)* | Decentralized transport-backbone growth + self-heal (robot mesh follow-up, Sci.Rep. 2025) |
| Runions et al., SIGGRAPH 2005 — space colonization | *(journal ref)* | Growth-toward-demand — the third face of the same condensation spine |
| Werfel et al., Science 2014 — TERMES | *(journal ref)* | Provable global-target → local-rule compilation (guarantees don't survive our adversarial port) |
| Holmberg, Thore, Klarbring, SMO 2017 | *(journal ref)* | Game-theoretic robust topopt IS a blue/red min-max — drop-in citable for the POSG |
| Sigmund 2011 | *(journal ref)* | BESO sensitivity ½uᵀKu is a true gradient ≡ our difference-reward credit ≡ remove-k probe |

## 10 · Methodology & surveys (the anchors)

Reproducibility: **Henderson — Deep RL that Matters** (1709.06560) · **Agarwal — rliable / statistical
precipice** (2108.13264) · Papoudakis MARL benchmark (2006.07869). Surveys (all verified 2026-08-06, memory
`ref-survey-anchors`): Pateria HRL (ACM CSUR 2021) · Kirk zero-shot generalization (JAIR 2023) · Zhu
transfer (TPAMI 2023) · Gronauer & Diepold + Oroojlooy & Hajinezhad + Zhang–Yang–Başar (cooperative MARL) ·
Ilahi adversarial RL (2001.09684) · Moerland MBRL (FnT 2023) · Wang robot-team RL (2204.03516) ·
Yuan/Zhang open-env MARL (2312.01058). **No survey covers our 4-way intersection** — the survey-level gap.

---

*Maintenance: when a paper changes a decision, add it to the right section with one line of "what we
took." When a section's verdict changes, fix the line — don't append history (that's `JOURNEY.md`'s job).
Sources for this compilation: the memory `lit-*` corpus, `STRATEGY.md`, `asymmetry_litreview.html`
references, and the 2026-08-09 verification scouts (potential games · EBM · heterogeneous credit).*
