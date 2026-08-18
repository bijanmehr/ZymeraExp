# zymera2 — World Simulator Design Specification

**Status:** draft for review · **Date:** 2026-08-10 · **Scope:** design only (no implementation this cycle)
**Session record:** L3/env brainstorm, 2026-08-10 · **Companion spec:** `2026-08-10-l3-agent-architecture-design.md`
**Evidence base:** two survey scouts (JAX-native: Jumanji/JaxMARL/Gymnax/PGX/Craftax/Kinetix/Brax/MJX/dm_control ·
classic: Minigrid/Griddly/dmlab2d+MeltingPot/mesa/PettingZoo), source-verified; citations inline.

---

## 1 · Purpose and scope

**zymera2 is a world, not a mind — and not a referee.** A published, standalone, pure-JAX discrete
grid-world simulator: it generates maps, moves entities, resolves collisions, answers sensing queries,
and reports events. It contains **no rewards, no episode semantics, no missions, no communication, and
no agent machinery.** Flexibility target: RL missions, classical-algorithm studies, pursuit games,
network sims, and demos run on the same core with equal indifference.

The four-layer stack (software mirrors the research formalism micro/bridge/macro):

```
zymera2.world     the physical substrate — maps · entities · movement · collision · sensing readout
zymera2.comms     the BRIDGE — separate module; consumes world geometry via a narrow query API;
                  owns topology/delay/dropout/occlusion/delivery of OPAQUE payloads
task layer        missions: reward terms · done · metrics · ledger · NPC scripts · oracles
                  (zymera_lab during research; small public companion package at paper release)
agent layer       KB · proposers · L3 · nets · trainers (zymera_lab, private)
```

Dependency direction is enforced: `world` imports nothing above it; `comms` imports only the world's
geometry queries; the world never learns that messages exist.

**Precedent:** the world-only core follows the physics-pipeline convention (Brax `pipeline_step`,
`mjx.step`, dm_control Physics/Task) and Melting Pot's substrate/scenario split. **No published JAX
gridworld ships this split, and no surveyed simulator models radio in the world core** (MPE's `state.c`
register is a zero-fidelity broadcast with delivery semantics in scenario code) — both halves of the
design are unoccupied ground, not deviations.

## 2 · Package identity

| Item | Decision |
|---|---|
| Repo | `zymera2` — new independent git repo at the workspace root; GitHub private until release |
| Import / distribution name | `zymera2` (coexists with frozen v1 `zymera` in the lab venv during migration; PyPI name availability checked at release) |
| Version | semver, `0.1.0-dev` → **`1.0.0` tagged at paper release**; per-world-behavior `-vN` recipe versioning (Gymnasium/PGX discipline: bump on any behavior change) |
| License | Apache-2.0 (patent grant; dependency tree stays GPL-free) |
| Citation | `CITATION.cff` from first commit |
| Core deps | `jax`, `chex`, `numpy` only; extras `[viz]` (matplotlib/pillow), `[dev]` (pytest/hypothesis/ruff/pyright) |
| CI | GitHub Actions: py3.10–3.13 × linux+mac · ruff · pyright · pytest · coverage floor · **performance benchmark job** (steps/sec regression gate — the field's #1 complaint is throughput decay; NAVIX/XLand-MiniGrid/mesa-frames all exist because of it) |

```
zymera2/
├── pyproject.toml · LICENSE · README · CHANGELOG · CITATION.cff
├── src/zymera2/
│   ├── typing.py            # all Protocols — the contract in one importable place
│   ├── validate.py          # construction-time validation; debug-mode chex asserts
│   ├── params.py            # StaticWorldParams / WorldParams (two-tier, §4)
│   ├── state.py             # World state: entity arrays + property layers (§3)
│   ├── world.py             # world_step (§4) · conflict resolution
│   ├── worldgen/            # Generator protocol + families + file loaders + seed pools (§5)
│   ├── sensing.py           # pure readout functions over state (§6)
│   ├── events.py            # typed event stream (§7)
│   ├── geometry.py          # the narrow query API the bridge consumes (§8)
│   ├── comms.py             # the bridge module (§8)
│   ├── rollout.py           # minimal composition convenience (policy = opaque callable)
│   └── viz/                 # top-down · isometric (ported from archived v0 iso.py) · frame tools
├── tests/                   # unit · property · parity · leakage · oracles · perf
├── docs/                    # quickstart · concepts · assumptions · how-tos · adr/ · audit/
├── examples/
└── .github/workflows/ci.yml
```

## 3 · State model

**Pattern:** entity arrays + property layers (Mesa 3.0's space/PropertyLayer separation, mapped to
JAX struct-of-arrays; avoids Minigrid's one-object-per-cell rigidity and Python-object-per-agent
state — the two documented forks-causing limitations).

```python
@chex.dataclass(frozen=True)
class World:
    # entities — struct-of-arrays, fixed caps from StaticWorldParams, alive-masked
    agent_pos:   i32[N_max, 2];  agent_alive: bool[N_max]
    body_pos:    i32[M_max, 2];  body_alive:  bool[M_max];  body_kind: i32[M_max]
    # property layers — named per-cell field arrays
    wall:        bool[H_max, W_max]
    arena:       bool[H_max, W_max]      # validity mask → non-rectangular arena SHAPES
    # bookkeeping
    step_count:  i32
```

- **Generic bodies** are how mission entities exist without the world knowing missions: the task layer
  supplies controllers that drive bodies (a "lead," an "evader" are task-side notions; the world moves
  bodies and reports them).
- Property layers are extensible by name (terrain cost, hazard fields later) without schema surgery.
- No RNG in state; keys are passed per call. No rewards, no done flags, no comms state in `World`.

## 4 · Core API — the world step

```python
world_step(static: StaticWorldParams, params: WorldParams,
           state: World, actions: i32[N_max], key) -> (World, Events)
```

- **Reward-free, done-free, no auto-reset.** Rewards are the task layer's job (dm_control
  `get_reward(physics)` pattern); auto-reset-in-step is an indicted anti-pattern (gymnax/JaxMARL
  discard the true terminal observation via `lax.select`; Craftax documents invalid-state episodes).
  The world can step forever; episodes are a harness concept.
- **Two-tier parameters** (the one working JAX answer to "different sizes," from Kinetix/gymnax):
  `StaticWorldParams` = compile-time caps (`H_max, W_max, N_max, M_max` — hashable; recompile across
  size tiers) · `WorldParams` = runtime values passed per call (actual H×W, actual N, physics knobs —
  vmappable, sweepable without recompiling). Pad-to-caps + `arena`/alive masks inside a tier.
- **Simultaneous actions with documented deterministic conflict resolution** (two-stage: forbid moves
  onto occupied cells; break same-empty-cell claims by index priority — collision-freedom is a stated,
  tested world guarantee). Explicit because hidden tie-break order is a documented multi-agent bug
  class (PettingZoo's AEC-vs-parallel race-condition analysis).
- Determinism: same `(static, params, state, actions, key)` → bit-identical `(state', events)` on CPU;
  stated policy in docs; GPU caveats documented.

**§4b · Interaction mechanics (agent↔agent, agent↔body — beyond collision).** The world owns every
interaction that operates through *space and contact*; the bridge owns interaction through *signals*.
Contact/proximity mechanics are a first-class, extensible rule system (Griddly's declarative
proximity/behavior rules; Melting Pot's zaps are world actions, not radio):

```python
class InteractionRule(Protocol):
    def apply(self, static, params, state, events) -> (state', events')
    # pure; runs inside world_step after movement resolution; emits typed events
```

Shipped rules: `Collision` (the §4 guarantee, expressed in the same frame) · `Tag/Capture(radius, k)`
(an entity adjacent to ≥k agents flips captured/frozen — cornering's contact physics) · `Block` ·
extensible by registration. Rules are world physics: they read positions/kinds, never payloads or
beliefs; scoring a capture remains the task layer's job (it watches the `captured` event). The rule
SET is static per world version (`-vN` bumps on change), keeping traces fixed-shape and behavior
versioned.

## 5 · Worldgen

**Pattern:** Jumanji's `Generator` abstraction **with jit-compatible parameters** (explicitly fixing
Jumanji issue #212 — parameterless reset blocks curricula and splits), plus Griddly's lesson that
declarative map families work *with a code escape hatch*.

```python
class Generator(Protocol):
    def __call__(self, gparams, key) -> (WorldParams, World)   # a fresh world instance
```

- **Families:** open · random-obstacle(density) · rooms(n) · corridors · pillars(spacing, size) ·
  mazes **incl. looped mazes** (cycle-guaranteed — the tether mission's hard precondition) · composite.
- **Shapes:** the `arena` mask makes non-rectangular arenas (L-shaped, ring, arbitrary) first-class —
  no surveyed simulator supports this natively.
- **Files:** authored ASCII arenas (the existing `experiments/maps/*.txt` format) + MovingAI loader.
- **Seed discipline:** disjoint named seed pools (`train` / `test` / `eval-frozen`) as a first-class
  Generator feature — the generalization methodology is packaged, not ad-hoc.
- **Provenance:** every generated world carries a content hash into run manifests.

## 6 · Sensing — pure readouts, not step side-effects

dm_control-observables pattern: sensing is a library of pure functions over `World`, composed by the
harness into observations; the step never bakes an observation format in (Minigrid's hardcoded
encoding is the counterexample that forced forks).

```python
sense_occupancy(state, agent_id, radius)  -> local patch {unknown/free/wall} (ground truth within range)
sense_entities(state, agent_id, radius)   -> in-range agents/bodies (the declared "sight" assumption)
sense_statics(static, params)             -> declared statics (arena bounds/boundary ring, if enabled)
```

dm_env-style specs describe every readout's shape/dtype. **The world emits sensed truth only** — there
is no belief anywhere in the package, so there is nothing world-side to leak.

## 7 · Events — typed instrumentation stream

Melting Pot's events system, generalized: `world_step` (and `comms.deliver`) emit a typed, fixed-shape
event record — moves, blocked moves, conflicts resolved, cells sensed, deliveries (from the bridge).
Events are **observer-only**: the task layer computes rewards/metrics/the ledger from them; agents
never receive events. This is what lets scoring exist entirely outside the world.

## 8 · The bridge — `zymera2.comms`

Communication is **not a world property** (ruling 2026-08-10; consistent with every surveyed
simulator). It is a separate module — the research program's *bridge* layer made literal:

```python
geometry.py (world-side, the ONLY surface comms sees):
    pairwise_dist(state, params)          # Chebyshev (metric pluggable)
    wall_crossings(state, a, b)           # segment wall-count → occlusion d_eff
class Topology(Protocol):   adjacency(geom) -> bool[N,N]          # who COULD talk
class Channel(Protocol):
    init(topology, static) -> ChannelState                         # ring buffers etc.
    deliver(geom, payloads: PyTree, cstate, key)
        -> (mail: PyTree, delivered: bool[N,N], cstate', Events)   # who DID talk, what arrived
```

- **Payloads are opaque** — authored by agents (see companion spec), transported subject to range,
  delay, dropout, occlusion. The bridge never interprets content; content cannot leak *from* the world
  because the world never held it.
- Delivery semantics are first-class and testable — the exact thing MPE cannot express (its delivery
  lives in scenario obs code) and no surveyed package models. This module is a stated contribution of
  the release.
- The harness composes a tick: `world_step → geometry → comms.deliver → obs assembly` (assembly =
  sensing readouts + mail + delivered-adjacency row; performed by the task/rollout layer).

## 9 · Task-layer contract (interface only — lives above the package)

For reference, so the boundary is checkable: a Task supplies `generator`, NPC controllers for bodies,
`reward(events, state, state')`, `done(state)`, metrics, the **ledger** (honest-union information-cone
oracle computed from sense+delivery events — the later definition of "a lie" is broadcast-vs-ledger
divergence), and frozen, versioned evaluation scenarios (Melting Pot discipline). Ships in
`zymera_lab` during research; published as a small companion package at paper time (results are not
reproducible from the world alone).

## 10 · Engineering standards (definition of done)

1. **Contract:** every Protocol in `typing.py`, typed, documented per method (shapes, dtypes, key
   protocol, conflict rules); construction-time validation; debug chex asserts for invariants
   (collision-freedom, mask validity, event-shape stability).
2. **Tests:** unit + **property-based** (hypothesis: collision-freedom under random joint actions,
   arena-mask containment, conflict-resolution determinism) + golden parity per `-vN` + **leakage
   suite** (T1 cone-nullity · T2 counterfactual non-interference, bit-exact on CPU · T3 channel
   ablation · T4 delivered-graph consistency · T5 static whitelist guard) + **independent oracles**
   (graph quantities vs networkx/scipy; paths vs brute-force BFS; hand-computed cases) + perf gate.
3. **Audit:** nothing ports from v1 unaudited — per-component verification cards in `docs/audit/`
   (claimed semantics → property tests → oracle cross-check → sign-off); defaults audit (dangerous
   knobs get no default); determinism audit. The audit report ships with the package.
4. **Docs:** quickstart (10 lines → a GIF) · concepts (world/bridge/events — the sketchbook intuition)
   · **assumptions page** (exact pose, shared frame, bounded arena, sight-of-neighbors — declared
   once, cited by the paper) · how-to per extension point (generator, channel, property layer,
   observer) · ADRs (first three: world-only core; comms-as-bridge; reward-free step).
5. **Reproducibility bill:** lock file/container · run manifests (map hash + params + seeds + package
   version) · one-command regeneration of paper figures (task-layer script) · DOI-archived artifact
   bundle at release.

## 11 · Migration

- **v1 (`zymera` in `zymera_lab`) is frozen** with its golden files — archival reproduction of all
  pre-v2 results; never edited again.
- `zymera_lab` consumes `zymera2` as a pinned pip-from-git dependency; pipelines migrate per the
  companion spec.
- Leak closures L1–L3 (λ₂-oracle mask · god-view planner fallback · potential-vs-delivered adjacency)
  are **structurally impossible** in zymera2 rather than patched: no mask mechanism in the world, no
  belief to fall back from, delivered adjacency is what the bridge emits.

## 12 · Survey ledger (adopt / avoid)

**Adopted, with attribution:** reward-free physics step (Brax/MJX/dm_control) · substrate/scenario
split + frozen eval scenarios (Melting Pot) · two-tier static/runtime params, pad-to-caps, size tiers
(Kinetix/gymnax) · parameterized Generator (Jumanji, fixing #212) · entity-arrays + PropertyLayers
(Mesa 3.0) · pure sensing readouts + specs (dm_control/dm_env) · typed events (Melting Pot) ·
pluggable observers incl. isometric (Griddly) · `-vN` behavior versioning (Gymnasium/PGX) · explicit
conflict resolution (PettingZoo's race-condition analysis) · dict-per-agent boundary, stacked internals
(JaxMARL).

**Avoided, with evidence:** auto-reset in step (terminal-obs discard) · reward/done fused into the
world (why no surveyed world is mission-reusable) · comms as a state register with delivery in
scenario code (MPE) · parameterless reset (Jumanji #212) · one-object-per-cell + hardcoded obs
encodings (Minigrid forks) · framework-owned schedulers (deleted in Mesa 3.0) · per-entity Python
objects / host callbacks (the NAVIX/mesa-frames throughput lesson) · heavy native build chains
(dmlab2d/Griddly).

## 13 · Open items

PyPI name availability (release-time) · task-layer packaging form at release (companion package vs
paper artifact repo) · PettingZoo adapter (task-level, Tier-3) · Isaac Lab portability note (world and
L1 swap, bridge and KB port — recorded from the 2026-08-10 analysis) · GPU determinism statement
wording after first benchmarks.

---

## 14 · Corrections adopted from adversarial review (2026-08-10) — BINDING over §§1–13

Five hostile critics reviewed §§1–13. The following corrections are **normative and supersede** any
conflicting text above; the implementation builds from these, not from the earlier prose. Build scope is
**LEAN**: the corrected contract (§14.1) + the full test suite (§14.4) are the definition-of-done for
"solid"; the publish ceremony in §10.4/§10.5 (CI matrix, DOI, docs site, license, PyPI) is **deferred**
behind an explicit go/no-go gate on the P1/P2 probe results (§14.6).

### 14.1 · Contract fixes (the API holes)

- **C1 — bodies can move.** `world_step(static, params, state, actions: i32[N_max],
  body_actions: i32[M_max], key) -> (state', events)`. One **unified conflict pass** over all entities
  (agents + bodies); priority rule is explicit and part of `-vN` (default: lower global entity index
  wins; agents indexed before bodies). Agent↔body cell-claim and swap cases are defined, not left open.
- **C2 — payloads: uninterpreted, not shapeless.** Add `PayloadSpec` (a dm_env-style spec) passed to
  `Channel.init(topology, static, payload_spec) -> ChannelState`. Under jit, payload leaves are
  `[N_max, *leaf_shape]`, fixed dtype, declared at trace. The opacity claim is reworded everywhere:
  **contents are uninterpreted (never branched on — verified by the content-swap test), schema is
  static.** Changing what agents *say* (schema) is a re-baseline event; changing what they *send*
  (values) is not.
- **C3 — `mail` shape is fixed.** `mail: [N_max, N_max, *leaf] + delivered_mask: bool[N_max,N_max]`
  (each receiver row = senders' delivered payloads, masked). Dense and correct for our N≈10; a
  bandwidth-capped `[N_max, K_in, …]` variant is a documented later option, not v1.
- **C4 — events = stable kernel + per-rule extensions.** Kernel events are **dense masked arrays**,
  never a count-capped stream: `moved: bool[N]`, `blocked: bool[N]`, `conflict: bool[N,N]`,
  `alive/captured: bool[E]`, `delivered: bool[N,N]`. Each `InteractionRule` bundles its **own** event
  extension (so adding a mission's contact mechanic does not mutate the core schema). Sensing returns
  `(readout, SenseEvent)`; the **harness** owns event-log concatenation. This admits, rather than
  denies, that the event vocabulary is the union of built rules (ADR-004).
- **C5 — the tick is normative.** The composed tick order and the RNG key-tree live **in the package**
  (`world_tick`/`rollout.py` as the reference implementation), golden-tested at the **composed** level,
  not just per-function. Run manifests record the **tick-spec version**. `typing.py` is the single
  source of every signature; the HTML/docs embed excerpts, never restate them.
- **C6 — RNG discipline is a contract.** Per-entity / per-edge keys via `fold_in(base_key, id)` — never
  sequential `split` — so a counterfactual edit outside an agent's cone cannot perturb unrelated draws
  (this is what makes T2 meaningful). Golden files are **version-stamped** (JAX ≥0.5.0
  `threefry_partitionable` changes `split`/`fold_in` outputs); the lockfile covers runs *and* golden
  provenance.
- **C7 — a `PolicyFn` Protocol** lives in `typing.py`: `policy(obs, state, key) -> (action, outbox,
  state')` — the agent contract is versioned inside the package even though implementations live above.
- **C8 — interaction rules are an immutable ordered tuple** in `StaticWorldParams` (order ∈ `-vN`,
  optional per-rule key); **no mutable registry** (it silently freezes under jit).

### 14.2 · The bridge, corrected

- Delivery-time vs enqueue-time gating is **specified**: a payload enqueued at `t` is gated by
  adjacency/occlusion at **delivery time** `t+delay` (stated in the `Channel` contract; makes G6/T4
  authorable).
- `wall_crossings` rasterization algorithm is **pinned** (supercover) — occlusion counts are otherwise
  implementation-defined and break golden parity.
- The novelty claim is **scoped**: "first *among surveyed JAX-native RL gridworlds* to model a geometric
  lossy channel in a separate bridge layer"; the channel-modeling lineage is cited
  (ARGoS range-and-bearing, Webots Emitter/Receiver, ns-3/OMNeT++, the ONE DTN sim, `adversarial_comms`).
  Drop "unclaimed ground"/"the field's #1 complaint" (scientific register, per preference).

### 14.3 · Honest claims (the audit is narrower than marketed)

- **No unqualified "no leakage."** A three-row claim taxonomy is normative: **execution-time** transport
  (certified by T1–T5) / **training-time** (centralized, privileged — CTDE reward reads global state;
  *trained weights are a channel* — NOT certified) / **evaluation-time**. The published claim is exactly:
  "decision-time inputs are measurable w.r.t. each agent's information cone; training is centralized and
  privileged."
- **§11 "structurally impossible" is downgraded** to "the three v1 leaks are absent from the package by
  construction; harness misuse (e.g. a mask smuggled via `statics`) is guarded by a **typed `statics`
  schema frozen per `-vN`** + T5 + the deploy-graph audit."
- **The leakage suite must prove its own sensitivity:** a **mutation test** injects ≥5 known leak types
  (statics-smuggled λ₂ mask · god-view proposer · potential-for-delivered swap · out-of-cone sense ·
  stale-mail bypass) and ships **100% detection** as an artifact. A suite that never caught a planted
  leak certifies nothing.
- **The lie-oracle (task-layer, but specified here) is three tests, not one divergence:** (i) fabrication
  = broadcast content ∉ sender's cone (one-sided, vs a **shadow-KB replay** of the sender's actual input
  trace); (ii) self-contradiction across an agent's own broadcasts; (iii) intention-lie = claim-vs-behavior
  divergence. Staleness/compression get **budgets**, not lie labels. This requires an **observer-only
  harness→task edge carrying outboxes** — added to the boundary matrix as an observer class (the ledger
  cannot see content otherwise).
- **Determinism is tiered:** G11 scoped to "same jaxlib version + platform"; a **CPU bit-exact** T2 tier
  and a **GPU tolerance** T2 tier; each paper figure states which (bit vs statistical) — written before
  experiments, not deferred.

### 14.4 · Test suite = the definition of "solid" (the gate)

Nothing merges without: unit + **property** (collision-freedom under random joint actions ·
arena-mask containment · **padding-invariance** — output invariant to garbage in padded/out-of-arena
rows, the #1 pad-to-caps bug, currently untested · conflict determinism) + **golden parity** per `-vN`
at the composed-tick level + **leakage suite T1–T5 + the mutation test** + **independent oracles**
(graph quantities vs networkx/scipy · paths vs brute-force BFS · channel delivery-ratio vs closed-form)
+ a **spec-derived NumPy reference** differential-tested against the JAX world (external validation, not
same-author-only) + perf-regression gate. A minimal cone oracle ships **inside `tests/`** (T2 depends on
it; it must not live in the task layer).

### 14.5 · Minor but real

Action space defined in `typing.py` (`0=STAY,1=N,2=E,3=S,4=W`) · `sense_occupancy` radius static-or-capped
(runtime radius ⇒ max-radius `dynamic_slice`+mask) · tri-state occupancy `{unknown/free/wall}` is an
**obs-builder** encoding, not a world-state field (don't repeat Minigrid's baked encoding) · GPU
determinism is *committed to*, not deferred (integer core, sort-based conflict resolution, no float
scatter) · G1/G9 get **negative smuggling tests** (a vacuous "verified by: the API has no such output"
is not verification) · G14 scoped to worldgen (it does not cover rollouts).

### 14.6 · Build sequencing & the go/no-go gate

Phased, test-first (see the implementation plan): **P0** package skeleton + `typing.py` (the whole
corrected contract) + `validate.py`. **P1** world core (state · worldgen · `world_step` with unified
conflict pass + interaction rules) + its property/parity/oracle tests. **P2** sensing + events + the
harness tick (normative order + key tree) + the leakage suite (T1–T5 + mutation). **P3** the comms
bridge (geometry · topology · channel with delivery-time gating) + channel oracles. **P4** viz +
NumPy reference + audit cards.
**GATE:** the publish ceremony (§10.4 docs site, §10.5 DOI, CI matrix, Apache/PyPI) is **not started**
until the P1/P2 red-pilot probes say the substrate — not the nominal agent — is the binding constraint
worth publishing. Until then zymera2 is a clean, tested, private package `zymera_lab` depends on.
