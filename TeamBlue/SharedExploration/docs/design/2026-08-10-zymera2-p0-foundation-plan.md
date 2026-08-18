# zymera2 P0 — Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use superpowers:subagent-driven-development or
> superpowers:executing-plans to implement task-by-task. Steps use `- [ ]` checkboxes.

**Goal:** Stand up the standalone `zymera2` package with its corrected contract encoded as code
(`typing.py`) and construction-time validation (`validate.py`), test-first — the foundation every later
phase (world core, bridge, sensing, harness) builds on.

**Architecture:** Pure-JAX, `src/`-layout package. `typing.py` is the single source of every signature,
dtype, and shape (spec §14.1/C5). Dataclasses are `chex`/`@dataclass(frozen=True)` pytrees; caps are
two-tier (`StaticWorldParams` compile-time / `WorldParams` runtime, spec §4). No behavior yet — P0 is
the contract + validation, so its tests are construction, shape/dtype, and validation-rejection tests.

**Tech Stack:** Python ≥3.10, `jax`, `chex`, `numpy`; `pytest` (+`hypothesis` later). No CI/license/DOI
this phase (deferred behind the go/no-go gate, spec §14.6).

## Global Constraints (from spec §14 — every task inherits these)

- Package/import name **`zymera2`**; `src/`-layout; version `0.1.0.dev0`.
- Every signature lives **only** in `typing.py` (C5); nothing restates them.
- Action space fixed: `STAY=0, N=1, E=2, S=3, W=4` (§14.5).
- Two-tier params: `StaticWorldParams` = hashable compile caps (`H_max,W_max,N_max,M_max`, rule tuple);
  `WorldParams` = runtime values passed per call (§4, C-block).
- `world_step` signature is `(static, params, state, actions: i32[N_max], body_actions: i32[M_max],
  key) -> (state', events)` (C1) — declared in P0, implemented in P1.
- Interaction rules are an **immutable ordered tuple** in `StaticWorldParams`; **no mutable registry**
  (C8).
- `PayloadSpec` (dm_env-style) is part of the channel contract; payloads are `[N_max, *leaf]` fixed
  dtype (C2) — declared in P0, used in P3.
- RNG: `fold_in(base_key, id)` discipline, never sequential `split` for per-entity/edge keys (C6).
- No RNG in `World` state; no rewards/done/comms-state in `World` (spec §3).

---

### Task 1: Package skeleton + smoke import

**Files:**
- Create: `zymera2/pyproject.toml`
- Create: `zymera2/src/zymera2/__init__.py`
- Create: `zymera2/README.md`
- Create: `zymera2/.gitignore`
- Test: `zymera2/tests/test_smoke.py`

**Interfaces:**
- Produces: an importable `zymera2` package exposing `__version__`.

- [ ] **Step 1: Write the failing test** — `zymera2/tests/test_smoke.py`

```python
def test_import_and_version():
    import zymera2
    assert isinstance(zymera2.__version__, str)
    assert zymera2.__version__.startswith("0.1.0")
```

- [ ] **Step 2: Create `pyproject.toml`** (lean — no CI/license/entrypoints yet)

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "zymera2"
version = "0.1.0.dev0"
description = "A pure-JAX discrete grid-world simulator (world-only core)."
requires-python = ">=3.10"
dependencies = ["jax>=0.4", "chex>=0.1", "numpy"]

[project.optional-dependencies]
dev = ["pytest>=8.0", "hypothesis>=6.0"]
viz = ["matplotlib>=3.5", "pillow"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 3: Create `src/zymera2/__init__.py`**

```python
"""zymera2 — a pure-JAX discrete grid-world simulator (world-only core).

The world owns maps, entities, movement, collision, sensing, and events. It has no
rewards, no episodes, no missions, no communication (that is the ``zymera2.comms`` bridge),
and no agent machinery. See docs/design/2026-08-10-zymera2-world-design.md (esp. §14).
"""
__version__ = "0.1.0.dev0"
```

- [ ] **Step 4: Create `README.md`** (one paragraph) and `.gitignore` (`__pycache__/`, `*.egg-info/`,
  `.venv/`, `.pytest_cache/`).

- [ ] **Step 5: Create the venv + install, run the test**

```bash
cd zymera2 && python3 -m venv .venv && . .venv/bin/activate && pip install -e ".[dev]"
pytest tests/test_smoke.py -q
```
Expected: 1 passed.

- [ ] **Step 6: init git + commit**

```bash
cd zymera2 && git init && git add -A && git commit -m "chore: zymera2 package skeleton + smoke test"
```

---

### Task 2: The contract — `typing.py` (state + params + Protocols)

**Files:**
- Create: `zymera2/src/zymera2/typing.py`
- Test: `zymera2/tests/test_contract.py`

**Interfaces:**
- Consumes: nothing.
- Produces (later phases rely on these EXACT names/types):
  - `class ActionId(IntEnum): STAY=0 N=1 E=2 S=3 W=4`
  - `@chex.dataclass(frozen=True) class World:` fields `agent_pos:i32[N_max,2]`,
    `agent_alive:bool[N_max]`, `body_pos:i32[M_max,2]`, `body_alive:bool[M_max]`,
    `body_kind:i32[M_max]`, `wall:bool[H_max,W_max]`, `arena:bool[H_max,W_max]`, `step_count:i32`.
  - `@dataclass(frozen=True) class StaticWorldParams:` `h_max:int, w_max:int, n_max:int, m_max:int,
    rules:tuple`.
  - `@dataclass(frozen=True) class WorldParams:` `h:int, w:int, n:int, m:int, sense_r:int` (comms radius is bridge config — review fix).
  - `@dataclass(frozen=True) class PayloadSpec:` `shape:tuple, dtype:Any`.
  - `@chex.dataclass(frozen=True) class KernelEvents:` `moved:bool[E_max]`, `blocked:bool[E_max]`,
    `conflict:bool[E_max,E_max]` (unified entity index, E=N+M), `captured:bool[M_max]`. No `sensed` field —
    sensing is a harness-side readout with its own SenseEvent (§14/C4; review fix).
  - Protocols: `Generator`, `InteractionRule`, `Topology`, `Channel`, `PolicyFn`, and a `WorldStepFn`
    type alias — all with the spec-§14 signatures, bodies `...`.

- [ ] **Step 1: Write the failing test** — `zymera2/tests/test_contract.py`

```python
import jax.numpy as jnp
import chex
from zymera2 import typing as zt

def test_action_ids():
    assert (zt.ActionId.STAY, zt.ActionId.N, zt.ActionId.E, zt.ActionId.S, zt.ActionId.W) == (0,1,2,3,4)

def test_world_constructs_with_declared_shapes():
    N, M, H, W = 4, 2, 8, 8
    w = zt.World(
        agent_pos=jnp.zeros((N,2), jnp.int32), agent_alive=jnp.ones((N,), bool),
        body_pos=jnp.zeros((M,2), jnp.int32), body_alive=jnp.ones((M,), bool),
        body_kind=jnp.zeros((M,), jnp.int32),
        wall=jnp.zeros((H,W), bool), arena=jnp.ones((H,W), bool),
        step_count=jnp.zeros((), jnp.int32))
    chex.assert_shape(w.agent_pos, (N,2)); chex.assert_shape(w.wall, (H,W))
    assert w.agent_pos.dtype == jnp.int32

def test_world_has_no_forbidden_fields():
    # world holds no reward/done/comms/RNG (spec §3)
    forbidden = {"reward","done","channel","comm_graph","key","rng","belief"}
    assert forbidden.isdisjoint(set(zt.World.__dataclass_fields__))

def test_protocols_present_and_runtime_checkable():
    for name in ("Generator","InteractionRule","Topology","Channel","PolicyFn"):
        assert hasattr(zt, name)
    assert isinstance(zt.PayloadSpec(shape=(1,), dtype=jnp.float32), zt.PayloadSpec)

def test_rules_are_a_tuple_on_static_params():
    sp = zt.StaticWorldParams(h_max=8, w_max=8, n_max=4, m_max=2, rules=())
    assert isinstance(sp.rules, tuple)  # immutable ordered — no registry (C8)
```

- [ ] **Step 2: Run to verify it fails** — `pytest tests/test_contract.py -q` → FAIL (no `typing`).

- [ ] **Step 3: Implement `src/zymera2/typing.py`** — the full contract. `IntEnum` actions;
  `chex.dataclass` `World` and `KernelEvents`; `@dataclass(frozen=True)` `StaticWorldParams`,
  `WorldParams`, `PayloadSpec`; `@runtime_checkable Protocol` classes `Generator`
  (`__call__(self, gparams, key) -> tuple["WorldParams","World"]`), `InteractionRule`
  (`apply(self, static, params, state, events) -> tuple["World","KernelEvents"]`), `Topology`
  (`adjacency(self, geom) -> "Array"`), `Channel` (`init(self, topology, static, payload_spec)
  -> "ChannelState"`; `deliver(self, geom, payloads, cstate, key) -> tuple`), `PolicyFn`
  (`__call__(self, obs, state, key) -> tuple`); a `WorldStepFn` `Callable` alias with the C1
  signature. Every method body is `...`; every signature carries a one-line shape/dtype docstring.

- [ ] **Step 4: Run to verify it passes** — `pytest tests/test_contract.py -q` → 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/zymera2/typing.py tests/test_contract.py
git commit -m "feat(contract): typing.py — corrected world/params/events/Protocols (spec §14.1)"
```

---

### Task 3: Construction-time validation — `validate.py`

**Files:**
- Create: `zymera2/src/zymera2/validate.py`
- Test: `zymera2/tests/test_validate.py`

**Interfaces:**
- Consumes: `StaticWorldParams`, `WorldParams` from Task 2.
- Produces: `validate_params(static, params) -> None` (raises `ValueError` with an actionable message at
  Python/trace-construction time — never inside a jitted step, spec §10.1); `validate_seed_pools(pools:
  dict[str, set[int]]) -> None` (raises if pools intersect, C-block/G13).

- [ ] **Step 1: Write the failing test** — `zymera2/tests/test_validate.py`

```python
import pytest
from zymera2 import typing as zt
from zymera2.validate import validate_params, validate_seed_pools

def _static(): return zt.StaticWorldParams(h_max=8, w_max=8, n_max=4, m_max=2, rules=())
def _params(**kw):
    d = dict(h=8, w=8, n=4, m=2, sense_r=1); d.update(kw); return zt.WorldParams(**d)

def test_valid_params_pass():
    validate_params(_static(), _params())  # no raise

def test_runtime_exceeds_caps_rejected():
    with pytest.raises(ValueError, match="exceeds"):
        validate_params(_static(), _params(h=16))
    with pytest.raises(ValueError, match="exceeds"):
        validate_params(_static(), _params(n=8))

def test_nonpositive_rejected():
    with pytest.raises(ValueError):
        validate_params(_static(), _params(sense_r=0))

def test_disjoint_seed_pools_ok_and_overlap_rejected():
    validate_seed_pools({"train": {1,2,3}, "test": {4,5}, "eval": {6}})
    with pytest.raises(ValueError, match="overlap"):
        validate_seed_pools({"train": {1,2}, "test": {2,3}})
```

- [ ] **Step 2: Run to verify it fails** — `pytest tests/test_validate.py -q` → FAIL.

- [ ] **Step 3: Implement `src/zymera2/validate.py`** — `validate_params` checks `h<=h_max, w<=w_max,
  n<=n_max, m<=m_max` (raise `ValueError(f"runtime {name}={v} exceeds cap {cap}")`), positivity of
  `sense_r, h, w, n`, `rules` is a `tuple`. `validate_seed_pools` checks pairwise-disjoint,
  raising `ValueError("seed pools overlap: ...")`.

- [ ] **Step 4: Run to verify it passes** — `pytest tests/test_validate.py -q` → passed.

- [ ] **Step 5: Run the whole suite + commit**

```bash
pytest -q   # all P0 tests green
git add src/zymera2/validate.py tests/test_validate.py
git commit -m "feat(validate): construction-time param + seed-pool validation (spec §14, G13)"
```

---

## P0 exit criteria (the "solid foundation" gate)

`pip install -e ".[dev]"` clean · `pytest -q` all green · `import zymera2` and
`from zymera2 import typing as zt` work · the contract file is the single source of every signature ·
no forbidden fields on `World` · validation rejects over-cap params and overlapping seed pools.
**Only then** P1 (world core: worldgen · `world_step` unified conflict pass · interaction rules) gets
its own plan.

## Self-review

- **Spec coverage (P0 slice):** C1 signature declared (Task 2 `WorldStepFn`) ✓ · C2 `PayloadSpec` ✓ ·
  C5 single-source `typing.py` ✓ · C7 `PolicyFn` ✓ · C8 rules-as-tuple ✓ · §3 no forbidden World
  fields ✓ · G13 seed-pool disjointness ✓ · two-tier params ✓. (Behavioral corrections C3/C4/C6 and
  the test suite §14.4 belong to P1–P3, correctly out of P0 scope.)
- **Placeholders:** none — every step has real code or an exact command.
- **Type consistency:** `StaticWorldParams(h_max,w_max,n_max,m_max,rules)` and
  `WorldParams(h,w,n,m,sense_r)` used identically in Tasks 2 and 3 ✓ (comm_r removed by post-P0 review).
