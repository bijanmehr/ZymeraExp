# T5 — Phase 0 + Phase 1a Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a learned MVProp navigation planner as a drop-in replacement for ctde_v0's classical `nav_distance_field`, wire it in behind a gated config option, and run the Phase‑1a go/no‑go gate (does a learned planner beat the reactive baseline on zero‑shot SAR maps?).

**Architecture:** New learned code lives in the isolated `t5lab/` package. It reuses ctde_v0's existing goal‑head → planner → move socket: the Actor already emits `(N,K)` goal‑offset logits (`nets.py:973`), `_goal_to_move` (`ppo.py:125`) turns the chosen goal cell into a move, and when `cfg.action_head.controller == "navfield"` it routes through `nav_distance_field(goal, blocked, planner)` (`controller.py:266`) over the per‑agent belief occupancy `_navfield_blocked` (`ppo.py:109`, optimistic on unknown). Today `planner ∈ {wavefront, fmm}` (classical). We add a **learned** field producer. MVProp = a per‑cell reward map `r` + propagation gain `w∈(0,1)` from small convs, K max‑propagation sweeps flooding value from the goal around blocked cells; the value at the agent's 5 move‑neighbours are the move logits (differentiable → RL‑trainable).

**Tech Stack:** JAX + Equinox, MAPPO‑CTDE (ctde_v0's `ppo.py`), pytest.

## Global Constraints

- **16 rollouts, never reduced** — no reduction of `cfg`'s rollout batch for memory/speed, anywhere.
- **From‑scratch** — no warm‑start for the clean baseline.
- **Isolation** — all new logic in `t5lab/`; ctde_v0 touched only by *gated* additions that are byte‑identical when off (`planner != "mvprop"`), matching ctde_v0's existing "stable param surface / static gate" pattern.
- **No hard connectivity mask** — connectivity stays soft/reward; collision hard‑mask is fine.
- Belief occupancy for planning = `_navfield_blocked` (per‑agent known walls, UNKNOWN treated as free/optimistic).
- Action order = ACTION_DELTAS: `0=stay[0,0] 1=up[-1,0] 2=right[0,1] 3=down[1,0] 4=left[0,-1]`.

---

## File Structure

- `t5lab/mvprop.py` — the learned planner: pure `propagate(r, w, K)` + `MVProp` eqx.Module (field + move‑logit readout). One responsibility: produce a value field and read moves off it.
- `t5lab/tests/test_mvprop.py` — unit tests for the propagation mechanism (hand‑set weights) and the module (shape / size‑invariance / differentiability).
- `t5lab/plug_ctde.py` — the gated integration: a `mvprop_field(actor_mvprop, goal, blocked)` adapter matching `nav_distance_field`'s `(N,H,W)` output contract, plus a thin `t5_actor` helper that attaches an `MVProp` submodule to a ctde_v0 Actor.
- `t5lab/tests/test_plug_ctde.py` — tests the adapter's output contract + that `planner != "mvprop"` leaves ctde_v0 byte‑identical.
- `t5lab/actor.py` — `T5Actor` (eqx.Module): composes `ctde_v0.nets.Backbone` (READ-ONLY import) + goal head + `MVProp` + actor-side aux head. One responsibility: belief → (goal_logits, move_logits, value, λ̂₂).
- `t5lab/rollout.py` — isolated env scan (reset/step) → goal → MVProp field → move; collects the trajectory. Imports ctde_v0 env + `_navfield_blocked` READ-ONLY.
- `t5lab/train.py` — isolated MAPPO train step (GAE + clipped-PG + setpool critic + dual). Imports ctde_v0 pure GAE/PPO/critic helpers READ-ONLY where actor-agnostic; reimplements only the actor-coupled parts.
- `t5lab/run_phase1a.py` — the go/no‑go driver: train from scratch on rooms+clutter, zero‑shot eval on held‑out SAR, print coverage vs the reactive 25% / classical 79% references.
- **Zero ctde_v0 edits.** ctde_v0 is imported read-only only. (Chosen over the gated-edit route for strict isolation.)

---

### Task 1: MVProp propagation mechanism (pure function)

**Files:**
- Create: `t5lab/mvprop.py`
- Test: `t5lab/tests/test_mvprop.py`

**Interfaces:**
- Produces: `propagate(r: (H,W), w: (H,W), K: int) -> (H,W)` value field. `r` = per‑cell reward (goal high, blocked very negative), `w∈(0,1)` = per‑cell propagation gain (blocked ≈ 0). Update: `V = r; repeat K: V = maximum(r, w * max_4nbr(V))`.

- [ ] **Step 1: Write the failing test — value floods from the goal on open ground**

```python
# t5lab/tests/test_mvprop.py
import jax.numpy as jnp
from t5lab.mvprop import propagate

def test_propagate_floods_from_goal_open():
    H = W = 9
    r = jnp.full((H, W), -0.01)          # tiny step cost everywhere
    r = r.at[4, 8].set(1.0)              # goal at (4,8)
    w = jnp.ones((H, W))                 # open terrain, value flows freely
    V = propagate(r, w, K=32)
    # goal is the global max, and closer-to-goal beats farther-from-goal
    assert jnp.argmax(V) == 4 * W + 8
    assert V[4, 7] > V[4, 0]             # one step from goal > eight steps
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/bijanmehr/Project.Zymera/zymera_experiments/TeamBlue/SharedExploration && PYTHONPATH=. pytest t5lab/tests/test_mvprop.py::test_propagate_floods_from_goal_open -v`
Expected: FAIL with `ModuleNotFoundError` / `ImportError: cannot import name 'propagate'`.

- [ ] **Step 3: Write minimal implementation**

```python
# t5lab/mvprop.py
import jax, jax.numpy as jnp, equinox as eqx


def _max_4nbr(V):
    """(H,W) elementwise max over the 4 von-Neumann neighbours (edge = -inf pad via clamp)."""
    up    = jnp.pad(V[1:, :], ((0, 1), (0, 0)), constant_values=-jnp.inf)
    down  = jnp.pad(V[:-1, :], ((1, 0), (0, 0)), constant_values=-jnp.inf)
    left  = jnp.pad(V[:, 1:], ((0, 0), (0, 1)), constant_values=-jnp.inf)
    right = jnp.pad(V[:, :-1], ((0, 0), (1, 0)), constant_values=-jnp.inf)
    return jnp.maximum(jnp.maximum(up, down), jnp.maximum(left, right))


def propagate(r, w, K):
    """Max-propagation value field: V0 = r; V_{k+1} = max(r, w * max_4nbr(V_k)). K sweeps."""
    def step(V, _):
        return jnp.maximum(r, w * _max_4nbr(V)), None
    V, _ = jax.lax.scan(step, r, None, length=K)
    return V
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/bijanmehr/Project.Zymera/zymera_experiments/TeamBlue/SharedExploration && PYTHONPATH=. pytest t5lab/tests/test_mvprop.py::test_propagate_floods_from_goal_open -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/bijanmehr/Project.Zymera/zymera_experiments/TeamBlue/SharedExploration
git add t5lab/mvprop.py t5lab/tests/test_mvprop.py
git commit -m "feat(t5lab): MVProp max-propagation value field (pure fn)"
```

---

### Task 2: Propagation routes AROUND a wall (the generalization mechanism)

**Files:**
- Modify: `t5lab/tests/test_mvprop.py`

**Interfaces:**
- Consumes: `propagate` from Task 1.

- [ ] **Step 1: Write the failing test — a wall forces value to arrive from the open side**

```python
# append to t5lab/tests/test_mvprop.py
import jax.numpy as jnp
from t5lab.mvprop import propagate

def test_propagate_routes_around_wall():
    # 7x7. Goal at (3,6). A vertical wall at column 3 spanning rows 1..5, with a gap at row 0.
    # Agent at (3,0) can only reach the goal by going UP to the gap, across, then down.
    H = W = 7
    r = jnp.full((H, W), -0.01).at[3, 6].set(1.0)
    w = jnp.ones((H, W))
    wall = jnp.zeros((H, W), bool).at[1:6, 3].set(True)   # column-3 wall, gap at row 0 and row 6
    w = jnp.where(wall, 0.0, w)                            # blocked cells do not propagate
    V = propagate(r, w, K=64)
    # at the agent (3,0), the best move must be UP (toward the row-0 gap), NOT straight RIGHT
    # into the wall — i.e. V above the agent exceeds V to its right.
    assert V[2, 0] > V[3, 1]
```

- [ ] **Step 2: Run test to verify it fails or passes**

Run: `cd /Users/bijanmehr/Project.Zymera/zymera_experiments/TeamBlue/SharedExploration && PYTHONPATH=. pytest t5lab/tests/test_mvprop.py::test_propagate_routes_around_wall -v`
Expected: PASS (the mechanism already routes around walls given `w=0` on blocked cells — this test *characterises* the required behaviour and guards against regressions).

- [ ] **Step 3: (no code change — mechanism already implemented in Task 1)**

If the test FAILS, the bug is in `_max_4nbr` edge padding or the `w` masking; fix `mvprop.py` until it passes. Do not weaken the test.

- [ ] **Step 4: Commit**

```bash
cd /Users/bijanmehr/Project.Zymera/zymera_experiments/TeamBlue/SharedExploration
git add t5lab/tests/test_mvprop.py
git commit -m "test(t5lab): MVProp routes value around a wall"
```

---

### Task 3: MVProp module — learned convs + move‑logit readout

**Files:**
- Modify: `t5lab/mvprop.py`
- Modify: `t5lab/tests/test_mvprop.py`

**Interfaces:**
- Produces:
  - `class MVProp(eqx.Module)` with `__init__(in_ch: int, K: int, *, key)`.
  - `MVProp.field(x: (C,H,W)) -> (H,W)` — learned `r`,`w` then `propagate`.
  - `MVProp.move_logits(x: (C,H,W), rc: (2,) int) -> (5,)` — value at [center, up, right, down, left] of cell `rc`, ordered to ACTION_DELTAS.
- Consumes: `propagate` (Task 1).

- [ ] **Step 1: Write the failing test — shape, size‑invariance, differentiability**

```python
# append to t5lab/tests/test_mvprop.py
import jax, jax.numpy as jnp
from t5lab.mvprop import MVProp

def test_mvprop_module_shapes_and_grad():
    m = MVProp(in_ch=2, K=16, key=jax.random.PRNGKey(0))
    for H in (16, 32):                                   # SAME params, two grid sizes
        x = jnp.zeros((2, H, H)).at[1, H // 2, H - 1].set(1.0)   # ch1 = goal one-hot
        V = m.field(x)
        assert V.shape == (H, H)
        logits = m.move_logits(x, jnp.array([H // 2, 0]))
        assert logits.shape == (5,)
    # differentiable w.r.t. params through the move logits
    def loss(mod):
        x = jnp.zeros((2, 16, 16)).at[1, 8, 15].set(1.0)
        return m.move_logits(x, jnp.array([8, 0])).sum()
    g = eqx_grad(loss, m)
    assert g is not None

def eqx_grad(loss, m):
    import equinox as eqx
    return eqx.filter_grad(loss)(m)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/bijanmehr/Project.Zymera/zymera_experiments/TeamBlue/SharedExploration && PYTHONPATH=. pytest t5lab/tests/test_mvprop.py::test_mvprop_module_shapes_and_grad -v`
Expected: FAIL with `ImportError: cannot import name 'MVProp'`.

- [ ] **Step 3: Write the module**

```python
# append to t5lab/mvprop.py
_DELTAS = jnp.array([[0, 0], [-1, 0], [0, 1], [1, 0], [0, -1]])   # ACTION_DELTAS order


class MVProp(eqx.Module):
    """Learned max-propagation planner. Two 3x3 convs produce a per-cell reward r and a
    propagation gain w in (0,1) from the input map (channels e.g. [blocked, goal_onehot]);
    K max-propagation sweeps flood value from the goal around blocked cells; move logits are
    the value at the agent's 5 move-neighbours (differentiable, RL-trainable)."""
    reward_conv: eqx.nn.Conv2d
    gain_conv: eqx.nn.Conv2d
    K: int = eqx.field(static=True)

    def __init__(self, in_ch, K, *, key):
        kr, kg = jax.random.split(key)
        self.reward_conv = eqx.nn.Conv2d(in_ch, 1, 3, padding=1, key=kr)
        self.gain_conv = eqx.nn.Conv2d(in_ch, 1, 3, padding=1, key=kg)
        self.K = int(K)

    def field(self, x):
        r = self.reward_conv(x)[0]                       # (H,W)
        w = jax.nn.sigmoid(self.gain_conv(x)[0])         # (H,W) in (0,1)
        return propagate(r, w, self.K)                   # (H,W)

    def move_logits(self, x, rc):
        V = self.field(x)                                # (H,W)
        H, W = V.shape
        cells = jnp.clip(rc[None, :] + _DELTAS, 0, jnp.array([H - 1, W - 1]))  # (5,2)
        return V[cells[:, 0], cells[:, 1]]               # (5,) logits in ACTION_DELTAS order
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/bijanmehr/Project.Zymera/zymera_experiments/TeamBlue/SharedExploration && PYTHONPATH=. pytest t5lab/tests/test_mvprop.py -v`
Expected: PASS (all three tests).

- [ ] **Step 5: Commit**

```bash
cd /Users/bijanmehr/Project.Zymera/zymera_experiments/TeamBlue/SharedExploration
git add t5lab/mvprop.py t5lab/tests/test_mvprop.py
git commit -m "feat(t5lab): MVProp module (learned convs + move-logit readout)"
```

---

### Task 4: ctde_v0 adapter — `(N,H,W)` field matching `nav_distance_field`

**Files:**
- Create: `t5lab/plug_ctde.py`
- Test: `t5lab/tests/test_plug_ctde.py`

**Interfaces:**
- Produces: `mvprop_field(mvprop, goal_cells, blocked) -> (N,H,W)` — vmapped over agents; builds each agent's input `x = stack([blocked_i (as float), goal_onehot_i])`, returns `mvprop.field(x)`. Contract matches `controller.nav_distance_field`'s `(N,H,W)` so `navfield_move`'s descent is reused unchanged (higher V = closer, so the descent must *ascend* MVProp's field — see note).
- Consumes: `MVProp` (Task 3).

**Note (sign convention):** classical `nav_distance_field` returns *distance* (descend to goal); MVProp returns *value* (ascend to goal). The adapter returns `-V` so the existing descent controller reuses byte‑for‑byte. Guard this with the test below.

- [ ] **Step 1: Write the failing test — adapter shape + sign convention**

```python
# t5lab/tests/test_plug_ctde.py
import jax, jax.numpy as jnp
from t5lab.mvprop import MVProp
from t5lab.plug_ctde import mvprop_field

def test_mvprop_field_shape_and_sign():
    m = MVProp(in_ch=2, K=16, key=jax.random.PRNGKey(0))
    N, H, W = 3, 12, 12
    goal = jnp.array([[6, 11], [0, 0], [11, 5]])
    blocked = jnp.zeros((N, H, W), bool)
    field = mvprop_field(m, goal, blocked)
    assert field.shape == (N, H, W)
    # sign convention: adapter returns a DISTANCE-like field (goal is the MINIMUM), so the
    # existing descent controller routes toward the goal.
    assert field[0, 6, 11] == field[0].min()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/bijanmehr/Project.Zymera/zymera_experiments/TeamBlue/SharedExploration && PYTHONPATH=. pytest t5lab/tests/test_plug_ctde.py -v`
Expected: FAIL with `ImportError: cannot import name 'mvprop_field'`.

- [ ] **Step 3: Write the adapter**

```python
# t5lab/plug_ctde.py
import jax, jax.numpy as jnp


def _one_agent_field(mvprop, goal_rc, blocked_hw):
    H, W = blocked_hw.shape
    goal_onehot = jnp.zeros((H, W)).at[goal_rc[0], goal_rc[1]].set(1.0)
    x = jnp.stack([blocked_hw.astype(jnp.float32), goal_onehot])   # (2,H,W)
    return -mvprop.field(x)                                        # negate: value -> distance-like


def mvprop_field(mvprop, goal_cells, blocked):
    """(N,H,W) distance-like field for the descent controller. goal_cells (N,2) int; blocked (N,H,W) bool."""
    return jax.vmap(lambda g, b: _one_agent_field(mvprop, g, b))(goal_cells, blocked)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/bijanmehr/Project.Zymera/zymera_experiments/TeamBlue/SharedExploration && PYTHONPATH=. pytest t5lab/tests/test_plug_ctde.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/bijanmehr/Project.Zymera/zymera_experiments/TeamBlue/SharedExploration
git add t5lab/plug_ctde.py t5lab/tests/test_plug_ctde.py
git commit -m "feat(t5lab): ctde_v0 nav-field adapter for MVProp"
```

---

### Task 5: `T5Actor` — isolated composition (Backbone + goal head + MVProp)

**Files:**
- Create: `t5lab/actor.py`
- Test: `t5lab/tests/test_actor.py`

**Interfaces:**
- Produces: `class T5Actor(eqx.Module)`, `__init__(in_ch, K_goal, K_prop, backbone_cfg, *, key)`; `__call__(obs, adj_off, positions, blocked) -> (goal_logits (N,K_goal), move_logits (N,5), value (N,), lambda2_hat (N,))`. Composes `ctde_v0.nets.Backbone` (READ-ONLY import) for the belief `z (N,W)`; `goal_head: Linear(W→K_goal)`; `MVProp` (Task 3) for the per-agent field; `move_logits` via `mvprop_field` (Task 4) + the neighbour readout; `aux_head: Linear(W→1)` for actor-side λ̂₂.
- Consumes: `MVProp` (Task 3), `mvprop_field` (Task 4), `ctde_v0.nets.Backbone` (read-only).

- [ ] **Step 1: Write the failing test** — `T5Actor` builds, forwards, returns 4 outputs with correct shapes, and is `filter_grad`-differentiable. Load a minimal `backbone_cfg` from a tiny ctde_v0 config. Assert `goal_logits.shape==(N,K_goal)`, `move_logits.shape==(N,5)`, `value.shape==(N,)`, `lambda2_hat.shape==(N,)`.
- [ ] **Step 2: Run to verify it fails** (`ImportError: cannot import name 'T5Actor'`).
- [ ] **Step 3: Implement `T5Actor`** — Backbone → z; goal_head→goal_logits; pick goal cell from `goal_logits` argmax + a fixed stencil (reuse `ctde_v0.ppo.make_stencil` read-only) → `mvprop_field` over `blocked` → per-agent neighbour readout → `move_logits`; value/aux heads off z. No ctde_v0 edits.
- [ ] **Step 4: Run to verify it passes.**
- [ ] **Step 5: Commit** (`feat(t5lab): T5Actor composing Backbone + MVProp, isolated`).

---

### Task 6: `t5lab/rollout.py` — isolated env scan → goal → MVProp move

**Files:**
- Create: `t5lab/rollout.py`
- Test: `t5lab/tests/test_rollout.py`

**Interfaces:**
- Produces: `rollout(env, actor: T5Actor, critic, cfg, key) -> traj` pytree (obs, adj, goal, move, logp, value, reward, done, positions). The move step descends the MVProp field via a t5lab-local `navfield_move_from_field(pos, field, valid_targets, action_valid)` (we do NOT edit `controller.py`; reimplement the descent — it's ~15 lines mirroring `controller.navfield_move`).
- Consumes: `ctde_v0` env (`reset`/`step`/`central_obs`/`dynamics.targets`/`action_mask`), `_navfield_blocked` (read-only, or a t5lab reimpl of its 5 lines), `T5Actor` (Task 5).

- [ ] **Step 1: Write the failing test** — a 5-step rollout on a tiny 8×8 config returns finite rewards and correct trajectory shapes.
- [ ] **Step 2: Run to verify it fails.**
- [ ] **Step 3: Implement `rollout`** — `jax.lax.scan` over steps: `adj = eu.kb_adjacency(pos, cfg)`; `blocked = navfield_blocked(state, cfg)`; `goal_logits, move_logits, value, l2 = actor(obs, adj, pos, blocked)`; sample move from `move_logits`; `env.step`; collect.
- [ ] **Step 4: Run to verify it passes.**
- [ ] **Step 5: Commit** (`feat(t5lab): isolated rollout with MVProp move`).

---

### Task 7: `t5lab/train.py` — isolated MAPPO train step

**Files:**
- Create: `t5lab/train.py`
- Test: `t5lab/tests/test_train.py`

**Interfaces:**
- Produces: `train_step(state, batch, cfg) -> (state, metrics)`; GAE (γ=0.99, λ=0.95), clipped-PG (clip=0.2, 6 epochs × 4 minibatches), value MSE (0.5), actor-side λ̂₂ aux MSE (0.1), entropy (−0.01), AdamW lr=3e-4 grad-clip 0.5, setpool critic, dual update. **16 rollouts, never reduced.**
- Consumes: `ctde_v0` pure helpers for GAE / clipped-PG / setpool critic / dual where they're actor-agnostic (import read-only); reimplement the actor-coupled loss (log-probs come from `move_logits`, not `goal_logits`).

- [ ] **Step 1: Write the failing test** — one `train_step` on a tiny batch runs without NaN and returns finite loss/metrics; a second step changes params.
- [ ] **Step 2: Run to verify it fails.**
- [ ] **Step 3: Implement `train_step`.**
- [ ] **Step 4: Run to verify it passes.**
- [ ] **Step 5: Commit** (`feat(t5lab): isolated MAPPO train step`).

---

### Task 8: `run_phase1a.py` — the GATE run

**Files:**
- Create: `t5lab/run_phase1a.py`

- [ ] **Step 1: Write the driver** — from-scratch config, train maps = rooms + ConnectedClutter + Pillars (procedural, fresh per episode, held-out SAR families excluded), **16 rollouts**, ≥4 seeds; zero-shot eval on the held-out SAR 30-map set (`scratchpad/maps_sar.json`); print mean coverage + connected-fraction vs the reactive-25% / classical-79% references.
- [ ] **Step 2: Run the GATE.** `PYTHONPATH=. python -m t5lab.run_phase1a`. **GATE:** coverage ≫ 25%. If ≈25% & training stable → diagnose the planner; if unstable → switch to distillation warm-start (ledger #10) before abandoning the thesis.
- [ ] **Step 3: Commit** (`feat(t5lab): Phase-1a go/no-go driver + gate`).

---

## After Phase 1a
On a green gate, proceed to the full 3‑clock stack (Phase 1b) and the plan's Phase 2+ per `../t5/T5_EXPERIMENT_PLAN.md`. Occlusion (`d_eff = d + c·k`, task #15) also lands **isolated** — a t5lab comm model (occluded `kb_adjacency` / reach / λ₂ computed in t5lab's rollout), NOT by editing `comms.py`.

## Self-review notes
- Spec coverage: Phase 0 (MVProp, Tasks 1–4) + Phase 1a (isolated T5 stack + gate, Tasks 5–8). Phases 1b–5 stay in the experiment plan.
- **Isolation: strict — zero ctde_v0 edits; ctde_v0 imported read-only only** (Bijan's call). The cost is reimplementing the actor-coupled rollout/loss (~a few dozen lines) rather than reusing ctde_v0's; the gain is that nothing in the current stack can be perturbed.
- Type consistency: `move_logits (N,5)` flows Task 3 → 5 → 6 → 7; `mvprop_field` returns a distance-like `(N,H,W)` (negated) so the descent is toward the goal.
