# Fiedler Estimation — Slice 1 (JAX Substrate) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the JAX-side substrate of the Fiedler-value study plus a thin end-to-end slice that produces the first **accuracy-vs-history-length** result for two reference estimators (analytic power-iteration + degree-regression).

**Architecture:** A small package `fidler/` inside `zymera_experiments/FidlerValueEstimation/`. `fiedler.py` computes the ground-truth λ₂ from the zymera comm graph; `features.py` extracts per-agent local features; `datagen.py` rolls out episodes and dumps `(features, adjacency, λ₂)` datasets to `.npz`; `metrics.py` scores predictions; `methods/` holds the two reference estimators; `run_slice.py` ties it together. PyTorch learned methods, the message/size/regularization sweeps, and the balthar harness are a **separate later plan**.

**Tech Stack:** Python · JAX/`jnp` (via the editable-installed `zymera` package) · NumPy · pytest. The study folder gets its own git repo (`zymera_experiments` is not a repo).

**Scope note (features):** Slice 1 uses the **6 instantaneous** node features (degree ×2, neighbor-distance ×2, neighbor-degree ×2). The 2 temporal features (`reach_frac`, `steps_since_new`) from `ARCHITECTURES.md` §0.1 are added during temporal assembly in the later PyTorch plan; the methods here don't need them.

---

### Task 1: Package skeleton + zymera smoke test + git

**Files:**
- Create: `zymera_experiments/FidlerValueEstimation/fidler/__init__.py`
- Create: `zymera_experiments/FidlerValueEstimation/fidler/methods/__init__.py`
- Create: `zymera_experiments/FidlerValueEstimation/.gitignore`
- Create: `zymera_experiments/FidlerValueEstimation/tests/test_smoke.py`

- [ ] **Step 1: Init git + gitignore** (the folder is not yet a repo)

```bash
cd /Users/bijanmehr/Project.Zymera/zymera_experiments/FidlerValueEstimation
git init
printf '%s\n' '__pycache__/' '*.pyc' 'results/' 'data/' '*.npz' '.pytest_cache/' > .gitignore
```

- [ ] **Step 2: Create empty package markers**

```bash
mkdir -p fidler/methods tests
touch fidler/__init__.py fidler/methods/__init__.py
```

- [ ] **Step 3: Write the smoke test**

`tests/test_smoke.py`:
```python
import jax
import jax.numpy as jnp
import zymera


def test_zymera_rollout_exposes_positions_and_comm_graph():
    env = zymera.make("comm-coverage", grid=12, n_agents=4, comm_r=5)
    traj = zymera.rollout(env, zymera.random_policy, n_steps=10, key=jax.random.PRNGKey(0), keep="all")
    pos = traj["world"].body.position          # (T+1, N, 2)
    comm = traj["world"].comm_graph            # (T+1, N, N)
    assert pos.shape == (11, 4, 2)
    assert comm.shape == (11, 4, 4)
    assert pos.dtype == jnp.int32
```

- [ ] **Step 4: Run the test**

Run: `cd /Users/bijanmehr/Project.Zymera/zymera_experiments/FidlerValueEstimation && python -m pytest tests/test_smoke.py -v`
Expected: PASS (confirms the zymera venv is active and the API shapes match). If it fails on `import zymera`, activate the zymera lab venv first.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "chore: scaffold fidler package + zymera smoke test"
```

---

### Task 2: `fiedler.py` — ground-truth λ₂ from the comm graph

**Files:**
- Create: `fidler/fiedler.py`
- Create: `tests/test_fiedler.py`

- [ ] **Step 1: Write the failing test** (uses graphs with known algebraic connectivity)

`tests/test_fiedler.py`:
```python
import numpy as np
import jax.numpy as jnp
from fidler import fiedler


def _complete(n):
    a = np.ones((n, n), dtype=bool); np.fill_diagonal(a, True); return jnp.asarray(a)

def _path(n):
    a = np.eye(n, k=1, dtype=bool) | np.eye(n, k=-1, dtype=bool); np.fill_diagonal(a, True); return jnp.asarray(a)

def _two_components(n):  # two disjoint edges/cliques → disconnected
    a = np.zeros((n, n), dtype=bool); a[0, 1] = a[1, 0] = True; a[2, 3] = a[3, 2] = True
    np.fill_diagonal(a, True); return jnp.asarray(a)


def test_complete_graph_lambda2_is_n():
    assert abs(float(fiedler.true_lambda2(_complete(5))) - 5.0) < 1e-4

def test_path_graph_lambda2_known():
    # P_n algebraic connectivity = 2(1 - cos(pi/n))
    n = 4
    expected = 2 * (1 - np.cos(np.pi / n))
    assert abs(float(fiedler.true_lambda2(_path(n))) - expected) < 1e-4

def test_disconnected_lambda2_zero_and_flag_false():
    a = _two_components(4)
    assert float(fiedler.true_lambda2(a)) < 1e-6
    assert bool(fiedler.connected_flag(a)) is False

def test_connected_flag_true_for_path():
    assert bool(fiedler.connected_flag(_path(4))) is True
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_fiedler.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fidler.fiedler'`.

- [ ] **Step 3: Implement**

`fidler/fiedler.py`:
```python
"""Ground-truth Fiedler value (algebraic connectivity) from a comm adjacency."""
import jax.numpy as jnp
import zymera.metrics as zmetrics

CONNECTED_TAU = 1e-3  # lambda2 > tau  <=>  connected


def potential_adjacency(positions, comm_r):
    """(N,2) int positions -> (N,N) bool potential comm graph (in-range, diag True)."""
    return zmetrics.adjacency(positions, radius=comm_r)


def _laplacian(adj):
    """Combinatorial Laplacian L = D - A with self-loops stripped."""
    a = adj.astype(jnp.float32)
    a = a - jnp.diag(jnp.diag(a))          # strip self-loops
    deg = a.sum(-1)
    return jnp.diag(deg) - a


def true_lambda2(adj):
    """Second-smallest eigenvalue of the Laplacian; clamped at 0."""
    evals = jnp.linalg.eigvalsh(_laplacian(adj))
    return jnp.maximum(evals[1], 0.0)


def connected_flag(adj, tau: float = CONNECTED_TAU):
    """True iff lambda2 > tau."""
    return true_lambda2(adj) > tau
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_fiedler.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: fiedler.py — true lambda2 + connected flag from comm graph"
```

---

### Task 3: `features.py` — per-agent local features

**Files:**
- Create: `fidler/features.py`
- Create: `tests/test_features.py`

- [ ] **Step 1: Write the failing test**

`tests/test_features.py`:
```python
import numpy as np
import jax.numpy as jnp
from fidler import features


def test_node_features_shape_and_degree_column():
    # 3 agents in a line at columns 0,1,2 on the same row; comm_r=1 => chain 0-1-2
    pos = jnp.asarray([[0, 0], [0, 1], [0, 2]], dtype=jnp.int32)
    from fidler.fiedler import potential_adjacency
    adj = potential_adjacency(pos, comm_r=1)
    f = features.node_features(pos, adj, comm_r=1)
    assert f.shape == (3, 6)                    # 6 instantaneous features
    # degree (col 0 = raw degree): endpoints have 1 neighbor, middle has 2
    deg = np.asarray(f[:, 0])
    assert deg.tolist() == [1.0, 2.0, 1.0]

def test_isolated_agent_has_zero_degree_and_safe_stats():
    pos = jnp.asarray([[0, 0], [9, 9]], dtype=jnp.int32)  # far apart, comm_r=1 => no edge
    from fidler.fiedler import potential_adjacency
    adj = potential_adjacency(pos, comm_r=1)
    f = features.node_features(pos, adj, comm_r=1)
    assert f.shape == (2, 6)
    assert np.all(np.isfinite(np.asarray(f)))   # no NaNs from empty-neighbor stats
    assert np.asarray(f[:, 0]).tolist() == [0.0, 0.0]
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_features.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fidler.features'`.

- [ ] **Step 3: Implement**

`fidler/features.py`:
```python
"""Per-agent local features (6 instantaneous dims) from positions + adjacency."""
import jax.numpy as jnp


def node_features(positions, adj, comm_r):
    """positions (N,2), adj (N,N) bool (diag True). Returns (N,6) float32:
    [degree, log1p(degree), mean_nbr_dist, std_nbr_dist, mean_nbr_deg, std_nbr_deg].
    Distances normalized by comm_r; empty-neighbor stats are 0 (safe)."""
    a = adj.astype(jnp.float32)
    a = a - jnp.diag(jnp.diag(a))                      # strip self-loops
    deg = a.sum(-1)                                    # (N,)

    pos = positions.astype(jnp.float32)
    diff = pos[:, None, :] - pos[None, :, :]           # (N,N,2)
    dist = jnp.sqrt((diff ** 2).sum(-1)) / float(comm_r)  # (N,N) normalized
    cnt = jnp.maximum(deg, 1.0)                        # avoid /0

    mean_d = (a * dist).sum(-1) / cnt
    var_d = (a * (dist - mean_d[:, None]) ** 2).sum(-1) / cnt
    mean_deg = (a * deg[None, :]).sum(-1) / cnt
    var_deg = (a * (deg[None, :] - mean_deg[:, None]) ** 2).sum(-1) / cnt

    has = (deg > 0).astype(jnp.float32)                # zero stats when isolated
    return jnp.stack([
        deg,
        jnp.log1p(deg),
        mean_d * has,
        jnp.sqrt(var_d) * has,
        mean_deg * has,
        jnp.sqrt(var_deg) * has,
    ], axis=-1).astype(jnp.float32)
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_features.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: features.py — 6-dim instantaneous local node features"
```

---

### Task 4: `metrics.py` — accuracy scoring

**Files:**
- Create: `fidler/metrics.py`
- Create: `tests/test_metrics.py`

- [ ] **Step 1: Write the failing test**

`tests/test_metrics.py`:
```python
import numpy as np
from fidler import metrics


def test_accuracy_perfect_is_one():
    y = np.array([0.5, 1.0, 2.0])
    assert metrics.accuracy(y, y) == 1.0

def test_accuracy_is_one_minus_median_rel_error():
    true = np.array([1.0, 1.0, 1.0])
    pred = np.array([1.1, 0.9, 1.2])          # rel errs 0.1, 0.1, 0.2 -> median 0.1
    assert abs(metrics.accuracy(pred, true) - 0.9) < 1e-9

def test_within_pct_fraction():
    true = np.array([1.0, 1.0, 1.0, 1.0])
    pred = np.array([1.01, 1.04, 1.10, 0.96])  # within 5%: T,T,F,T -> 0.75
    assert metrics.within_pct(pred, true, pct=0.05) == 0.75

def test_r2_perfect():
    y = np.array([0.1, 0.5, 0.9])
    assert abs(metrics.r2(y, y) - 1.0) < 1e-9

def test_connected_accuracy():
    pred_flag = np.array([True, True, False, True])
    true_flag = np.array([True, False, False, True])
    assert metrics.connected_accuracy(pred_flag, true_flag) == 0.75
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fidler.metrics'`.

- [ ] **Step 3: Implement**

`fidler/metrics.py`:
```python
"""Scoring for Fiedler-value predictions (numpy, eval-side)."""
import numpy as np

EPS = 1e-6


def accuracy(pred, true):
    """1 - median relative error (clamped to [0,1])."""
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    rel = np.abs(pred - true) / np.maximum(np.abs(true), EPS)
    return float(np.clip(1.0 - np.median(rel), 0.0, 1.0))


def within_pct(pred, true, pct=0.05):
    """Fraction of predictions within `pct` relative error."""
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    rel = np.abs(pred - true) / np.maximum(np.abs(true), EPS)
    return float(np.mean(rel <= pct))


def r2(pred, true):
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - true.mean()) ** 2) + EPS
    return float(1.0 - ss_res / ss_tot)


def connected_accuracy(pred_flag, true_flag):
    return float(np.mean(np.asarray(pred_flag, bool) == np.asarray(true_flag, bool)))
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_metrics.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: metrics.py — accuracy / within-pct / r2 / connected-acc"
```

---

### Task 5: `config.py` + `datagen.py` — dataset generation to `.npz`

**Files:**
- Create: `fidler/config.py`
- Create: `fidler/datagen.py`
- Create: `tests/test_datagen.py`

- [ ] **Step 1: Write the failing test**

`tests/test_datagen.py`:
```python
import numpy as np
from fidler.config import DataCfg
from fidler import datagen


def test_generate_dataset_shapes_and_labels():
    cfg = DataCfg(n_agents=4, grid=12, comm_r=5, n_episodes=2, n_steps=8, seed=0)
    ds = datagen.generate_dataset(cfg)
    T = cfg.n_steps + 1
    assert ds["features"].shape == (cfg.n_episodes, T, 4, 6)
    assert ds["adjacency"].shape == (cfg.n_episodes, T, 4, 4)
    assert ds["lambda2"].shape == (cfg.n_episodes, T)
    assert np.all(ds["lambda2"] >= 0.0)                 # lambda2 is non-negative
    assert ds["lambda2"].dtype == np.float32

def test_save_and_load_roundtrip(tmp_path):
    cfg = DataCfg(n_agents=4, grid=12, comm_r=5, n_episodes=1, n_steps=4, seed=1)
    ds = datagen.generate_dataset(cfg)
    p = tmp_path / "ds.npz"
    datagen.save_npz(str(p), ds)
    back = np.load(str(p))
    assert np.allclose(back["lambda2"], ds["lambda2"])
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_datagen.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fidler.config'`.

- [ ] **Step 3: Implement config**

`fidler/config.py`:
```python
from dataclasses import dataclass

@dataclass(frozen=True)
class DataCfg:
    n_agents: int = 4
    grid: int = 16
    comm_r: int = 5
    n_obstacles: int = 0
    spawn_radius: int = 2
    n_episodes: int = 8
    n_steps: int = 100
    seed: int = 0
```

- [ ] **Step 4: Implement datagen**

`fidler/datagen.py`:
```python
"""Roll out episodes -> per-step features, adjacency, and true lambda2 -> .npz."""
import jax
import jax.numpy as jnp
import numpy as np
import zymera

from .config import DataCfg
from . import fiedler, features


def generate_dataset(cfg: DataCfg) -> dict:
    env = zymera.make("comm-coverage", grid=cfg.grid, n_agents=cfg.n_agents,
                      comm_r=cfg.comm_r, n_obstacles=cfg.n_obstacles, spawn_radius=cfg.spawn_radius)

    def one_episode(key):
        traj = zymera.rollout(env, zymera.random_policy, n_steps=cfg.n_steps, key=key, keep="all")
        pos = traj["world"].body.position                      # (T+1, N, 2)

        def per_step(p):
            adj = fiedler.potential_adjacency(p, cfg.comm_r)    # (N,N) bool
            return (features.node_features(p, adj, cfg.comm_r), adj, fiedler.true_lambda2(adj))

        feats, adj, lam = jax.vmap(per_step)(pos)               # (T+1,N,6),(T+1,N,N),(T+1,)
        return feats, adj, lam

    keys = jax.random.split(jax.random.PRNGKey(cfg.seed), cfg.n_episodes)
    feats, adj, lam = jax.vmap(one_episode)(keys)               # leading (E, T+1, ...)
    return {
        "features": np.asarray(feats, np.float32),
        "adjacency": np.asarray(adj, bool),
        "lambda2": np.asarray(lam, np.float32),
        "n_agents": np.int32(cfg.n_agents),
        "comm_r": np.int32(cfg.comm_r),
    }


def save_npz(path: str, ds: dict) -> None:
    np.savez_compressed(path, **ds)
```

- [ ] **Step 5: Run to verify it passes**

Run: `python -m pytest tests/test_datagen.py -v`
Expected: PASS (2 tests). (First run JIT-compiles; allow a few seconds.)

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat: datagen.py + config.py — rollout -> (features, adj, lambda2) -> npz"
```

---

### Task 6: `methods/power_iteration.py` — analytic decentralized estimator

**Files:**
- Create: `fidler/methods/power_iteration.py`
- Create: `tests/test_power_iteration.py`

- [ ] **Step 1: Write the failing test** (converges toward true λ₂ on a static graph)

`tests/test_power_iteration.py`:
```python
import numpy as np
import jax.numpy as jnp
from fidler.methods import power_iteration as pi
from fidler import fiedler


def _path(n):
    a = np.eye(n, k=1, dtype=bool) | np.eye(n, k=-1, dtype=bool); np.fill_diagonal(a, True)
    return jnp.asarray(a)


def test_converges_to_true_lambda2_on_static_path():
    adj = _path(5)
    true = float(fiedler.true_lambda2(adj))
    est = float(pi.estimate(adj, n_rounds=400, eps=0.1, seed=0))
    assert abs(est - true) / true < 0.1

def test_error_decreases_with_more_rounds():
    adj = _path(5)
    true = float(fiedler.true_lambda2(adj))
    e_few = abs(float(pi.estimate(adj, n_rounds=20, eps=0.1, seed=0)) - true)
    e_many = abs(float(pi.estimate(adj, n_rounds=400, eps=0.1, seed=0)) - true)
    assert e_many < e_few
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_power_iteration.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fidler.methods.power_iteration'`.

- [ ] **Step 3: Implement**

`fidler/methods/power_iteration.py`:
```python
"""Decentralized power-iteration estimate of lambda2 (Yang-style: deflate + diffuse + Rayleigh)."""
import jax
import jax.numpy as jnp
from fidler.fiedler import _laplacian


def estimate(adj, n_rounds: int, eps: float = 0.1, seed: int = 0):
    """adj (N,N) bool. Returns scalar lambda2 estimate after n_rounds local rounds."""
    lap = _laplacian(adj)                                   # (N,N)
    x = jax.random.normal(jax.random.PRNGKey(seed), (adj.shape[0],))

    def body(x, _):
        x = x - jnp.mean(x)                                 # deflate constant (consensus mean)
        x = x - eps * (lap @ x)                             # power-iter on (I - eps L)
        x = x / (jnp.linalg.norm(x) + 1e-12)
        return x, None

    x, _ = jax.lax.scan(body, x, None, length=n_rounds)
    return (x @ lap @ x) / (x @ x + 1e-12)                  # Rayleigh quotient
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_power_iteration.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: methods/power_iteration.py — decentralized analytic lambda2"
```

---

### Task 7: `methods/degree_regression.py` — degree→λ₂ regression floor

**Files:**
- Create: `fidler/methods/degree_regression.py`
- Create: `tests/test_degree_regression.py`

- [ ] **Step 1: Write the failing test**

`tests/test_degree_regression.py`:
```python
import numpy as np
from fidler.methods import degree_regression as dr


def test_fit_predict_recovers_monotone_relationship():
    rng = np.random.default_rng(0)
    deg = rng.uniform(1, 8, size=400)
    lam = 0.3 * deg + 0.05 * deg ** 2          # synthetic monotone target
    model = dr.fit(deg, lam, degree=2, ridge=1e-6)
    pred = dr.predict(model, deg)
    # explains most variance
    ss_res = np.sum((lam - pred) ** 2); ss_tot = np.sum((lam - lam.mean()) ** 2)
    assert 1 - ss_res / ss_tot > 0.98

def test_predict_shape():
    model = dr.fit(np.array([1.0, 2.0, 3.0]), np.array([0.3, 0.7, 1.2]), degree=1, ridge=1e-6)
    assert dr.predict(model, np.array([1.5, 2.5])).shape == (2,)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_degree_regression.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fidler.methods.degree_regression'`.

- [ ] **Step 3: Implement**

`fidler/methods/degree_regression.py`:
```python
"""Reference floor: ridge polynomial regression of degree-features -> lambda2 (numpy)."""
import numpy as np


def _design(deg, degree):
    deg = np.asarray(deg, float).reshape(-1)
    return np.stack([deg ** k for k in range(degree + 1)], axis=1)   # (M, degree+1)


def fit(deg, lam, degree: int = 2, ridge: float = 1e-6):
    X = _design(deg, degree)
    A = X.T @ X + ridge * np.eye(X.shape[1])
    w = np.linalg.solve(A, X.T @ np.asarray(lam, float).reshape(-1))
    return {"w": w, "degree": degree}


def predict(model, deg):
    return _design(deg, model["degree"]) @ model["w"]
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_degree_regression.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: methods/degree_regression.py — degree->lambda2 ridge floor"
```

---

### Task 8: `run_slice.py` — first accuracy-vs-H result

**Files:**
- Create: `run_slice.py`
- Create: `tests/test_run_slice.py`

**Note on `H` for the two reference methods:** power-iteration uses `H` as **rounds** on the *current* graph (per-step adjacency); degree-regression is instantaneous, so it is evaluated once and reported flat across `H` as the zero-history floor. This establishes the harness; temporal-window `H` for learned methods arrives in the PyTorch plan.

- [ ] **Step 1: Write the failing test**

`tests/test_run_slice.py`:
```python
import json
from run_slice import run_slice


def test_run_slice_produces_accuracy_curve(tmp_path):
    out = tmp_path / "acc.json"
    res = run_slice(n_agents_list=(4, 8), H_list=(1, 3, 5), n_episodes=2, n_steps=8, out_path=str(out))
    assert out.exists()
    saved = json.loads(out.read_text())
    # one accuracy per method per H
    assert set(saved.keys()) == {"power_iteration", "degree_regression"}
    assert len(saved["power_iteration"]) == 3            # one per H
    assert all(0.0 <= a <= 1.0 for a in saved["power_iteration"])
    assert res["power_iteration"][-1] >= res["power_iteration"][0]  # more rounds -> not worse
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_run_slice.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'run_slice'`.

- [ ] **Step 3: Implement**

`run_slice.py`:
```python
"""Slice-1 end-to-end: generate small datasets, run the two reference estimators across H,
report accuracy-vs-H. Run: python run_slice.py"""
import json
import numpy as np
import jax

from fidler.config import DataCfg
from fidler import datagen, metrics
from fidler.methods import power_iteration as pi
from fidler.methods import degree_regression as dr


def run_slice(n_agents_list=(4, 8), H_list=(1, 2, 3, 5), n_episodes=4, n_steps=50, out_path="results/accuracy_vs_h.json"):
    # pool datasets across team sizes
    feats, adjs, lams = [], [], []
    for n in n_agents_list:
        ds = datagen.generate_dataset(DataCfg(n_agents=n, grid=16, comm_r=5, n_episodes=n_episodes, n_steps=n_steps, seed=n))
        feats.append(ds["features"].reshape(-1, n, 6))
        adjs.append(ds["adjacency"].reshape(-1, n, n))
        lams.append(ds["lambda2"].reshape(-1))

    # --- degree-regression: fit on per-agent mean degree (feature col 0), predict per-graph lambda2 via mean ---
    deg_X = np.concatenate([f[:, :, 0].mean(1) for f in feats])     # mean degree per graph
    lam_y = np.concatenate(lams)
    dr_model = dr.fit(deg_X, lam_y, degree=2, ridge=1e-4)
    dr_pred = dr.predict(dr_model, deg_X)
    dr_acc = metrics.accuracy(dr_pred, lam_y)

    # --- power-iteration: estimate per graph for each H (rounds) ---
    pi_acc = {}
    for H in H_list:
        preds, trues = [], []
        for a_group, l_group in zip(adjs, lams):
            for g in range(a_group.shape[0]):
                preds.append(float(pi.estimate(np.asarray(a_group[g]), n_rounds=int(H) * 20, eps=0.1, seed=0)))
                trues.append(float(l_group[g]))
        pi_acc[H] = metrics.accuracy(np.array(preds), np.array(trues))

    result = {
        "power_iteration": [pi_acc[H] for H in H_list],
        "degree_regression": [dr_acc for _ in H_list],   # flat floor
    }
    import os; os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2)
    return result


if __name__ == "__main__":
    print(json.dumps(run_slice(), indent=2))
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_run_slice.py -v`
Expected: PASS.

- [ ] **Step 5: Run the slice for real + eyeball**

Run: `python run_slice.py`
Expected: prints a JSON dict; `power_iteration` accuracy should rise with `H` and approach `degree_regression` floor or beat it. Writes `results/accuracy_vs_h.json`.

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat: run_slice.py — first accuracy-vs-H result for the two reference methods"
```

---

## Done criteria
- `python -m pytest tests/ -v` → all green (smoke + fiedler + features + metrics + datagen + power_iteration + degree_regression + run_slice).
- `python run_slice.py` writes `results/accuracy_vs_h.json` with a real accuracy-vs-H curve at N∈{4,8}.
- The pipeline **rollout → comm graph → true λ₂ → features → estimator → accuracy** is proven end-to-end, ready for the PyTorch learned methods + sweeps + balthar harness (next plan).
