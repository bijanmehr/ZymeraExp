"""Train the EXACT previous architecture with greedy -> MVProp planner (the deal).

Config is built through ctde_v0's OWN CLI parser (train_ctde._parse_args) for byte-parity with
the proven pipeline, then the ONLY change is ``action_head.controller = 'mvprop'`` and the frozen
distilled planner is injected into ``ppo.train(..., mvplanner=)``. Everything else — the central
setpool critic, the lagrangian dual, local_edge_margin, the reach_fraction reward, the λ̂₂ aux
head, the hard collision-mask, no connectivity hard-mask — is the previous architecture, untouched.

Grading is CONNECTIVITY-FIRST: connectivity_real (λ₂>0.5 connected-fraction, the binding mission
bar) is the lead metric; a run whose graph breaks is a failure regardless of coverage.

Run (GPU):
  PY=$HOME/ZymeraLab/.venv/bin/python
  PYTHONPATH=.:../../../FiedlerValueEstimation GRID=32 NAG=10 ITERS=8000 \
    PLANNER=mvprop_distilled.eqx $PY -m t5lab.run_mvprop
"""
import os
import json
import dataclasses
import jax
import jax.numpy as jnp
import numpy as np

from ctde_v0 import train_ctde, ppo
from ctde_v0 import env_utils as eu
from ctde_v0 import checkpoint as ckpt
from t5lab.planner import load_planner

RUN_DIR = os.environ.get("RUN_DIR", "")

GRID = int(os.environ.get("GRID", 32))
NAG = int(os.environ.get("NAG", 10))
COMMR = int(os.environ.get("COMMR", 5))
ITERS = int(os.environ.get("ITERS", 8000))
TERRAIN = os.environ.get("TERRAIN", "open")
NOBST = int(os.environ.get("NOBST", 40))     # obstacle count for clutter/mixed/walls
ROOMS = int(os.environ.get("ROOMS", 3))      # rooms for rooms/mixed
SEED = int(os.environ.get("SEED", 0))
PLANNER = os.environ.get("PLANNER", "mvprop_distilled.eqx")
KPROP = int(os.environ.get("KPROP", 32))
CONTROLLER = os.environ.get("CONTROLLER", "mvprop")   # mvprop | greedy | navfield (the ONE swap axis)
CRITIC = os.environ.get("CRITIC", "setpool")          # central critic arch: setpool | setattn | conv
OCCLUSION = os.environ.get("OCCLUSION", "0") == "1"   # "wall RF": walls inflate effective comm distance
WCONN = os.environ.get("WCONN", "2")                  # connectivity reward weight (reach_fraction)
DEGREE = os.environ.get("DEGREE", "1")                # soft-degree floor for the local_edge_margin dual
BARRIER = os.environ.get("BARRIER", "0")              # connectivity-floor barrier weight (0 = off)
ROLE = os.environ.get("ROLE", "off")                  # role_picker: off | expl_relay (division of labor)
TAG = os.environ.get("TAG", f"{TERRAIN}_{GRID}_{CONTROLLER}")


def build_cfg():
    """The approved-brief architecture: setpool central critic + lagrangian + local_edge_margin
    + goal_head + collision-on + no hard mask, cover_r=0, 7-ch SLAM belief. controller swapped
    to mvprop AFTER parsing (the parser only knows greedy/navfield)."""
    argv = [
        "--grid", str(GRID), "--n-agents", str(NAG), "--comm-r", str(COMMR), "--horizon", "100",
        "--terrain", TERRAIN, "--cover-r", "0",
        "--n-obstacles", str(NOBST), "--rooms", str(ROOMS),
        "--sense-walls", "--sense-free", "--boundary",
        "--critic-arch", CRITIC,
        "--mechanism", "lagrangian", "--conn-signal", "local_edge_margin",
        "--collision-mask", "on",
        "--explorer-tool", "goal_head", "--role-picker", ROLE,
        "--w-coverage", "3", "--w-connectivity", WCONN,
        "--degree-target", DEGREE, "--barrier-weight", BARRIER,
        "--iters", str(ITERS), "--rollouts", "16", "--seed", str(SEED),
    ]
    cfg, _rd, _ck, _init = train_ctde._parse_args(argv)
    # THE swap axis — the ONLY change vs the previous architecture (greedy baseline vs planner).
    cfg = dataclasses.replace(cfg, action_head=dataclasses.replace(cfg.action_head,
                                                                   controller=CONTROLLER))
    if OCCLUSION:                                          # "wall RF": walls block comms (d_eff = d + c*k)
        cfg = dataclasses.replace(cfg, world=dataclasses.replace(cfg.world, occlusion=True))
    return cfg


def evaluate(env, state, cfg, planner, key, rollouts=32):
    """Fresh eval rollout -> connectivity-first metrics."""
    cfg_eval = dataclasses.replace(cfg, rollouts_per_iter=rollouts)
    stencil = ppo.make_stencil(cfg_eval)
    traj = ppo.collect(env, state.actor, state.critic, cfg_eval, stencil, key,
                       state.dual.lam, planner)
    cov = float(traj["coverage"][:, -1].mean())
    return {
        "coverage_pct": 100 * cov,
        "connectivity_real": None,   # filled from training logs (per-step grader)
        "n_rollouts": rollouts,
    }


def main():
    key = jax.random.PRNGKey(SEED)
    kt, ke = jax.random.split(key)
    cfg = build_cfg()
    env = eu.build_env(cfg)
    # planner only needed for the mvprop controller; greedy/navfield baselines run with None.
    planner = load_planner(PLANNER, in_ch=2, K=KPROP, gamma=0.9) if CONTROLLER == "mvprop" else None
    print(f"=== run_mvprop [{TAG}] grid={GRID} N={NAG} iters={ITERS} controller={cfg.action_head.controller} "
          f"critic={cfg.critic_mode}/{cfg.critic_arch} mech={cfg.mission_safety.mechanism}/"
          f"{cfg.mission_safety.conn_signal} planner={PLANNER} ===", flush=True)

    hist_keys_printed = [False]

    def logf(it, logs):
        if not hist_keys_printed[0]:
            print(f"[logkeys] {sorted(logs.keys())}", flush=True)
            hist_keys_printed[0] = True
        if it % 50 == 0 or it == ITERS - 1:
            cov = logs.get("coverage_pct", float("nan"))
            creal = logs.get("connectivity_real", float("nan"))
            cpct = logs.get("connectivity_pct", float("nan"))
            rew = logs.get("ep_reward", logs.get("ret", float("nan")))
            ent = logs.get("entropy", float("nan"))
            dl = logs.get("dual_lambda", float("nan"))
            # CONNECTIVITY FIRST in the log line.
            print(f"[it {it:5d}] CONN_real={creal:.3f} conn={cpct:.3f} | cov={cov:.3f} "
                  f"rew={rew:.1f} ent={ent:.2f} dualλ={dl:.3f}", flush=True)

    state, hist = ppo.train(env, cfg, key=kt, log_fn=logf, mvplanner=planner)

    # save the trained (actor, critic) + config so render_mvprop / eval can reload (the planner
    # lives at PLANNER, referenced in the run meta). Mirrors train_ctde --ckpt on-disk format.
    run_dir = RUN_DIR or f"runs/mvprop/{TAG}_seed{SEED}"
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)
    with open(os.path.join(run_dir, "history.json"), "w") as f:
        json.dump(hist, f, indent=2)
    with open(os.path.join(run_dir, "mvprop_run.json"), "w") as f:
        json.dump({"planner": os.path.abspath(PLANNER), "kprop": KPROP, "tag": TAG}, f, indent=2)
    ckpt.save_model(os.path.join(run_dir, "model.eqx"), (state.actor, state.critic),
                    meta=cfg.to_dict())
    print(f"[saved] checkpoint + config -> {run_dir}", flush=True)

    # final: mean of last 50 iters (connectivity-first) + a fresh eval rollout for coverage.
    tail = hist[-50:] if len(hist) >= 50 else hist
    m = lambda k: float(np.mean([h.get(k, np.nan) for h in tail]))
    ev = evaluate(env, state, cfg, planner, ke)
    print("\n================ CONNECTIVITY-FIRST SUMMARY ================", flush=True)
    print(f"  [{TAG}]  (mean of last {len(tail)} iters)", flush=True)
    print(f"  CONNECTIVITY_real (λ₂>0.5) = {m('connectivity_real'):.3f}   <-- mission validity", flush=True)
    print(f"  connectivity_pct  (λ₂>1e-3)= {m('connectivity_pct'):.3f}", flush=True)
    print(f"  final dual λ               = {float(state.dual.lam):.3f}", flush=True)
    print(f"  coverage_pct (train)       = {m('coverage_pct'):.3f}", flush=True)
    print(f"  coverage_pct (eval, {ev['n_rollouts']} roll)= {ev['coverage_pct']:.1f}%", flush=True)
    verdict = "PASS" if m('connectivity_real') >= 0.90 else "CONNECTIVITY FAILURE"
    print(f"  VERDICT (conn-first): {verdict}", flush=True)
    print("===========================================================", flush=True)
    print("=== RUN_MVPROP DONE ===", flush=True)


if __name__ == "__main__":
    main()
