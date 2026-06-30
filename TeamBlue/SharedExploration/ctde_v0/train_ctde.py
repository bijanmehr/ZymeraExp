"""Entry point: train the grounded CTDE v0 agent (LPAC + goal-pointer + λ̂₂).

Short CPU validation (16x16 / 4 agents):
    JAX_PLATFORMS=cpu \
    /Users/bijanmehr/Project.Zymera/zymera_lab/.venv/bin/python \
        -m ctde_v0.train_ctde --iters 20 --rollouts 8

Longer run (with a checkpoint + saved config sidecar):
    JAX_PLATFORMS=cpu \
    /Users/bijanmehr/Project.Zymera/zymera_lab/.venv/bin/python \
        -m ctde_v0.train_ctde --iters 200 --rollouts 16 \
        --grid 16 --n-agents 4 --agg max --mechanism action_mask \
        --run-dir runs/ctde_v0_16x16x4

Per iter it logs: episode reward, coverage %, connectivity %, mean λ₂, and the
aux-λ₂ accuracy (1 - median rel-err of λ̂₂ vs true λ₂), plus the controller
valid-move fraction (should be 100%). ``--run-dir`` writes ``config.json``,
``history.json`` and (if asked) a checkpoint next to the logs.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys

# Allow `python ctde_v0/train_ctde.py` (script) as well as `-m ctde_v0.train_ctde`.
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from ctde_v0 import config as _cfgmod  # type: ignore
    from ctde_v0 import env_utils, ppo  # type: ignore
    from ctde_v0.config import (  # type: ignore
        Backbone, CTDEConfig, Loss, MissionSafety, Regularization, Reward,
        Trainer, World,
    )
else:
    from . import config as _cfgmod
    from . import env_utils, ppo
    from .config import (
        Backbone, CTDEConfig, Loss, MissionSafety, Regularization, Reward,
        Trainer, World,
    )


def _parse_args(argv=None) -> tuple[CTDEConfig, str | None, bool, str | None]:
    p = argparse.ArgumentParser(description="grounded CTDE v0 (LPAC + goal head + aux-λ₂)")
    # world
    p.add_argument("--grid", type=int, default=16)
    p.add_argument("--n-agents", type=int, default=4)
    p.add_argument("--comm-r", type=int, default=5)
    p.add_argument("--horizon", type=int, default=100)
    # backbone / KB
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--mp-rounds", type=int, default=2)
    p.add_argument("--agg", choices=["mean", "max", "multihead"], default="max")
    p.add_argument("--norm", choices=["layer", "none"], default="layer")
    p.add_argument("--message-content",
                   choices=["learned", "edge_distance", "index"], default="learned",
                   help="GNN message-design dial (I2): what each agent puts in its comm "
                        "message beyond the learned feature transform. learned=msg(feats) "
                        "(v0, byte-unchanged); edge_distance=append the comm_r-normalized "
                        "sender->receiver distance (receiver knows how far each neighbour "
                        "is); index=append a fixed sinusoidal embedding of the sender's "
                        "normalized index (receiver tells neighbours apart). Scale- and "
                        "agent-count-invariant.")
    p.add_argument("--recurrence", choices=["feedforward", "recurrent"],
                   default="feedforward",
                   help="per-agent temporal memory (the recurrence axis): "
                        "feedforward=heads read the per-step belief (v0, byte-unchanged); "
                        "recurrent=a GRU carries a per-agent hidden state across the "
                        "100-step episode so the agent remembers its trajectory/coverage "
                        "history (heads read the hidden). Threads through the rollout "
                        "scan AND the PPO loss (BPTT, minibatched over episodes).")
    # action head
    p.add_argument("--goal-k", type=int, default=9)
    p.add_argument("--stride", type=int, default=3)
    p.add_argument("--explorer-tool", choices=["goal_head", "frontier_attn"],
                   default="goal_head",
                   help="how the explorer picks its goal sector (I2 / L4 'disperse'): "
                        "goal_head=belief-only (v0); frontier_attn=learned attention "
                        "biasing the goal toward the most uncovered compass sector")
    p.add_argument("--relay-tool", choices=["lambda2_anchor", "hold"],
                   default="lambda2_anchor",
                   help="which relay controller the relay role calls (I2 relay-tool "
                        "axis): lambda2_anchor=active local-λ̂₂ climb (v0); "
                        "hold=static beacon (STAY unless about to isolate). Only the "
                        "relay role uses it (inert under --role-picker off)")
    p.add_argument("--compass", choices=["off", "on"], default="off",
                   help="append a scale-invariant directional feature to the belief z "
                        "before the heads (I2 compass): off=z unchanged (v0); "
                        "on=add gather (toward in-range teammates) + explore (toward "
                        "nearest uncovered cell) soft-sector directions")
    # mechanism / aux
    p.add_argument("--mechanism",
                   choices=["action_mask", "soft_lambda", "lagrangian",
                            "pid_lagrangian"],
                   default="action_mask",
                   help="connectivity mechanism (I1b adds the two adaptive duals)")
    p.add_argument("--collision-mask", choices=["off", "on"], default="off",
                   help="on=hard collision mask (never step onto an occupied cell)")
    p.add_argument("--conn-signal",
                   choices=["global_lambda2", "local_edge_margin"],
                   default="global_lambda2",
                   help="connectivity SIGNAL source for the penalty mechanisms "
                        "(I1c; orthogonal to --mechanism): global broadcast λ₂ floor "
                        "vs per-agent edge-margin. action_mask ignores it.")
    p.add_argument("--degree-target", type=float, default=1.0,
                   help="per-agent soft-degree floor for --conn-signal local_edge_margin")
    p.add_argument("--lambda-lr", type=float, default=0.05,
                   help="dual-ascent step size (lagrangian mechanism)")
    p.add_argument("--constraint-threshold", type=float, default=None,
                   help="connectivity floor τ for the dual violation "
                        "(default: reuse connectivity.threshold)")
    p.add_argument("--aux-loss", choices=["mse", "huber"], default="mse")
    p.add_argument("--beta", type=float, default=0.1, help="aux λ₂ loss weight")
    # Increment-1: role picker + anti-overlap reward
    p.add_argument("--role-picker", choices=["off", "expl_relay"], default="off",
                   help="off=homogeneous goal head (v0); expl_relay=learned role head")
    p.add_argument("--anti-overlap", choices=["off", "on"], default="off",
                   help="on=subtract same_step_overlap from the composed reward")
    p.add_argument("--anti-overlap-weight", type=float, default=1.0)
    # connectivity-FLOOR barrier ("Hyper-Singularity"): a capped per-agent wall at the
    # disconnection edge, COMPOSES with --conn-signal / --mechanism (not a replacement).
    p.add_argument("--barrier-weight", type=float, default=0.0,
                   help="k; 0 (default)=barrier OFF / reward byte-unchanged")
    p.add_argument("--barrier-a", type=float, default=None,
                   help="barrier launch point a (0 below it); default=comm_r*0.6")
    p.add_argument("--barrier-M", type=float, default=None,
                   help="barrier wall / break range M; default=comm_r")
    p.add_argument("--barrier-p", type=float, default=2.0,
                   help="barrier explosion power on (M - x)")
    p.add_argument("--barrier-cap", type=float, default=50.0,
                   help="barrier finite 'almost infinity' ceiling")
    # reward weights (coverage vs connectivity balance — the scale-huddle suspect:
    # defaults weight connectivity 2x coverage, which a huddle collects for free)
    p.add_argument("--w-coverage", type=float, default=1.0,
                   help="reward weight on the coverage term (default 1.0)")
    p.add_argument("--w-connectivity", type=float, default=2.0,
                   help="reward weight on the connectivity term (default 2.0)")
    p.add_argument("--soft-lambda-penalty", type=float, default=1.0,
                   help="soft_lambda connectivity-penalty scale — the cov<->conn FRONTIER dial: "
                        "0 = no soft penalty (pure coverage), higher = connectivity-dominant. "
                        "Only active when --mechanism soft_lambda.")
    # trainer
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--rollouts", type=int, default=8, help="episodes per iteration")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--ppo-epochs", type=int, default=4)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--minibatches", type=int, default=4)
    # regularization
    p.add_argument("--degree-reg", type=float, default=1e-3)
    p.add_argument("--entropy-coef", type=float, default=0.01)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.0)
    # run control
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-dir", type=str, default=None,
                   help="dir to save config.json + history.json (+ checkpoint)")
    p.add_argument("--ckpt", action="store_true", help="save a model checkpoint in run-dir")
    p.add_argument("--init-from", type=str, default=None,
                   help="warm-start: path to a previously-saved model.eqx (a prior "
                        "run's --ckpt actor+critic) to INITIALIZE training from "
                        "instead of random init (the scale-strategy / warm-start "
                        "ladder). The LPAC backbone is scale-invariant, so a model "
                        "trained @16²/4 loads into a @32²/10 run unchanged (same "
                        "in-channels/width/K/n_roles -> same param shapes; only "
                        "--grid/--n-agents/--comm-r differ). The optimizer state is "
                        "NOT carried — the policy is warm-started, the optimizer is "
                        "re-initialised fresh on the loaded params. Default None = "
                        "random init (byte-unchanged).")
    p.add_argument("--warmstart-noise", type=float, default=0.0,
                   help="sigma>=0; when >0 WITH --init-from, perturb the loaded actor by "
                        "sigma*RMS(leaf)*N(0,1) to break symmetry on transfer (Arm A "
                        "curriculum 'add randomness then scale'). 0=warm-start unchanged.")
    p.add_argument("--critic-mode", choices=["central", "decentral"], default="central",
                   help="central (default)=CTDE centralized critic sources the value "
                        "(byte-unchanged); decentral=DTE tail, value comes from the "
                        "actor's own per-agent value head (central critic unused).")
    p.add_argument("--diversity-residual", choices=["off", "on"], default="off",
                   help="off (default)=v0 goal logits (byte-unchanged); on=Arm B-dico, "
                        "add a mean-zero per-agent identity-conditioned residual to the "
                        "goal logits (controlled behavioural diversity without forking).")
    p.add_argument("--fork-groups", type=int, default=1,
                   help="1 (default)=single shared actor (byte-unchanged); >1=Arm B-fork, "
                        "that many independent sub-actors over a fixed team partition "
                        "(2=explorer/relay halves). Warm-start replicates a single "
                        "bootstrap into the groups. Requires --critic-mode central.")
    # skills + selector (the two-level cognition)
    p.add_argument("--selector", choices=["off", "on"], default="off",
                   help="off (default)=v0 goal/role head (byte-unchanged); on=the learned "
                        "selector over the skill library {disperse,flock,hold} (hierarchical "
                        "skill+offset policy).")
    p.add_argument("--flock", choices=["scripted", "learned"], default="scripted",
                   help="the flock (connectivity-repair) skill flavor the selector calls: "
                        "scripted (weakest-link repair, no params) | learned (a head off z). "
                        "Only used with --selector on.")
    p.add_argument("--congestion", choices=["off", "on"], default="off",
                   help="off (default)=no anti-collapse price; on=the free-market congestion "
                        "penalty (choosing a skill in-range neighbours also chose costs you).")
    p.add_argument("--congestion-weight", type=float, default=0.5,
                   help="weight on the same-skill-crowding penalty (--congestion on)")
    # obstacle terrain + exploration incentives (the obstacle-world reruns)
    p.add_argument("--terrain",
                   choices=["open", "rooms", "walls", "clutter", "pillars", "mixed", "crowded_mix"],
                   default="open",
                   help="open | rooms | walls (RandomWalls) | CONNECTIVITY-SAFE crowded set: "
                        "clutter (ConnectedClutter via --n-obstacles) | pillars (lattice via "
                        "--pillar-spacing/--pillar-size) | mixed (rooms+clutter) | crowded_mix "
                        "(per-reset draw over all three — the training distribution)")
    p.add_argument("--rooms", type=int, default=3, help="rooms when --terrain rooms/mixed/crowded_mix")
    p.add_argument("--n-obstacles", type=int, default=0,
                   help="obstacle count for --terrain walls/clutter/mixed/crowded_mix")
    p.add_argument("--cover-r", type=int, default=0,
                   help="coverage footprint radius (0=point/own-cell; 1='cover what you sense', 3x3). "
                        "cover_r=0 caps the god-view oracle ~72%% cov @32-sq/100-steps; cover_r=1 -> ~100%%.")
    p.add_argument("--pillar-spacing", type=int, default=4, help="lattice period (--terrain pillars)")
    p.add_argument("--pillar-size", type=int, default=2, help="block size (--terrain pillars)")
    p.add_argument("--sense-walls", action=argparse.BooleanOptionalAction, default=True,
                   help="SLAM-style wall perception: fold walls sensed within sense_r into the "
                        "shared belief (known_walls carries signal; walls stop reading as "
                        "frontier). Default on; --no-sense-walls = old wall-blind baseline.")
    p.add_argument("--sense-free", action=argparse.BooleanOptionalAction, default=False,
                   help="occupancy belief: fold the FULL sensed region (free+walls within "
                        "sense_r) into the shared belief, adding the occ_frontier channel (a "
                        "true Yamauchi frontier; the egocentric local_frontier collapses under "
                        "it). Default off (= plain sense_walls).")
    p.add_argument("--boundary", action=argparse.BooleanOptionalAction, default=False,
                   help="add the field-edge (boundary ring) obs channel so the position-blind "
                        "policy perceives the mission-field extent. Default off.")
    p.add_argument("--explore-infogain", choices=["off", "on"], default="off",
                   help="on=add a per-agent exploration bonus (count of uncovered cells in "
                        "sensor range). The 'coverage-bump' incentive is just --w-coverage 3.")
    p.add_argument("--info-gain-weight", type=float, default=0.1)
    args = p.parse_args(argv)

    ckpt_path = (os.path.join(args.run_dir, "model.eqx")
                 if (args.run_dir and args.ckpt) else None)
    cfg = CTDEConfig(
        world=World(grid=args.grid, n_agents=args.n_agents, comm_r=args.comm_r,
                    horizon=args.horizon, terrain=args.terrain, rooms=args.rooms,
                    n_obstacles=args.n_obstacles, pillar_spacing=args.pillar_spacing,
                    pillar_size=args.pillar_size, sense_walls=args.sense_walls,
                    sense_free=args.sense_free, boundary=args.boundary,
                    cover_r=args.cover_r),
        backbone=Backbone(width=args.width, depth=args.depth, mp_rounds=args.mp_rounds,
                          agg=args.agg, norm=args.norm,
                          message_content=args.message_content,
                          recurrence=args.recurrence),
        action_head=dataclasses.replace(CTDEConfig().action_head, K=args.goal_k,
                                        stride=args.stride,
                                        explorer_tool=args.explorer_tool,
                                        relay_tool=args.relay_tool,
                                        compass=args.compass),
        mission_safety=MissionSafety(mechanism=args.mechanism,
                                     conn_signal=args.conn_signal,
                                     degree_target=args.degree_target,
                                     lambda_lr=args.lambda_lr,
                                     constraint_threshold=args.constraint_threshold),
        reward=Reward(w_coverage=args.w_coverage, w_connectivity=args.w_connectivity,
                      soft_lambda_penalty=args.soft_lambda_penalty,
                      barrier_weight=args.barrier_weight, barrier_a=args.barrier_a,
                      barrier_M=args.barrier_M, barrier_p=args.barrier_p,
                      barrier_cap=args.barrier_cap),
        loss=Loss(ppo_clip=args.clip, aux_beta=args.beta, aux_loss=args.aux_loss),
        trainer=Trainer(lr=args.lr, clip=args.clip, ppo_epochs=args.ppo_epochs,
                        minibatches=args.minibatches),
        regularization=Regularization(degree_reg=args.degree_reg,
                                      entropy_coef=args.entropy_coef,
                                      weight_decay=args.weight_decay,
                                      dropout=args.dropout),
        role_picker=args.role_picker,
        reward_anti_overlap=args.anti_overlap,
        anti_overlap_weight=args.anti_overlap_weight,
        collision_mask=args.collision_mask,
        warmstart_noise=args.warmstart_noise,
        critic_mode=args.critic_mode,
        diversity_residual=args.diversity_residual,
        fork_groups=args.fork_groups,
        selector=args.selector,
        flock=args.flock,
        congestion=args.congestion,
        congestion_weight=args.congestion_weight,
        explore_infogain=args.explore_infogain,
        info_gain_weight=args.info_gain_weight,
        scale=f"{args.grid}x{args.grid}/{args.n_agents}",
        iters=args.iters, rollouts_per_iter=args.rollouts, seed=args.seed,
        ckpt_path=ckpt_path,
    )
    return cfg, args.run_dir, bool(args.ckpt), args.init_from


def _log(it, logs):
    # λ tail only when an adaptive mechanism is moving the dual (kept off the v0 line).
    dual = (f"  λ={logs['dual_lambda']:.4f}->{logs.get('dual_lambda_next', 0.0):.4f}"
            f" (v={logs.get('dual_violation', 0.0):.3f})"
            if logs.get("dual_lambda_next", 0.0) or logs.get("dual_lambda", 0.0)
            else "")
    print(
        f"[iter {it:3d}] reward={logs['ep_reward']:8.2f}  "
        f"cov={logs['coverage_pct']*100:5.1f}%  "
        f"conn={logs['connectivity_pct']*100:5.1f}%  "
        f"conn.5={logs.get('connectivity_real', 0.0)*100:5.1f}%  "
        f"snd={logs.get('snd', 0.0):.3f}  "
        f"role={logs.get('role_div', 0.0):.3f}  "
        f"red={logs.get('redundancy', 0.0):.2f}  "
        f"top={logs.get('top_agent_share', 0.0)*100:4.1f}%  "
        f"meanλ2={logs['mean_lambda2']:.3f}  "
        f"aux_acc={logs['aux_acc']*100:5.1f}%  "
        f"(med_rel={logs['median_rel_l2']:.3f})  "
        f"expl={logs.get('explorer_frac', 1.0)*100:5.1f}%  "
        f"ctrl_valid={logs['ctrl_valid_frac']*100:5.1f}%{dual}",
        flush=True,
    )


def main(argv=None):
    cfg, run_dir, want_ckpt, init_from = _parse_args(argv)
    print("config:", json.dumps(cfg.to_dict(), indent=2), flush=True)
    if init_from:
        # the scale-strategy / warm-start ladder: load a prior rung's actor+critic.
        print(f"WARM-START: init actor+critic from {init_from} "
              f"(scale-invariant cross-rung load; fresh optimizer + dual) "
              f"-> training {cfg.scale}", flush=True)
    if run_dir:
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump(cfg.to_dict(), f, indent=2)

    env = env_utils.build_env(cfg)
    print("env:", repr(env), flush=True)

    state, history = ppo.train(env, cfg, log_fn=_log, init_from=init_from)

    if len(history) >= 2:
        first, last = history[0], history[-1]
        print(
            f"\naux_loss: {first['aux_loss']:.4f} (iter 0) -> "
            f"{last['aux_loss']:.4f} (iter {len(history)-1})  "
            f"[{'DOWN' if last['aux_loss'] < first['aux_loss'] else 'UP'}]",
            flush=True,
        )
        print(
            f"aux_acc:  {first['aux_acc']*100:.1f}% -> {last['aux_acc']*100:.1f}%  "
            f"(median rel-err {first['median_rel_l2']:.3f} -> {last['median_rel_l2']:.3f})",
            flush=True,
        )
        print(
            f"controller valid-move fraction (final): "
            f"{last['ctrl_valid_frac']*100:.1f}%",
            flush=True,
        )

    if run_dir:
        with open(os.path.join(run_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)
        print(f"saved config.json + history.json -> {run_dir}", flush=True)

    if cfg.ckpt_path:
        # Save the deployable (actor, critic) snapshot + a config-meta sidecar via the
        # local checkpoint module (self-contained, byte-compatible with the Fiedler
        # format). This is EXACTLY what `--init-from` reads back to warm-start the next
        # scale rung: a same-shaped (actor, critic) tuple the loader deserialises into a
        # fresh current-config skeleton.
        if __package__ in (None, ""):
            from ctde_v0 import checkpoint as ckpt  # type: ignore
        else:
            from . import checkpoint as ckpt
        ckpt.save_model(cfg.ckpt_path, (state.actor, state.critic), meta=cfg.to_dict())
        print(f"saved checkpoint -> {cfg.ckpt_path}", flush=True)

    return state, history


if __name__ == "__main__":
    main()
