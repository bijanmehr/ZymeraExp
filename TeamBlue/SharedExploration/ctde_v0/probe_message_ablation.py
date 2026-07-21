"""STANDALONE CPU probe — does the LPAC GNN backbone actually USE neighbour messages?

No training. Loads a run's ``config.json`` + ``model.eqx`` (exactly like ``render.py``),
drives ONE rollout per seed with the base actor, samples states every ``--sample-stride``
steps, and — at each sampled state — ablates the comm graph / a neighbour's belief and
measures how much the per-agent belief ``z`` and the goal distribution move. The 5th
return value of ``actor(obs, adj, dist=dist, h=h, inference=True)`` is ``feat = z``
(``nets.py:874``); that is the vector EVERY head reads, so it is what we watch.

Three modes (``--mode``):

  * ``graphoff`` — set ``adj`` all-False (every agent isolated) and recompute. Report
    per-agent ``rel_dz = ‖z1_i − z0_i‖ / ‖z0_i‖`` and the goal-argmax-flip fraction. THE
    decisive number: ``mean rel_dz ≈ 0`` ⇒ the GNN IGNORES neighbour messages (the belief
    is a pure per-agent CNN readout; message passing is dead weight).

  * ``edge`` — for every PRESENT directed edge ``i←j`` remove just that one edge
    (``adj.at[i,j].set(False)``) and measure the receiver ``i``'s ``rel_dz``, goal-KL and
    argmax-flip → a per-edge influence heatmap (who listens to whom, and how hard).

  * ``perturb`` — inject ``Gaussian(0, sigma)`` into ONE agent's belief channels
    (``known`` + ``*frontier``) and measure Δz / Δgoal at its in-range neighbours → does a
    corrupted local belief propagate through the graph to the agents that hear it.

Outputs (into ``--out`` as a directory): per-mode heatmap PNG(s), histogram PNG(s), and
``summary.json`` with an explicit ``verdict`` string.

    JAX_PLATFORMS=cpu PYTHONPATH=.:../../../FiedlerValueEstimation \
    /Users/bijanmehr/Project.Zymera/zymera_lab/.venv/bin/python -m ctde_v0.probe_message_ablation \
        --run-dir runs/warmab/warm_s0 --out probes/msg_warm_s0 --mode graphoff \
        --seeds 0,1,2 --sample-stride 10
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys

import numpy as np

import jax
import jax.numpy as jnp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from ctde_v0 import env_utils, ppo
    from ctde_v0.config import from_dict
else:
    from . import env_utils, ppo
    from .config import from_dict

EPS = 1e-8


# =============================================================================
# Loader + rollout (mirrors render.py) — collect sampled states
# =============================================================================
def _load(run_dir: str, seed: int, steps: int):
    """Load ``(cfg, env, actor)`` for ``run_dir`` — same path as ``render.render_run``."""
    cfg = from_dict(json.load(open(os.path.join(run_dir, "config.json"))))
    if steps:
        cfg = dataclasses.replace(cfg, world=dataclasses.replace(cfg.world, horizon=steps))
    env = env_utils.build_env(cfg)
    ckpt = os.path.join(run_dir, "model.eqx")
    state = ppo.init_state_from_checkpoint(env, cfg, ckpt, jax.random.PRNGKey(seed))
    return cfg, env, state.actor


def _rollout_snapshots(cfg, env, actor, stencil, seed: int, stride: int):
    """Drive one base-actor rollout; snapshot ``(obs, adj, dist, h, st)`` every ``stride``
    steps BEFORE stepping (the ``h`` is the carry that produced the action at that step)."""
    edge_msg = cfg.backbone.message_content != "learned"
    k = jax.random.PRNGKey(seed + 7)
    rk0, kk = jax.random.split(k)
    obs, st = env.reset(rk0)
    h = actor.init_hidden(st.n_agents)
    snaps = []
    for t in range(cfg.world.horizon):
        kk, ak, rk, sk = jax.random.split(kk, 4)
        adj = env_utils.kb_adjacency(st.body.position, cfg)
        dist = env_utils.kb_distance(st.body.position, cfg) if edge_msg else None
        if t % stride == 0:
            snaps.append(dict(obs=obs, adj=adj, dist=dist, h=h, st=st))
        goal_logits, _r, _v, _l2, _z, h = actor(obs, adj, dist=dist, h=h, inference=True)
        gmask = ppo._goal_mask(env, st, cfg, stencil)
        masked = jnp.where(gmask, goal_logits, ppo._NEG)
        goal = jax.random.categorical(ak, masked, axis=-1)
        move, _ = ppo._goal_to_move(env, st, goal, stencil, None, cfg)
        obs, st, _r2, _d, _i = env.step(st, move, sk)
    return snaps


# =============================================================================
# Forward helpers
# =============================================================================
def _forward(actor, obs, adj, dist, h):
    """Return ``(goal_logits (N,K), z (N,W))`` from the base actor path (5th ret = feat)."""
    goal_logits, _r, _v, _l2, z, _h = actor(obs, adj, dist=dist, h=h, inference=True)
    return goal_logits, z


def _rel_dz(z1, z0):
    """(N,) per-agent ``‖z1_i − z0_i‖ / ‖z0_i‖``."""
    return jnp.linalg.norm(z1 - z0, axis=-1) / (jnp.linalg.norm(z0, axis=-1) + EPS)


def _masked_logp(logits, gmask):
    """log-softmax over goal offsets with the env goal-mask applied (N,K)."""
    return jax.nn.log_softmax(jnp.where(gmask, logits, ppo._NEG), axis=-1)


def _goal_kl(lp0, lp1):
    """(N,) KL(p0 ‖ p1) between the two masked goal distributions."""
    return jnp.sum(jnp.exp(lp0) * (lp0 - lp1), axis=-1)


# =============================================================================
# Mode (a): graph-off  — the decisive "are messages used at all" number
# =============================================================================
def run_graphoff(cfg, env, actor, stencil, snaps):
    rel, flips = [], []
    per_agent_rel = None  # (N,) running mean for a bar chart
    for s in snaps:
        obs, adj, dist, h, st = s["obs"], s["adj"], s["dist"], s["h"], s["st"]
        gmask = ppo._goal_mask(env, st, cfg, stencil)
        g0, z0 = _forward(actor, obs, adj, dist, h)
        g1, z1 = _forward(actor, obs, jnp.zeros_like(adj), dist, h)
        rd = np.asarray(_rel_dz(z1, z0))
        lp0, lp1 = _masked_logp(g0, gmask), _masked_logp(g1, gmask)
        fl = np.asarray(jnp.argmax(lp0, -1) != jnp.argmax(lp1, -1))
        rel.append(rd)
        flips.append(fl)
        per_agent_rel = rd if per_agent_rel is None else per_agent_rel + rd
    rel = np.concatenate(rel)
    flips = np.concatenate(flips)
    stats = _dist_stats(rel, "rel_dz")
    stats["goal_argmax_flip_fraction"] = float(flips.mean())
    stats["n_agent_state_pairs"] = int(rel.size)
    m = stats["rel_dz_mean"]
    if m < 0.02:
        verdict = (f"MESSAGES UNUSED: mean rel_dz={m:.4f} ≈ 0 with the graph fully OFF — "
                   "the belief z is essentially a per-agent CNN readout; the GNN "
                   "message-passing contributes ~nothing and goal flips "
                   f"{stats['goal_argmax_flip_fraction']*100:.1f}% of the time.")
    elif m < 0.10:
        verdict = (f"MESSAGES WEAK: mean rel_dz={m:.4f} (goal flip "
                   f"{stats['goal_argmax_flip_fraction']*100:.1f}%) — neighbour messages "
                   "nudge the belief but the CNN dominates.")
    else:
        verdict = (f"MESSAGES USED: mean rel_dz={m:.4f} (goal flip "
                   f"{stats['goal_argmax_flip_fraction']*100:.1f}%) — turning the graph off "
                   "materially moves the belief and the chosen goal.")
    return stats, verdict, {"rel_dz": rel, "flips": flips,
                            "per_agent_rel_mean": per_agent_rel / max(len(snaps), 1)}


# =============================================================================
# Mode (b): per-edge ablation  — influence heatmap
# =============================================================================
def run_edge(cfg, env, actor, stencil, snaps):
    all_rd, all_kl, all_fl = [], [], []
    heat_rd = heat_kl = None  # first snapshot (N,N) heatmaps (NaN = no edge)
    for s in snaps:
        obs, adj, dist, h, st = s["obs"], s["adj"], s["dist"], s["h"], s["st"]
        n = adj.shape[0]
        gmask = ppo._goal_mask(env, st, cfg, stencil)
        g0, z0 = _forward(actor, obs, adj, dist, h)
        lp0 = _masked_logp(g0, gmask)

        def rm(kk, _adj=adj, _n=n):                      # remove directed edge kk = i*n+j
            i, j = kk // _n, kk % _n
            return _adj.at[i, j].set(False)

        variants = jax.vmap(rm)(jnp.arange(n * n))       # (n*n, N, N)

        def fwd(a, _obs=obs, _dist=dist, _h=h):
            return _forward(actor, _obs, a, _dist, _h)

        gV, zV = jax.vmap(fwd)(variants)                 # (n*n, N, K), (n*n, N, W)
        i_idx = jnp.arange(n * n) // n                   # receiver of each removed edge
        dz = _rel_dz(zV, z0[None])                       # (n*n, N)
        rd_recv = dz[jnp.arange(n * n), i_idx]           # (n*n,) change at the receiver
        lpV = jax.nn.log_softmax(jnp.where(gmask[None], gV, ppo._NEG), axis=-1)
        kl = _goal_kl(lp0[None], lpV)                    # (n*n, N)
        kl_recv = kl[jnp.arange(n * n), i_idx]
        fl = jnp.argmax(lp0[None], -1) != jnp.argmax(lpV, -1)
        fl_recv = np.asarray(fl[jnp.arange(n * n), i_idx])

        present = np.asarray(adj).reshape(-1).astype(bool)
        rd_recv = np.asarray(rd_recv)
        kl_recv = np.asarray(kl_recv)
        all_rd.append(rd_recv[present])
        all_kl.append(kl_recv[present])
        all_fl.append(fl_recv[present])
        if heat_rd is None:
            heat_rd = np.where(present, rd_recv, np.nan).reshape(n, n)
            heat_kl = np.where(present, kl_recv, np.nan).reshape(n, n)
    rd = np.concatenate(all_rd) if all_rd else np.zeros(0)
    kl = np.concatenate(all_kl) if all_kl else np.zeros(0)
    fl = np.concatenate(all_fl) if all_fl else np.zeros(0)
    stats = _dist_stats(rd, "edge_rel_dz")
    stats.update(_dist_stats(kl, "edge_goal_kl"))
    stats["edge_goal_argmax_flip_fraction"] = float(fl.mean()) if fl.size else 0.0
    stats["n_edges"] = int(rd.size)
    mx = stats["edge_rel_dz_max"]
    mn = stats["edge_rel_dz_mean"]
    if rd.size == 0:
        verdict = "NO EDGES present in any sampled state (agents never in comm range)."
    elif mx < 0.02:
        verdict = (f"NO SINGLE EDGE MATTERS: max per-edge rel_dz={mx:.4f} — removing any one "
                   "link barely moves the receiver's belief; the GNN is not edge-sensitive.")
    else:
        verdict = (f"EDGE-SENSITIVE: max per-edge rel_dz={mx:.4f}, mean={mn:.4f}, goal flips "
                   f"{stats['edge_goal_argmax_flip_fraction']*100:.1f}% of edge removals — "
                   "specific links carry belief the receiver uses.")
    return stats, verdict, {"edge_rel_dz": rd, "edge_goal_kl": kl,
                            "heat_rd": heat_rd, "heat_kl": heat_kl}


# =============================================================================
# Mode (c): belief perturbation  — corrupt one agent's belief, watch neighbours
# =============================================================================
def run_perturb(cfg, env, actor, stencil, snaps, sigma, seed):
    ch_names = list(env.obs.channels)
    belief_ch = [i for i, nm in enumerate(ch_names)
                 if nm == "known" or nm.endswith("frontier")]
    if not belief_ch:
        belief_ch = [0]
    belief_ch_arr = jnp.asarray(belief_ch)

    all_rd, all_kl, all_fl, self_rd = [], [], [], []
    heat = None  # first snapshot (N,N) neighbour-effect heatmap [corruptor j, receiver i]
    key = jax.random.PRNGKey(9157 + seed)
    for s in snaps:
        obs, adj, dist, h, st = s["obs"], s["adj"], s["dist"], s["h"], s["st"]
        n = adj.shape[0]
        _c, H, W = obs.shape[1], obs.shape[2], obs.shape[3]
        gmask = ppo._goal_mask(env, st, cfg, stencil)
        g0, z0 = _forward(actor, obs, adj, dist, h)
        lp0 = _masked_logp(g0, gmask)

        key, nk = jax.random.split(key)
        noise = sigma * jax.random.normal(nk, (n, len(belief_ch), H, W))

        def make_obs(j, _obs=obs, _noise=noise, _ch=belief_ch_arr):
            return _obs.at[j, _ch].add(_noise[j])        # corrupt agent j's belief planes

        obsV = jax.vmap(make_obs)(jnp.arange(n))          # (n, N, C, H, W)

        def fwd(o, _adj=adj, _dist=dist, _h=h):
            return _forward(actor, o, _adj, _dist, _h)

        gV, zV = jax.vmap(fwd)(obsV)                      # (n, N, K/W): axis0 = corruptor j
        dz = np.asarray(_rel_dz(zV, z0[None]))            # (n, N) [j, i]
        lpV = jax.nn.log_softmax(jnp.where(gmask[None], gV, ppo._NEG), axis=-1)
        kl = np.asarray(_goal_kl(lp0[None], lpV))         # (n, N) [j, i]
        fl = np.asarray(jnp.argmax(lp0[None], -1) != jnp.argmax(lpV, -1))  # (n, N)

        adj_np = np.asarray(adj)                          # adj[i, j]: i hears j
        nb_heat = np.full((n, n), np.nan)
        for j in range(n):
            self_rd.append(dz[j, j])                      # the corrupted agent's own change
            recv = adj_np[:, j].copy()                    # agents i that hear j
            recv[j] = False
            idx = np.where(recv)[0]
            if idx.size:
                all_rd.append(dz[j, idx])
                all_kl.append(kl[j, idx])
                all_fl.append(fl[j, idx])
                nb_heat[j, idx] = dz[j, idx]
        if heat is None:
            heat = nb_heat
    rd = np.concatenate(all_rd) if all_rd else np.zeros(0)
    kl = np.concatenate(all_kl) if all_kl else np.zeros(0)
    fl = np.concatenate(all_fl) if all_fl else np.zeros(0)
    self_arr = np.asarray(self_rd)
    stats = _dist_stats(rd, "neighbor_rel_dz")
    stats.update(_dist_stats(kl, "neighbor_goal_kl"))
    stats["neighbor_goal_argmax_flip_fraction"] = float(fl.mean()) if fl.size else 0.0
    stats["self_rel_dz_mean"] = float(self_arr.mean()) if self_arr.size else 0.0
    stats["n_neighbor_pairs"] = int(rd.size)
    stats["belief_channels"] = [ch_names[i] for i in belief_ch]
    stats["noise_sigma"] = float(sigma)
    if rd.size == 0:
        verdict = ("NO NEIGHBOUR PAIRS: corruptors had no in-range listeners in any sampled "
                   "state — cannot assess propagation.")
    elif stats["neighbor_rel_dz_mean"] < 0.02:
        verdict = (f"CORRUPTION DOES NOT PROPAGATE: a σ={sigma} hit to one agent's belief "
                   f"moves its neighbours' z by mean {stats['neighbor_rel_dz_mean']:.4f} "
                   f"(self-change {stats['self_rel_dz_mean']:.4f}) — the graph does not carry "
                   "the poisoned belief onward.")
    else:
        verdict = (f"CORRUPTION PROPAGATES: a σ={sigma} hit to one agent's belief shifts its "
                   f"neighbours' z by mean {stats['neighbor_rel_dz_mean']:.4f} (max "
                   f"{stats['neighbor_rel_dz_max']:.4f}), flipping their goal "
                   f"{stats['neighbor_goal_argmax_flip_fraction']*100:.1f}% of the time — a "
                   "single bad belief spreads over the comm graph.")
    return stats, verdict, {"neighbor_rel_dz": rd, "neighbor_goal_kl": kl, "heat": heat}


# =============================================================================
# Small utils: stats, plots
# =============================================================================
def _dist_stats(x, prefix):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return {f"{prefix}_{k}": 0.0 for k in ("mean", "median", "p90", "max", "std")}
    return {
        f"{prefix}_mean": float(np.mean(x)),
        f"{prefix}_median": float(np.median(x)),
        f"{prefix}_p90": float(np.percentile(x, 90)),
        f"{prefix}_max": float(np.max(x)),
        f"{prefix}_std": float(np.std(x)),
    }


def _hist(x, title, xlabel, path):
    x = np.asarray(x)
    fig, ax = plt.subplots(figsize=(6, 4))
    if x.size:
        ax.hist(x, bins=40, color="#3b6ea5", edgecolor="white")
        ax.axvline(float(np.mean(x)), color="#c0392b", ls="--",
                   label=f"mean={np.mean(x):.4f}")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "no data", ha="center", va="center")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _heatmap(mat, title, path, xlabel="sender / corruptor j", ylabel="receiver i"):
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(np.asarray(mat), cmap="viridis", origin="upper")
    fig.colorbar(im, ax=ax, label="rel_dz")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _bar(x, title, xlabel, path):
    x = np.asarray(x)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(np.arange(x.size), x, color="#3b6ea5")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("mean rel_dz")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# =============================================================================
# Driver
# =============================================================================
def run_probe(run_dir, out, mode, seeds, sample_stride, noise_sigma, steps):
    os.makedirs(out, exist_ok=True)
    cfg0 = None
    all_snaps = []
    heat_seed_snaps = None
    for si, seed in enumerate(seeds):
        cfg, env, actor = _load(run_dir, seed, steps)
        stencil = ppo.make_stencil(cfg)
        snaps = _rollout_snapshots(cfg, env, actor, stencil, seed, sample_stride)
        if si == 0:
            cfg0, env0, actor0, stencil0 = cfg, env, actor, stencil
            heat_seed_snaps = snaps
        all_snaps.extend(snaps)
    # analysis runs against seed-0 env/actor/stencil (same architecture across seeds);
    # snapshots carry their own obs/adj/dist/h/st so cross-seed pooling is valid.
    if mode == "graphoff":
        stats, verdict, arts = run_graphoff(cfg0, env0, actor0, stencil0, all_snaps)
        _hist(arts["rel_dz"], "graph-OFF per-agent rel_dz", "rel_dz = ‖z_off − z‖/‖z‖",
              os.path.join(out, "hist_rel_dz.png"))
        _bar(arts["per_agent_rel_mean"], "graph-OFF mean rel_dz per agent (seed 0)",
             "agent index", os.path.join(out, "bar_per_agent_rel_dz.png"))
    elif mode == "edge":
        stats, verdict, arts = run_edge(cfg0, env0, actor0, stencil0, all_snaps)
        _hist(arts["edge_rel_dz"], "per-edge receiver rel_dz", "rel_dz",
              os.path.join(out, "hist_edge_rel_dz.png"))
        _hist(arts["edge_goal_kl"], "per-edge receiver goal-KL", "KL(p0‖p1)",
              os.path.join(out, "hist_edge_goal_kl.png"))
        if arts["heat_rd"] is not None:
            _heatmap(arts["heat_rd"], "per-edge influence rel_dz (1st state)",
                     os.path.join(out, "heat_edge_rel_dz.png"),
                     xlabel="sender j (removed)", ylabel="receiver i")
            _heatmap(arts["heat_kl"], "per-edge goal-KL (1st state)",
                     os.path.join(out, "heat_edge_goal_kl.png"),
                     xlabel="sender j (removed)", ylabel="receiver i")
    elif mode == "perturb":
        stats, verdict, arts = run_perturb(cfg0, env0, actor0, stencil0, all_snaps,
                                           noise_sigma, seeds[0])
        _hist(arts["neighbor_rel_dz"], "neighbour rel_dz under belief perturbation",
              "rel_dz", os.path.join(out, "hist_neighbor_rel_dz.png"))
        _hist(arts["neighbor_goal_kl"], "neighbour goal-KL under belief perturbation",
              "KL(p0‖p1)", os.path.join(out, "hist_neighbor_goal_kl.png"))
        if arts["heat"] is not None:
            _heatmap(arts["heat"], "belief-perturb neighbour Δz (1st state)",
                     os.path.join(out, "heat_perturb_neighbor.png"))
    else:
        raise ValueError(f"unknown mode {mode!r}")

    summary = {
        "run_dir": os.path.abspath(run_dir),
        "mode": mode,
        "seeds": seeds,
        "sample_stride": sample_stride,
        "steps": steps,
        "noise_sigma": noise_sigma,
        "n_snapshots": len(all_snaps),
        "message_content": cfg0.backbone.message_content,
        "mp_rounds": getattr(cfg0.backbone, "mp_rounds", None),
        "agg": getattr(cfg0.backbone, "agg", None),
        "n_agents": int(all_snaps[0]["adj"].shape[0]) if all_snaps else 0,
        "stats": stats,
        "verdict": verdict,
    }
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[{mode}] {verdict}", flush=True)
    print(f"wrote {os.path.join(out, 'summary.json')} (+ PNGs)", flush=True)
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(
        description="probe whether the ctde_v0 GNN backbone uses neighbour messages")
    p.add_argument("--run-dir", required=True, help="dir with config.json + model.eqx")
    p.add_argument("--out", required=True, help="output DIRECTORY (heatmaps/hists/summary.json)")
    p.add_argument("--mode", required=True, choices=["graphoff", "edge", "perturb"])
    p.add_argument("--seeds", default="0,1,2", help="comma-separated rollout seeds")
    p.add_argument("--sample-stride", type=int, default=10, help="sample a state every N steps")
    p.add_argument("--noise-sigma", type=float, default=0.25, help="perturb-mode Gaussian std")
    p.add_argument("--steps", type=int, default=100, help="rollout horizon")
    a = p.parse_args(argv)
    seeds = [int(x) for x in str(a.seeds).split(",") if x.strip() != ""]
    run_probe(a.run_dir, a.out, a.mode, seeds, a.sample_stride, a.noise_sigma, a.steps)


if __name__ == "__main__":
    main()
