"""Probe: does the frontier-attention explorer tool actually point at the frontier?

STANDALONE CPU diagnostic (NO training). For a ``explorer_tool == 'frontier_attn'``
checkpoint it replays the trained policy (the SAME un-jitted rollout step as
``render.py`` / ``ppo._single_rollout``) and, per (step, agent), recomputes two things
DIRECTLY off the saved ``Actor.frontier_attn`` params and each agent's own belief/obs:

  (i)  the module's own attention weights
         attn = softmax( q(z_i) · k(feats_i) / √d )    over the K compass sectors
       (exactly ``FrontierAttn.sector_logits`` internals — nets.py:429), and
  (ii) the TRUE per-sector frontier mass
         feats = sector_frontier_features(obs_i, K, sharp)   (nets.py:320),
       whose column 1 (``feats[:,1]``) is the frontier DENSITY toward each sector — the
       honest "how much uncovered ground actually lies that way".

It then answers three questions:

  * ALIGNMENT — how often does argmax(attn) land on argmax(true frontier), over the
    DIRECTIONAL sectors 1..K-1 ('here'=0 excluded)? Chance is 1/(K-1). If alignment ≈
    chance the learned attention is NOT tracking the frontier; the module's frontier-
    positivity would then come only from the hand-derived ``frac`` multiplier, not from
    anything the network learned.
  * GATE — what is ``alpha = softplus(log_alpha)``? ``alpha ≈ 0`` means the whole tool
    is dialed OFF (its additive bias is ~0) — a real, publishable finding on its own.
  * INFLUENCE — on what fraction of (step, agent) does adding the frontier bias actually
    FLIP the (masked) goal argmax vs. the bare ``goal_head``? If it rarely flips, the
    tool is inert regardless of alignment.

Outputs (under ``--out``): a matplotlib panel gallery (frontier heatmaps with the K
compass sectors overlaid, annotated with per-sector frac + attn, and the argmax-attn vs
argmax-frontier arrows), an ``alpha`` gauge, a per-rung alignment bar, an ``index.html``
(styled after ``make_report.py``) and ``summary.json`` with a one-line verdict.

    JAX_PLATFORMS=cpu PYTHONPATH=.:../../../FiedlerValueEstimation \
    /Users/bijanmehr/Project.Zymera/zymera_lab/.venv/bin/python -m ctde_v0.probe_frontier_align \
        --run-dir runs/frontier/seed0/pen0.0 --out probe_out --seeds 0,1,2 --sample-stride 10
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from ctde_v0 import controller, env_utils, nets, ppo
    from ctde_v0.config import from_dict
else:
    from . import controller, env_utils, nets, ppo
    from .config import from_dict


# =============================================================================
# rollout (copied from render.py:36-83, un-jitted, keeps obs + belief each step)
# =============================================================================

def _rollout(run_dir: str, cfg, seed: int):
    """Replay the trained frontier_attn policy; return the actor plus a per-step list
    of (obs, feat/belief, positions, role_idx, combined goal_logits, goal_mask)."""
    env = env_utils.build_env(cfg)
    ckpt = os.path.join(run_dir, "model.eqx")
    state = ppo.init_state_from_checkpoint(env, cfg, ckpt, jax.random.PRNGKey(seed))
    actor = state.actor
    stencil = ppo.make_stencil(cfg)
    edge_msg = cfg.backbone.message_content != "learned"
    use_roles = cfg.role_picker == "expl_relay"

    rk0, kk = jax.random.split(jax.random.PRNGKey(seed + 7))
    obs, st = env.reset(rk0)
    h = actor.init_hidden(st.n_agents)
    steps = []
    for _t in range(cfg.world.horizon):
        kk, ak, rk, sk = jax.random.split(kk, 4)
        adj = env_utils.kb_adjacency(st.body.position, cfg)
        dist = env_utils.kb_distance(st.body.position, cfg) if edge_msg else None
        # SAME forward as render.py's non-selector branch; `feat` is the belief z_i that
        # frontier_attn consumes (post-compass / post-recurrence, exactly as __call__ uses it).
        goal_logits, role_logits, _v, _l2, feat, h = actor(
            obs, adj, dist=dist, h=h, inference=True)
        role_idx = (jax.random.categorical(rk, role_logits, axis=-1) if use_roles else None)
        gmask = ppo._goal_mask(env, st, cfg, stencil)
        masked = jnp.where(gmask, goal_logits, ppo._NEG)
        goal = jax.random.categorical(ak, masked, axis=-1)
        steps.append({
            "obs": np.asarray(obs),
            "feat": np.asarray(feat),
            "pos": np.asarray(st.body.position),
            "role": (np.asarray(role_idx) if use_roles else None),
            "goal_logits": np.asarray(goal_logits),   # combined (head + frontier bias)
            "gmask": np.asarray(gmask),
        })
        move, _ = ppo._goal_to_move(env, st, goal, stencil, role_idx, cfg)
        obs, st, _r, _d, _i = env.step(st, move, sk)
    return actor, steps


# =============================================================================
# per-(step,agent) frontier-attention recomputation off the saved params
# =============================================================================

def _attn_and_feats(fa: nets.FrontierAttn, obs_np, feat_np, K: int):
    """Recompute, for a whole team snapshot, the module's attention weights AND the true
    per-sector frontier features — directly from ``fa.{q,k,d,sharp}`` (nets.py:429-440)
    and ``sector_frontier_features`` (nets.py:320). Returns (attn (N,K), feats (N,K,2))."""
    obs = jnp.asarray(obs_np)
    feat = jnp.asarray(feat_np)
    feats = jax.vmap(lambda o: nets.sector_frontier_features(o, K, fa.sharp))(obs)  # (N,K,2)

    def attn_one(z_i, feats_i):
        qz = fa.q(z_i)                                     # (d,) belief query
        kf = jax.vmap(fa.k)(feats_i)                       # (K,d) sector keys
        scores = (kf @ qz) / jnp.sqrt(float(fa.d))         # (K,) attention scores
        return jax.nn.softmax(scores, axis=0)              # (K,) sector weights

    attn = jax.vmap(attn_one)(feat, feats)                 # (N,K)
    return np.asarray(attn), np.asarray(feats)


def _goal_head_logits(actor, feat_np):
    """Bare goal-head logits (N,K) — the goal policy WITHOUT the frontier bias."""
    return np.asarray(jax.vmap(actor.goal_head)(jnp.asarray(feat_np)))


def _frontier_bias(fa: nets.FrontierAttn, obs_np, feat_np, K: int):
    """The gated additive frontier logits (N,K) exactly as __call__ adds them:
    ``alpha * K * attn * frac``."""
    return np.asarray(fa(jnp.asarray(obs_np), jnp.asarray(feat_np), K))


# =============================================================================
# matplotlib panels + report
# =============================================================================

def _compass_np(K: int):
    """(K,2) float (row,col) unit compass dirs matching controller._COMPASS ordering."""
    return np.asarray(nets._compass_unit_dirs(K))


def _panel(ax, rec, K, compass):
    """One frontier-heatmap panel with the K sectors overlaid + argmax arrows."""
    obs_i = rec["obs_i"]                       # (C,H,W)
    frontier = 1.0 - obs_i[nets._CH_KNOWN]     # (H,W) 1 = uncovered
    H, W = frontier.shape
    cr, cc = rec["cr"], rec["cc"]
    attn, frac = rec["attn"], rec["frac"]
    a_dir, t_dir = rec["attn_dir"], rec["true_dir"]

    ax.imshow(frontier, cmap="magma", origin="upper", vmin=0.0, vmax=1.0,
              interpolation="nearest")
    ax.plot(cc, cr, "o", ms=6, mfc="#00e5ff", mec="black", mew=0.8)  # the agent

    L = 0.36 * max(H, W)
    for k in range(1, K):
        dr, dc = compass[k]
        # faint sector spokes, opacity ∝ attention weight; annotate frac + attn at the tip
        aw = float(attn[k])
        ax.arrow(cc, cr, dc * L * 0.9, dr * L * 0.9, color="#7fdbff",
                 alpha=0.15 + 0.85 * aw, width=0.02 * L, head_width=0.10 * L,
                 length_includes_head=True, zorder=3)
        tx, ty = cc + dc * L * 1.02, cr + dr * L * 1.02
        ax.text(tx, ty, f"{frac[k]:.2f}\n{aw:.2f}", color="white", fontsize=5.2,
                ha="center", va="center", zorder=5,
                bbox=dict(boxstyle="round,pad=0.1", fc="#00000088", ec="none"))
    # bold arrows: LEARNED argmax-attn (cyan) vs TRUE argmax-frontier (lime)
    dr, dc = compass[a_dir]
    ax.arrow(cc, cr, dc * L, dr * L, color="#00e5ff", width=0.05 * L,
             head_width=0.20 * L, length_includes_head=True, zorder=6)
    dr, dc = compass[t_dir]
    ax.arrow(cc, cr, dc * L, dr * L, color="#39ff14", width=0.05 * L,
             head_width=0.20 * L, length_includes_head=True, zorder=6, alpha=0.9)

    ok = "MATCH" if a_dir == t_dir else "miss"
    role = "" if rec["role"] is None else f" {'expl' if rec['role'] == 0 else 'relay'}"
    ax.set_title(f"s{rec['seed']} t{rec['step']} a{rec['agent']}{role}  [{ok}]",
                 fontsize=6.5, color=("#2e7d32" if a_dir == t_dir else "#b71c1c"))
    ax.set_xticks([]); ax.set_yticks([])


INDEX_HTML = r'''<!doctype html><html><head><meta charset="utf-8">
<title>Frontier-attention alignment probe</title>
<style>
 body{{font-family:"Helvetica Neue",Arial,system-ui,sans-serif;margin:0;background:#ffffff;color:#1a1f29}}
 #bar{{padding:14px 24px;background:#f7f8fa;border-bottom:1px solid #dfe3e8}}
 h1{{font-size:19px;margin:0 0 4px}} .sub{{font-size:13px;color:#5a6472}}
 .verdict{{margin:18px 24px;padding:14px 18px;border-radius:6px;font-family:Georgia,serif;
   font-size:16px;line-height:1.5;background:#fcfbe9;border:1px solid #e6e2b0;color:#4a4620}}
 table{{border-collapse:collapse;margin:14px 24px;font-size:13px}}
 td,th{{border:1px solid #dfe3e8;padding:5px 11px;text-align:right}} th{{background:#f2f4f7}}
 td:first-child,th:first-child{{text-align:left}}
 img{{display:block;max-width:96%;margin:14px auto;border:1px solid #d7dce3;box-shadow:0 1px 4px rgba(20,30,50,.07)}}
 .cap{{text-align:center;font-size:12px;color:#5a6472;margin:-6px 0 20px}}
 .leg{{margin:8px 24px;font-size:12px;color:#5a6472}}
 code{{background:#eef1f5;padding:1px 5px;border-radius:3px}}
</style></head><body>
<div id="bar"><h1>Frontier-attention alignment probe</h1>
<div class="sub">{sub}</div></div>
<div class="verdict"><b>Verdict.</b> {verdict}</div>
<table>
<tr><th>metric</th><th>value</th></tr>
{rows}
</table>
<div class="leg">Panels: <span style="color:#0097a7">cyan</span> arrow = argmax <b>attention</b>
 (what the learned module points at) · <span style="color:#2e7d32">green</span> arrow = argmax
 <b>true frontier mass</b> (where uncovered ground actually is). Each sector spoke is annotated
 <code>frac</code> / <code>attn</code>; spoke opacity ∝ attention. A panel is a MATCH when the two
 bold arrows agree over sectors 1..K-1 (sector 0 = "here" is excluded).</div>
<img src="alpha_gauge.png" alt="alpha gauge"><div class="cap">Learned gate
 <code>alpha = softplus(log_alpha)</code>. alpha≈0 ⇒ the frontier bias is switched off.</div>
<img src="alignment_by_rung.png" alt="alignment by rung"><div class="cap">Per-rung attention↔frontier
 alignment vs. the 1/(K-1) chance line.</div>
<img src="gallery.png" alt="panel gallery"><div class="cap">Frontier heatmaps (bright = uncovered)
 with the compass sectors overlaid.</div>
</body></html>'''


# =============================================================================
# main
# =============================================================================

def probe(run_dir: str, out: str, seeds, sample_stride: int):
    cfg = from_dict(json.load(open(os.path.join(run_dir, "config.json"))))
    tool = cfg.action_head.explorer_tool
    if tool != "frontier_attn":
        raise SystemExit(
            f"ABORT: this probe requires explorer_tool == 'frontier_attn', but "
            f"{run_dir}/config.json has explorer_tool == '{tool}'. Nothing to probe "
            f"(the frontier-attention module contributes nothing under '{tool}').")

    os.makedirs(out, exist_ok=True)
    K = int(cfg.action_head.K)
    compass = _compass_np(K)
    chance = 1.0 / max(1, K - 1)

    records = []          # every sampled (seed, step, agent)
    alpha = None
    for seed in seeds:
        actor, steps = _rollout(run_dir, cfg, seed)
        fa = actor.frontier_attn
        alpha = float(jax.nn.softplus(fa.log_alpha))
        for t, s in enumerate(steps):
            if t % sample_stride != 0:
                continue
            attn, feats = _attn_and_feats(fa, s["obs"], s["feat"], K)      # (N,K),(N,K,2)
            base = _goal_head_logits(actor, s["feat"])                     # (N,K)
            bias = _frontier_bias(fa, s["obs"], s["feat"], K)              # (N,K)
            frac = feats[:, :, 0]                                          # (N,K) fraction
            dens = feats[:, :, 1]                                          # (N,K) mass/density
            gm = s["gmask"]
            # masked argmax with vs without the frontier bias -> does the bias flip the goal?
            NEG = -1e9
            base_g = np.argmax(np.where(gm, base, NEG), axis=-1)           # (N,)
            comb_g = np.argmax(np.where(gm, base + bias, NEG), axis=-1)    # (N,)
            N = attn.shape[0]
            for i in range(N):
                attn_dir = int(np.argmax(attn[i, 1:]) + 1)                 # over sectors 1..K-1
                true_dir = int(np.argmax(dens[i, 1:]) + 1)
                own = s["obs"][i, nets._CH_OWN_POS]
                mass = max(float(own.sum()), 1.0)
                rr = (own * np.arange(own.shape[0])[:, None]).sum() / mass
                ccc = (own * np.arange(own.shape[1])[None, :]).sum() / mass
                records.append({
                    "seed": seed, "step": t, "agent": i,
                    "role": (None if s["role"] is None else int(s["role"][i])),
                    "attn_dir": attn_dir, "true_dir": true_dir,
                    "aligned": bool(attn_dir == true_dir),
                    "flipped": bool(base_g[i] != comb_g[i]),
                    "attn": attn[i], "frac": frac[i], "dens": dens[i],
                    "obs_i": s["obs"][i], "cr": float(rr), "cc": float(ccc),
                })

    # ---- aggregate metrics -------------------------------------------------
    aligned = np.array([r["aligned"] for r in records], dtype=float)
    flipped = np.array([r["flipped"] for r in records], dtype=float)
    align_score = float(aligned.mean()) if len(aligned) else float("nan")
    flip_frac = float(flipped.mean()) if len(flipped) else float("nan")

    # explorer-only alignment (roles-on runs: relays discard the goal, so the tool only
    # actually drives explorers). None if role_picker off.
    expl = [r for r in records if r["role"] == 0]
    expl_align = (float(np.mean([r["aligned"] for r in expl])) if expl else None)

    # per-rung (== per-scale) alignment; with one run_dir this is a single rung across seeds.
    rung = cfg.scale
    by_rung = {rung: align_score}

    # ---- figures -----------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # gallery: spread ~24 panels across seeds & sampled steps (prefer explorers)
    n_panels = min(24, len(records))
    pool = [r for r in records if (r["role"] in (None, 0))] or records
    idx = np.linspace(0, len(pool) - 1, num=min(24, len(pool))).round().astype(int)
    picks = [pool[j] for j in dict.fromkeys(idx)]         # unique, order-preserving
    n_panels = len(picks)
    ncol = 6
    nrow = max(1, math.ceil(n_panels / ncol))
    figg, axes = plt.subplots(nrow, ncol, figsize=(2.05 * ncol, 2.25 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, rec in zip(axes, picks):
        ax.axis("on")
        _panel(ax, rec, K, compass)
    figg.suptitle(f"Frontier-attention: argmax-attn (cyan) vs argmax-frontier (green)   "
                  f"[{rung}, alpha={alpha:.3f}]", fontsize=10)
    figg.tight_layout(rect=(0, 0, 1, 0.97))
    figg.savefig(os.path.join(out, "gallery.png"), dpi=130)
    plt.close(figg)

    # alpha gauge
    figa, axa = plt.subplots(figsize=(6, 1.5))
    axa.barh([0], [alpha], color=("#c62828" if alpha < 0.05 else "#2e7d32"), height=0.5)
    axa.axvline(0.05, color="#888", ls="--", lw=1)
    axa.text(0.05, 0.42, " gated-off threshold", fontsize=8, color="#666", va="center")
    axa.set_yticks([]); axa.set_xlabel("alpha = softplus(log_alpha)")
    axa.set_xlim(0, max(1.0, alpha * 1.2))
    axa.set_title(f"frontier-bias gate  alpha = {alpha:.4f}"
                  + ("   (EFFECTIVELY OFF)" if alpha < 0.05 else ""), fontsize=10)
    figa.tight_layout()
    figa.savefig(os.path.join(out, "alpha_gauge.png"), dpi=130)
    plt.close(figa)

    # per-rung alignment bar
    figr, axr = plt.subplots(figsize=(5, 3))
    axr.bar(list(by_rung.keys()), list(by_rung.values()), color="#1565c0")
    axr.axhline(chance, color="#c62828", ls="--", lw=1.2, label=f"chance 1/(K-1)={chance:.3f}")
    axr.set_ylim(0, 1); axr.set_ylabel("P(argmax-attn == argmax-frontier)")
    axr.set_title("attention↔frontier alignment by rung"); axr.legend(fontsize=8)
    figr.tight_layout()
    figr.savefig(os.path.join(out, "alignment_by_rung.png"), dpi=130)
    plt.close(figr)

    # ---- verdict + summary -------------------------------------------------
    if alpha < 0.05:
        verdict = (f"The frontier-attention tool is EFFECTIVELY GATED OFF: "
                   f"alpha = {alpha:.4f} ≈ 0, so its additive bias contributes ~nothing to "
                   f"the goal logits. Whatever coverage this policy achieves does NOT come "
                   f"from the learned frontier module.")
    else:
        ratio = align_score / chance if chance > 0 else float("nan")
        if align_score >= 2 * chance:
            verdict = (f"The frontier attention TRACKS the true frontier: it points at the "
                       f"most-uncovered sector {align_score:.1%} of the time vs a {chance:.1%} "
                       f"chance baseline ({ratio:.1f}× chance), with alpha = {alpha:.3f} and the "
                       f"bias flipping the goal argmax on {flip_frac:.1%} of steps.")
        else:
            verdict = (f"The frontier attention does NOT meaningfully track the frontier: "
                       f"alignment {align_score:.1%} is near the {chance:.1%} chance baseline "
                       f"({ratio:.1f}× chance) despite alpha = {alpha:.3f}. Any dispersal it "
                       f"produces rides on the hand-derived frac multiplier, not learned attention.")

    summary = {
        "run_dir": run_dir, "rung": rung, "K": K, "seeds": seeds,
        "sample_stride": sample_stride, "n_samples": len(records),
        "alpha": alpha, "chance_1_over_Kminus1": chance,
        "alignment_score": align_score, "alignment_vs_chance_ratio":
            (align_score / chance if chance else None),
        "explorer_only_alignment": expl_align,
        "alignment_by_rung": by_rung,
        "goal_flip_fraction": flip_frac,
        "role_picker": cfg.role_picker,
        "verdict": verdict,
    }
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    sub = (f"run: <code>{run_dir}</code> &nbsp; rung {rung} &nbsp; K={K} &nbsp; "
           f"seeds {seeds} &nbsp; stride {sample_stride} &nbsp; {len(records)} samples")
    rows = "".join(
        f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in [
            ("alpha (gate)", f"{alpha:.4f}"),
            ("alignment_score", f"{align_score:.3f}"),
            ("chance 1/(K-1)", f"{chance:.3f}"),
            ("alignment / chance", f"{(align_score/chance if chance else float('nan')):.2f}×"),
            ("explorer-only alignment", ("n/a" if expl_align is None else f"{expl_align:.3f}")),
            ("goal-flip fraction", f"{flip_frac:.3f}"),
            ("samples (agent×step×seed)", str(len(records))),
        ])
    with open(os.path.join(out, "index.html"), "w") as f:
        f.write(INDEX_HTML.format(sub=sub, verdict=verdict, rows=rows))

    print(f"[probe] {verdict}", flush=True)
    print(f"[probe] wrote {out}/index.html + summary.json + 3 PNGs "
          f"({len(records)} samples, alpha={alpha:.4f}, align={align_score:.3f})", flush=True)
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description="probe frontier-attention ↔ true-frontier alignment")
    p.add_argument("--run-dir", required=True, help="dir with config.json + model.eqx")
    p.add_argument("--out", required=True, help="output dir (index.html, PNGs, summary.json)")
    p.add_argument("--seeds", default="0,1,2", help="comma-separated rollout seeds")
    p.add_argument("--sample-stride", type=int, default=10, help="subsample steps by this stride")
    a = p.parse_args(argv)
    seeds = [int(s) for s in a.seeds.split(",") if s.strip() != ""]
    probe(a.run_dir, a.out, seeds, a.sample_stride)


if __name__ == "__main__":
    main()
