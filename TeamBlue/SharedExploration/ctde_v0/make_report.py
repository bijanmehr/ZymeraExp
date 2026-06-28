"""Build an interactive HTML report of trained-policy rollouts.

For each (label, run_dir) it replays the trained policy (same step as ppo._single_rollout,
agents colored by chosen SKILL or ROLE), exports the per-step trajectory, and writes
``<out>/data.js`` + ``<out>/index.html`` — a self-contained canvas viewer with play / pause
/ scrub, per-agent SENSE region + COMM radius, DASHED comm links, and colored covered cells.
Open ``<out>/index.html`` in a browser (data.js loads via <script>, so no server needed).

    JAX_PLATFORMS=cpu PYTHONPATH=.:../../../FiedlerValueEstimation \
    /Users/bijanmehr/Project.Zymera/zymera_lab/.venv/bin/python -m ctde_v0.make_report \
        --out report --runs role_32:/tmp/pull/role_32 base_32:/tmp/pull/base_32
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from ctde_v0 import env_utils, ppo
    from ctde_v0.config import from_dict
else:
    from . import env_utils, ppo
    from .config import from_dict


def rollout_worlds(run_dir: str, *, steps: int = 100, seed: int = 0, world_override=None):
    """Replay the trained policy in ``run_dir``; return (cfg, [World,...]) with each
    World's ``group`` recoloured by the agent's chosen skill (selector) / role.

    ``world_override`` (dict) swaps World fields BEFORE building the env — e.g.
    ``{"terrain": "walls", "n_obstacles": 200}`` drops the trained policy into a
    denser obstacle map zero-shot (the LPAC backbone is obstacle/scale-invariant)."""
    cfg = from_dict(json.load(open(os.path.join(run_dir, "config.json"))))
    wkw = {"horizon": steps}
    if world_override:
        wkw.update(world_override)
    cfg = dataclasses.replace(cfg, world=dataclasses.replace(cfg.world, **wkw))
    env = env_utils.build_env(cfg)
    state = ppo.init_state_from_checkpoint(
        env, cfg, os.path.join(run_dir, "model.eqx"), jax.random.PRNGKey(seed))
    actor = state.actor
    stencil = ppo.make_stencil(cfg)
    edge_msg = cfg.backbone.message_content != "learned"
    use_sel = cfg.selector == "on"
    use_roles = cfg.role_picker == "expl_relay"

    rk0, kk = jax.random.split(jax.random.PRNGKey(seed + 7))
    obs, st = env.reset(rk0)
    h = actor.init_hidden(st.n_agents)
    worlds = [st]
    for _ in range(steps):
        kk, ak, rk, sk = jax.random.split(kk, 4)
        adj = env_utils.kb_adjacency(st.body.position, cfg)
        dist = env_utils.kb_distance(st.body.position, cfg) if edge_msg else None
        tag = None
        if use_sel:
            ck = jax.random.fold_in(rk, 0x5E1)
            sl, ol, _f, h = actor.skill_forward(
                obs, adj, st.body.position, dist=dist, h=h, inference=True)
            skill = jax.random.categorical(ck, sl, axis=-1)
            gl = jnp.take_along_axis(ol, skill[None, :, None], axis=0)[0]
            role_idx, tag = None, skill
        else:
            gl, rl, _v, _l2, _z, h = actor(obs, adj, dist=dist, h=h, inference=True)
            if use_roles:
                role_idx = jax.random.categorical(rk, rl, axis=-1)
                tag = role_idx
            else:
                role_idx = None
        gm = ppo._goal_mask(env, st, cfg, stencil)
        goal = jax.random.categorical(ak, jnp.where(gm, gl, ppo._NEG), axis=-1)
        move, _ = ppo._goal_to_move(env, st, goal, stencil, role_idx, cfg)
        obs, st, _r, _d, _i = env.step(st, move, sk)
        if tag is not None:
            st = st.replace(group=tag.astype(jnp.int32))
        worlds.append(st)
    return cfg, worlds


def to_dict(cfg, worlds, label: str, cat: str = "", desc: str = "") -> dict:
    h, w = int(worlds[0].grid_h), int(worlds[0].grid_w)
    walls = [int(x) for x in np.flatnonzero(np.asarray(worlds[0].wall))]
    frames = []
    for st in worlds:
        pos = np.asarray(st.body.position)
        adj = np.asarray(st.comm_graph)
        n = adj.shape[0]
        edges = [[int(i), int(j)] for i in range(n) for j in range(i + 1, n) if adj[i, j]]
        frames.append({
            "pos": pos.tolist(),
            "edges": edges,
            "cov": [int(x) for x in np.flatnonzero(np.asarray(st.covered))],
            "tags": [int(g) for g in np.asarray(st.group)],
        })
    kind = ("skill" if cfg.selector == "on"
            else "role" if cfg.role_picker == "expl_relay" else "base")
    cov_b = np.asarray(worlds[-1].covered).reshape(-1).astype(bool)
    free = ~np.asarray(worlds[0].wall).reshape(-1).astype(bool)
    cov_final = 100.0 * float(cov_b[free].sum()) / max(1, int(free.sum()))
    return {"label": label, "cat": cat, "desc": desc, "grid": [h, w],
            "comm_r": int(cfg.world.comm_r), "sense_r": int(cfg.world.sense_r),
            "kind": kind, "cov": round(cov_final, 1), "walls": walls, "frames": frames}


INDEX_HTML = r'''<!doctype html><html><head><meta charset="utf-8"><title>Zymera rollouts</title>
<style>
 body{font-family:system-ui,sans-serif;margin:0;background:#161616;color:#eee}
 #bar{padding:9px 14px;display:flex;gap:14px;align-items:center;flex-wrap:wrap;border-bottom:1px solid #333}
 select,button{font-size:14px;padding:5px 9px;background:#2c2c2c;color:#eee;border:1px solid #555;border-radius:5px}
 button{cursor:pointer} button:hover{background:#3a3a3a} #slider{width:320px}
 canvas{display:block;margin:14px auto;background:#1d1d1d;border:1px solid #333}
 .leg{font-size:12px;opacity:.85} .sw{display:inline-block;width:11px;height:11px;border-radius:50%;vertical-align:-1px;margin:0 3px}
</style></head><body>
<div id="bar">
  <select id="run"></select>
  <button id="play">▶ Play</button>
  <input id="slider" type="range" min="0" value="0">
  <span id="tlabel" class="leg">t = 0</span>
  <label class="leg">speed <input id="speed" type="range" min="1" max="30" value="8" style="width:90px"></label>
  <span class="leg" id="info"></span>
</div>
<div id="desc" style="padding:9px 16px;font-size:14px;line-height:1.45;background:#1a1a1a;border-bottom:1px solid #2a2a2a"></div>
<canvas id="cv"></canvas>
<div class="leg" style="text-align:center;padding-bottom:14px">
  <span style="background:#2e7d32;padding:1px 7px;border-radius:3px">covered</span> ·
  <span style="background:#111;padding:1px 7px;border-radius:3px;border:1px solid #444">wall</span> ·
  agents = dots (color = skill/role) · faint square = <b>sense range</b> · ring = <b>comm range</b> ·
  <span style="color:#26a69a">– – –</span> = <b>delivered comm link</b>
</div>
<div class="leg" id="clegend" style="text-align:center;padding-bottom:16px"></div>
<script src="data.js"></script>
<script>
const TRAJ=window.TRAJ, cv=document.getElementById('cv'), ctx=cv.getContext('2d');
const runSel=document.getElementById('run'), slider=document.getElementById('slider'),
 playBtn=document.getElementById('play'), tlabel=document.getElementById('tlabel'),
 speed=document.getElementById('speed'), info=document.getElementById('info');
let cur=null, t=0, playing=false, cell=18;
const COL=['#4fc3f7','#ff8a65','#aed581','#ba68c8','#fff176','#4db6ac','#f06292','#90a4ae','#a1887f','#dce775'];
const KINDS={skill:['disperse','flock','hold'],role:['explorer','relay']};
const clegend=document.getElementById('clegend');
function colorLegend(){
 if(cur.kind in KINDS){const nm=KINDS[cur.kind];
  clegend.innerHTML='agent dot color = the '+(cur.kind==='skill'?'skill':'role')+' it is using right now: '+
   nm.map((n,i)=>'<span class="sw" style="background:'+COL[i]+'"></span>'+n).join(' &nbsp; ')+
   ' &nbsp;—&nbsp; <i>a dot changes color the moment the policy switches that agent\'s mode</i>';
 } else {
  clegend.innerHTML='agent dot color = individual agent id &nbsp;—&nbsp; <i>homogeneous policy: no shared roles, each agent just gets its own color</i>';
 }}
const cats={};for(const k in TRAJ){const c=TRAJ[k].cat||'runs';(cats[c]=cats[c]||[]).push(k);}
for(const c in cats){const og=document.createElement('optgroup');og.label=c;
 for(const k of cats[c]){const o=document.createElement('option');o.value=k;
  o.textContent=TRAJ[k].label+'  ('+TRAJ[k].cov+'% cov)';og.appendChild(o);}
 runSel.appendChild(og);}
function xy(idx){const W=cur.grid[1];return [idx%W, Math.floor(idx/W)];}
function ctr(c){return (c+0.5)*cell;}
function load(k){cur=TRAJ[k];const H=cur.grid[0],W=cur.grid[1];
 cell=Math.max(6,Math.min(24,Math.floor(700/Math.max(H,W))));cv.width=W*cell;cv.height=H*cell;
 slider.max=cur.frames.length-1;t=0;slider.value=0;
 info.textContent=`${H}×${W} · comm_r ${cur.comm_r} · sense_r ${cur.sense_r} · ${cur.frames[0].pos.length} agents · ${cur.kind}`;
 document.getElementById('desc').innerHTML='<b>'+cur.label+'</b> &nbsp;<span style="opacity:.6">['+(cur.cat||'')+']</span><br>'+(cur.desc||'');
 colorLegend();draw();}
function draw(){const H=cur.grid[0],W=cur.grid[1],f=cur.frames[t];
 ctx.fillStyle='#1d1d1d';ctx.fillRect(0,0,cv.width,cv.height);
 for(const idx of f.cov){const[x,y]=xy(idx);ctx.fillStyle='#2e7d32';ctx.fillRect(x*cell,y*cell,cell,cell);}
 for(const idx of cur.walls){const[x,y]=xy(idx);ctx.fillStyle='#0c0c0c';ctx.fillRect(x*cell,y*cell,cell,cell);}
 ctx.strokeStyle='#2a2a2a';ctx.lineWidth=.5;
 for(let i=0;i<=W;i++){ctx.beginPath();ctx.moveTo(i*cell,0);ctx.lineTo(i*cell,H*cell);ctx.stroke();}
 for(let j=0;j<=H;j++){ctx.beginPath();ctx.moveTo(0,j*cell);ctx.lineTo(W*cell,j*cell);ctx.stroke();}
 f.pos.forEach((p,i)=>{const r=p[0],c=p[1],col=COL[(f.tags[i]||0)%COL.length],s=cur.sense_r;
  ctx.fillStyle=col+'1f';ctx.fillRect((c-s)*cell,(r-s)*cell,(2*s+1)*cell,(2*s+1)*cell);
  ctx.beginPath();ctx.arc(ctr(c),ctr(r),cur.comm_r*cell,0,7);ctx.strokeStyle=col+'40';ctx.lineWidth=1;ctx.stroke();});
 ctx.setLineDash([5,4]);ctx.lineWidth=1.6;ctx.strokeStyle='#26a69a';
 for(const e of f.edges){const a=f.pos[e[0]],b=f.pos[e[1]];ctx.beginPath();ctx.moveTo(ctr(a[1]),ctr(a[0]));ctx.lineTo(ctr(b[1]),ctr(b[0]));ctx.stroke();}
 ctx.setLineDash([]);
 f.pos.forEach((p,i)=>{const col=COL[(f.tags[i]||0)%COL.length];ctx.beginPath();ctx.arc(ctr(p[1]),ctr(p[0]),cell*.33,0,7);
  ctx.fillStyle=col;ctx.fill();ctx.strokeStyle='#000';ctx.lineWidth=1;ctx.stroke();});
 tlabel.textContent='t = '+t;}
slider.oninput=()=>{t=+slider.value;draw();};
playBtn.onclick=()=>{playing=!playing;playBtn.textContent=playing?'⏸ Pause':'▶ Play';};
runSel.onchange=()=>{playing=false;playBtn.textContent='▶ Play';load(runSel.value);};
function tick(){if(playing&&cur){t=(t+1)%cur.frames.length;slider.value=t;draw();}setTimeout(tick,1000/(+speed.value||8));}
load(runSel.value);tick();
</script></body></html>'''


def main(argv=None):
    p = argparse.ArgumentParser(description="build an interactive HTML rollout report")
    p.add_argument("--runs", nargs="+",
                   help="label:run_dir pairs (run_dir holds config.json + model.eqx)")
    p.add_argument("--manifest",
                   help="JSON list of {label,dir,cat,desc} — categorised gallery")
    p.add_argument("--out", default="report")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    if a.manifest:
        runs = json.load(open(a.manifest))
    else:
        runs = [{"label": e.split(":", 1)[0], "dir": e.split(":", 1)[1]} for e in a.runs]
    data = {}
    for r in runs:
        label, rd = r["label"], r["dir"]
        if not os.path.exists(os.path.join(rd, "model.eqx")):
            print(f"SKIP {label}: no model.eqx in {rd}", flush=True)
            continue
        try:
            cfg, worlds = rollout_worlds(rd, steps=a.steps, seed=a.seed,
                                         world_override=r.get("world"))
        except Exception as exc:                                  # one bad run shouldn't kill the report
            print(f"SKIP {label}: {type(exc).__name__}: {exc}", flush=True)
            continue
        data[label] = to_dict(cfg, worlds, label, r.get("cat", ""), r.get("desc", ""))
        print(f"exported {label} [{r.get('cat','')}]: {len(worlds)} frames, "
              f"{cfg.scale}, cov {data[label]['cov']}%", flush=True)
    with open(os.path.join(a.out, "data.js"), "w") as f:
        f.write("window.TRAJ = " + json.dumps(data) + ";")
    with open(os.path.join(a.out, "index.html"), "w") as f:
        f.write(INDEX_HTML)
    print(f"\nwrote {a.out}/index.html + data.js ({len(data)} runs) — open it in a browser",
          flush=True)


if __name__ == "__main__":
    main()
