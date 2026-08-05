"""Control: ctde_v0's OWN train() on my open config. Env vars ITERS(300), ANTI(off|on)
isolate my-pipeline-bug vs config/reward (undertraining? anti-overlap?)."""
import os
import dataclasses
import jax
from t5lab.run_phase1a import base_config
from ctde_v0 import env_utils as eu
from ctde_v0 import ppo

IT = int(os.environ.get("ITERS", 300))
ANTI = os.environ.get("ANTI", "off")
MECH = os.environ.get("MECH", "")        # override mission_safety.mechanism (e.g. soft_lambda, lagrangian)
CONNSIG = os.environ.get("CONNSIG", "")  # override conn_signal (e.g. local_edge_margin)
EXPTOOL = os.environ.get("EXPTOOL", "")  # override explorer_tool (e.g. frontier_attn)
cfg = base_config(terrain="open", grid=32, n_agents=10, iters=IT)
cfg = dataclasses.replace(cfg, reward_anti_overlap=ANTI)
ms = cfg.mission_safety
if MECH:
    ms = dataclasses.replace(ms, mechanism=MECH)
if CONNSIG:
    ms = dataclasses.replace(ms, conn_signal=CONNSIG)
cfg = dataclasses.replace(cfg, mission_safety=ms)
if EXPTOOL:
    cfg = dataclasses.replace(cfg, action_head=dataclasses.replace(cfg.action_head, explorer_tool=EXPTOOL))
print("mechanism=%s conn_signal=%s explorer_tool=%s" % (
    cfg.mission_safety.mechanism, cfg.mission_safety.conn_signal, cfg.action_head.explorer_tool), flush=True)
env = eu.build_env(cfg)


def logf(it, logs):
    if it % 25 == 0 or it == IT - 1:
        print("[it %d] cov=%.3f conn=%.3f connreal=%.3f rew=%.1f ent=%.2f" % (
            it, logs["coverage_pct"], logs["connectivity_pct"],
            logs["connectivity_real"], logs["ep_reward"], logs["entropy"]), flush=True)


print("=== ctde_v0 OWN train, open 32x32/10, iters=%d anti_overlap=%s ===" % (IT, ANTI), flush=True)
state, hist = ppo.train(env, cfg, key=jax.random.PRNGKey(0), log_fn=logf)
print("CTDE_V0 FINAL coverage_pct =", round(hist[-1]["coverage_pct"], 3), flush=True)
print("=== CTDE BASELINE DONE ===", flush=True)
