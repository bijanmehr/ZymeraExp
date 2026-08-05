"""Visualize a distilled planner: render its value field (heat) with an agent descending it from a
far start toward the goal — MVProp reaches; MSP gets trapped in a local maximum. The heatmap IS the
learned field (bright = high value = near goal), so the viz shows both the mechanism and the failure.

    PYTHONPATH=.:../../../FiedlerValueEstimation ARCH=mvprop MAP=rooms SEED=3 OUT=gifs/plan_mvprop.gif $PY -m t5lab.render_planner
"""
import os
import numpy as np
import jax, jax.numpy as jnp, equinox as eqx
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from t5lab.planner_arms import make_net
from t5lab.mvprop import _DELTAS, propagate
from t5lab.planner import mvprop_input
from t5lab.planner_generalization import GENS


def _oracle(wall, goal, grid):
    oh = jnp.zeros((grid, grid)).at[goal[0], goal[1]].set(1.0)
    return np.asarray(propagate(oh, (~jnp.asarray(wall)).astype(jnp.float32), 32, 0.9))

ARCH = os.environ["ARCH"]; MAP = os.environ.get("MAP", "rooms")
GRID = int(os.environ.get("GRID", 32)); SEED = int(os.environ.get("SEED", 3))
OUT = os.environ.get("OUT", f"gifs/plan_{ARCH}_{MAP}.gif")


def main():
    net = make_net(ARCH, 2, 32, key=jax.random.PRNGKey(0), gamma=0.9)
    net = eqx.tree_deserialise_leaves(f"planners/{ARCH}_distilled.eqx", net)
    rng = np.random.default_rng(SEED)
    wall = GENS[MAP](rng); free = np.argwhere(~wall)
    goal = free[rng.integers(len(free))]
    O = _oracle(wall, goal, GRID)                                       # geodesic gamma^dist
    cand = np.argwhere((O > 1e-3) & (~wall))                            # cells the goal can reach
    start = cand[int(np.argmin(O[cand[:, 0], cand[:, 1]]))]             # farthest REACHABLE cell
    V = np.asarray(net.field(mvprop_input(jnp.asarray(wall), jnp.asarray(goal))))
    D = np.asarray(_DELTAS); lim = [GRID - 1, GRID - 1]
    cur = start.copy(); path = [tuple(cur)]; seen = {tuple(cur)}; reached = False
    for _ in range(4 * GRID):
        if tuple(cur) == tuple(goal): reached = True; break
        nb = np.clip(cur[None] + D, 0, lim)
        nxt = nb[int(np.argmax(V[nb[:, 0], nb[:, 1]]))]
        if tuple(nxt) == tuple(cur) or tuple(nxt) in seen: break        # trapped
        seen.add(tuple(nxt)); cur = nxt; path.append(tuple(cur))
    Vm = np.where(wall, np.nan, V)
    vmin = np.nanpercentile(Vm, 2); vmax = np.nanmax(Vm)
    frames = []
    for t in range(len(path)):
        fig, ax = plt.subplots(figsize=(4, 4), dpi=115)
        ax.imshow(Vm, cmap="viridis", origin="upper", interpolation="nearest", vmin=vmin, vmax=vmax)
        wy, wx = np.where(wall); ax.scatter(wx, wy, c="#1a1a1a", s=7, marker="s")
        ax.scatter(goal[1], goal[0], c="#ff5aa8", marker="*", s=230, edgecolors="k", linewidths=1.2, zorder=4)
        p = np.array(path[:t + 1]); ax.plot(p[:, 1], p[:, 0], "-", c="w", lw=1.6, alpha=0.8, zorder=3)
        ax.scatter(p[-1, 1], p[-1, 0], c="#ffb44a", s=95, edgecolors="k", linewidths=1.3, zorder=5)
        last = (t == len(path) - 1)
        state = ("REACHED" if reached else "TRAPPED") if last else f"step {t}"
        ax.set_title(f"{ARCH} · {MAP} · {state}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([]); fig.tight_layout(pad=0.2)
        fig.canvas.draw(); frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()); plt.close(fig)
    frames += [frames[-1]] * 8
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(OUT, save_all=True, append_images=imgs[1:], duration=110, loop=0, optimize=True)
    print(f"saved {OUT}  arch={ARCH} map={MAP} reached={reached} steps={len(path)}", flush=True)


if __name__ == "__main__":
    main()
