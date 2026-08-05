"""Phase-2 planner ARMS — alternative L2 value-field planners that match the MVProp interface
(``.field(x) -> (H,W)``, ``.move_scores(x, rc) -> (5,)``) so they drop straight into the same
distillation trainer, generalization eval, and ``MVPropPlanner`` wrapper. One-file zoo + factory.

Arms (why each — the ablation question):
  * HighwayMVProp — MVProp + a per-cell HIGHWAY carry gate (Highway Value Iteration Networks,
      Wang et al. ICML 2024). ``V_{k+1} = g·max(r, γ·w·max4(V_k)) + (1-g)·V_k``. The ``(1-g)·V_k``
      skip is a GRADIENT HIGHWAY across the K sweeps, the published fix for the vanishing gradient
      that kills deep value iteration. THE mandatory control: does the architectural fix remove the
      need to distill (i.e. does RL-only now train)?
  * GPPN — Gated Path Planning Networks (Lee et al. NeurIPS 2018). Replaces VIN's hard max with a
      DATA-DEPENDENT (LSTM/GRU-style) update gate computed each sweep from [V_k, candidate,
      neighbour-mass], for stable long-horizon credit propagation.
  * MSP — a compact Spatial-planning transformer: AXIAL self-attention over grid rows/cols (size-
      invariant, O(HW·(H+W)) not O((HW)²)) with sinusoidal 2-D positional encoding predicts the
      value field DIRECTLY — no explicit value-iteration recurrence. Tests whether attention can
      learn to route.

All share MVProp's input ``x = (2,H,W) = [blocked, goal_onehot]`` and are fully convolutional /
axial → SIZE-INVARIANT (same weights at any grid), so they compare fairly with MVProp.
"""
import jax
import jax.numpy as jnp
import equinox as eqx

from t5lab.mvprop import _max_4nbr, _DELTAS


def _scores(V, rc):
    """(H,W) field, (2,) cell -> (5,) neighbour values in ACTION_DELTAS order (shared readout)."""
    H, W = V.shape
    cells = jnp.clip(rc[None, :] + _DELTAS, 0, jnp.array([H - 1, W - 1]))
    return V[cells[:, 0], cells[:, 1]]


def _mean_4nbr(V):
    """(H,W) mean over the 4 von-Neumann neighbours (edge cells see fewer, divided by count)."""
    up = jnp.pad(V[1:, :], ((0, 1), (0, 0)))
    down = jnp.pad(V[:-1, :], ((1, 0), (0, 0)))
    left = jnp.pad(V[:, 1:], ((0, 0), (0, 1)))
    right = jnp.pad(V[:, :-1], ((0, 0), (1, 0)))
    one = jnp.ones_like(V)
    cnt = (jnp.pad(one[1:, :], ((0, 1), (0, 0))) + jnp.pad(one[:-1, :], ((1, 0), (0, 0)))
           + jnp.pad(one[:, 1:], ((0, 0), (0, 1))) + jnp.pad(one[:, :-1], ((0, 0), (1, 0))))
    return (up + down + left + right) / jnp.maximum(cnt, 1.0)


# ============================================================ HighwayMVProp
class HighwayMVProp(eqx.Module):
    gain_conv: eqx.nn.Conv2d
    gate_conv: eqx.nn.Conv2d
    K: int = eqx.field(static=True)
    gamma: float = eqx.field(static=True)
    goal_ch: int = eqx.field(static=True)

    def __init__(self, in_ch, K, *, key, gamma=0.9, goal_ch=-1, w_init_bias=2.0):
        kg, kh = jax.random.split(key)
        conv = eqx.nn.Conv2d(in_ch, 1, 3, padding=1, key=kg)
        conv = eqx.tree_at(lambda c: c.bias, conv, conv.bias * 0.0 + jnp.asarray(w_init_bias, conv.bias.dtype))
        self.gain_conv = conv
        self.gate_conv = eqx.nn.Conv2d(in_ch, 1, 3, padding=1, key=kh)   # highway gate, bias 0 -> g~0.5
        self.K = int(K); self.gamma = float(gamma); self.goal_ch = int(goal_ch)

    def field(self, x):
        r = x[self.goal_ch]
        w = jax.nn.sigmoid(self.gain_conv(x)[0])
        g = jax.nn.sigmoid(self.gate_conv(x)[0])                        # (H,W) per-cell update gate
        def step(V, _):
            newV = jnp.maximum(r, self.gamma * w * _max_4nbr(V))
            return g * newV + (1.0 - g) * V, None                      # highway carry
        V, _ = jax.lax.scan(step, r, None, length=self.K)
        return V

    def move_scores(self, x, rc):
        return _scores(self.field(x), rc)


# ============================================================ GPPN
class GPPN(eqx.Module):
    gain_conv: eqx.nn.Conv2d
    gate_conv: eqx.nn.Conv2d          # data-dependent update gate from [V, cand, w*nbr]
    K: int = eqx.field(static=True)
    gamma: float = eqx.field(static=True)
    goal_ch: int = eqx.field(static=True)

    def __init__(self, in_ch, K, *, key, gamma=0.9, goal_ch=-1, w_init_bias=2.0):
        kg, kz = jax.random.split(key)
        conv = eqx.nn.Conv2d(in_ch, 1, 3, padding=1, key=kg)
        conv = eqx.tree_at(lambda c: c.bias, conv, conv.bias * 0.0 + jnp.asarray(w_init_bias, conv.bias.dtype))
        self.gain_conv = conv
        self.gate_conv = eqx.nn.Conv2d(3, 1, 3, padding=1, key=kz)      # input [V, cand, w*nbr]
        self.K = int(K); self.gamma = float(gamma); self.goal_ch = int(goal_ch)

    def field(self, x):
        r = x[self.goal_ch]
        w = jax.nn.sigmoid(self.gain_conv(x)[0])
        def step(V, _):
            nbr = _max_4nbr(V)
            cand = jnp.maximum(r, self.gamma * w * nbr)
            feat = jnp.stack([V, cand, self.gamma * w * nbr])          # (3,H,W) recurrent features
            z = jax.nn.sigmoid(self.gate_conv(feat)[0])                # data-dependent update gate
            return z * cand + (1.0 - z) * V, None
        V, _ = jax.lax.scan(step, r, None, length=self.K)
        return V

    def move_scores(self, x, rc):
        return _scores(self.field(x), rc)


# ============================================================ MSP (axial spatial transformer)
def _sincos_pe(H, W, d):
    """(H,W,d) sinusoidal 2-D positional encoding on NORMALIZED coords (size-invariant)."""
    dh = d // 2
    yy = jnp.linspace(0.0, 1.0, H)[:, None, None]
    xx = jnp.linspace(0.0, 1.0, W)[None, :, None]
    freqs = jnp.exp(jnp.arange(0, dh, 2) * (-jnp.log(10000.0) / max(dh, 1)))   # (dh/2,)
    pey = jnp.concatenate([jnp.sin(yy * freqs), jnp.cos(yy * freqs)], -1)      # (H,1,dh)
    pex = jnp.concatenate([jnp.sin(xx * freqs), jnp.cos(xx * freqs)], -1)      # (1,W,dh)
    pey = jnp.broadcast_to(pey, (H, W, dh)); pex = jnp.broadcast_to(pex, (H, W, dh))
    return jnp.concatenate([pey, pex], -1)                                     # (H,W,d)


def _axial_attn(z, q, k, v):
    """(H,W,d) -> (H,W,d) axial single-head attention: attend along ROWS then COLUMNS."""
    def attend(seq):                              # (L,d) -> (L,d)
        Q, K, V = jax.vmap(q)(seq), jax.vmap(k)(seq), jax.vmap(v)(seq)
        a = jax.nn.softmax((Q @ K.T) / jnp.sqrt(Q.shape[-1]), axis=-1)
        return a @ V
    z = jax.vmap(attend)(z)                        # rows: (H, W, d)
    z = jax.vmap(attend)(z.transpose(1, 0, 2)).transpose(1, 0, 2)   # cols
    return z


class MSP(eqx.Module):
    embed: eqx.nn.Linear
    q1: eqx.nn.Linear; k1: eqx.nn.Linear; v1: eqx.nn.Linear
    q2: eqx.nn.Linear; k2: eqx.nn.Linear; v2: eqx.nn.Linear
    mlp1: eqx.nn.Linear; mlp2: eqx.nn.Linear
    head: eqx.nn.Linear
    d: int = eqx.field(static=True)
    goal_ch: int = eqx.field(static=True)

    def __init__(self, in_ch, K, *, key, gamma=0.9, goal_ch=-1, d=48):
        ks = jax.random.split(key, 10)
        self.embed = eqx.nn.Linear(in_ch, d, key=ks[0])
        self.q1 = eqx.nn.Linear(d, d, key=ks[1]); self.k1 = eqx.nn.Linear(d, d, key=ks[2]); self.v1 = eqx.nn.Linear(d, d, key=ks[3])
        self.q2 = eqx.nn.Linear(d, d, key=ks[4]); self.k2 = eqx.nn.Linear(d, d, key=ks[5]); self.v2 = eqx.nn.Linear(d, d, key=ks[6])
        self.mlp1 = eqx.nn.Linear(d, d, key=ks[7]); self.mlp2 = eqx.nn.Linear(d, d, key=ks[8])
        self.head = eqx.nn.Linear(d, 1, key=ks[9])
        self.d = int(d); self.goal_ch = int(goal_ch)

    def field(self, x):
        C, H, W = x.shape
        feat = x.transpose(1, 2, 0)                                    # (H,W,C)
        z = jax.vmap(jax.vmap(self.embed))(feat) + _sincos_pe(H, W, self.d)   # (H,W,d)
        z = z + _axial_attn(z, self.q1, self.k1, self.v1)
        z = z + jax.vmap(jax.vmap(lambda t: self.mlp1(jax.nn.gelu(t))))(z)
        z = z + _axial_attn(z, self.q2, self.k2, self.v2)
        z = z + jax.vmap(jax.vmap(lambda t: self.mlp2(jax.nn.gelu(t))))(z)
        V = jax.vmap(jax.vmap(self.head))(z)[..., 0]                   # (H,W)
        return jax.nn.sigmoid(V)                                       # value field in (0,1), like gamma^d

    def move_scores(self, x, rc):
        return _scores(self.field(x), rc)


# ============================================================ factory
def make_net(arch, in_ch, K, *, key, gamma=0.9):
    arch = arch.lower()
    if arch in ("mvprop", "mv"):
        from t5lab.mvprop import MVProp
        return MVProp(in_ch=in_ch, K=K, key=key, gamma=gamma)
    if arch in ("highway", "highwaymvprop", "hvin"):
        return HighwayMVProp(in_ch=in_ch, K=K, key=key, gamma=gamma)
    if arch == "gppn":
        return GPPN(in_ch=in_ch, K=K, key=key, gamma=gamma)
    if arch == "msp":
        return MSP(in_ch=in_ch, K=K, key=key, gamma=gamma)
    raise ValueError(f"unknown planner arch: {arch}")
