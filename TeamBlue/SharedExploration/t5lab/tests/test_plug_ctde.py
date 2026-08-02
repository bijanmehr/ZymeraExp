import jax
import jax.numpy as jnp
from t5lab.mvprop import MVProp
from t5lab.plug_ctde import mvprop_field


def test_mvprop_field_shape_and_finite():
    """Adapter returns an (N,H,W) finite field. (Goal-as-minimum is a TRAINED property, not
    asserted here — the convs are random at init.)"""
    m = MVProp(in_ch=2, K=16, key=jax.random.PRNGKey(0))
    N, H, W = 3, 12, 12
    goal = jnp.array([[6, 11], [0, 0], [11, 5]])
    blocked = jnp.zeros((N, H, W), bool)
    field = mvprop_field(m, goal, blocked)
    assert field.shape == (N, H, W)
    assert jnp.isfinite(field).all()
