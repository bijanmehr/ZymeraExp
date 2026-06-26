import numpy as np
from fidler.methods import degree_regression as dr


def test_fit_predict_recovers_monotone_relationship():
    rng = np.random.default_rng(0)
    deg = rng.uniform(1, 8, size=400)
    lam = 0.3 * deg + 0.05 * deg ** 2          # synthetic monotone target
    model = dr.fit(deg, lam, degree=2, ridge=1e-6)
    pred = dr.predict(model, deg)
    ss_res = np.sum((lam - pred) ** 2); ss_tot = np.sum((lam - lam.mean()) ** 2)
    assert 1 - ss_res / ss_tot > 0.98

def test_predict_shape():
    model = dr.fit(np.array([1.0, 2.0, 3.0]), np.array([0.3, 0.7, 1.2]), degree=1, ridge=1e-6)
    assert dr.predict(model, np.array([1.5, 2.5])).shape == (2,)
