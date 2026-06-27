"""Reference floor: ridge polynomial regression of degree-features -> lambda2 (numpy)."""
import numpy as np


def _design(deg, degree):
    deg = np.asarray(deg, float).reshape(-1)
    return np.stack([deg ** k for k in range(degree + 1)], axis=1)   # (M, degree+1)


def fit(deg, lam, degree: int = 2, ridge: float = 1e-6):
    X = _design(deg, degree)
    A = X.T @ X + ridge * np.eye(X.shape[1])
    w = np.linalg.solve(A, X.T @ np.asarray(lam, float).reshape(-1))
    return {"w": w, "degree": degree}


def predict(model, deg):
    return _design(deg, model["degree"]) @ model["w"]
