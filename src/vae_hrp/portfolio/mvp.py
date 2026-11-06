"""Optimizador de Mínima Varianza (Minimum Variance Portfolio - MVP) y Paridad de Riesgo Clásica."""

import numpy as np
from scipy.optimize import minimize as sp_minimize


def mvp_weights_from_cov(cov: np.ndarray) -> np.ndarray:
    """Calcula los pesos del portafolio de mínima varianza mediante inversión analítica regularizada."""
    n = cov.shape[0]
    ones = np.ones(n)
    try:
        inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        inv = np.linalg.inv(cov + 1e-10 * np.eye(n))
    w = inv @ ones
    return w / w.sum()


def mvp_weights_long_only(cov: np.ndarray) -> np.ndarray:
    """Calcula los pesos de mínima varianza con restricción estricta de solo largo (0 <= w_i <= 1)."""
    n = cov.shape[0]
    cov_ = cov + 1e-10 * np.eye(n)

    def objective(w):
        return float(w @ cov_ @ w)

    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0)] * n
    init_w = np.ones(n) / n

    res = sp_minimize(
        objective,
        init_w,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'ftol': 1e-12, 'maxiter': 1000},
    )
    w = np.clip(res.x if res.success else init_w, 0.0, None)
    return w / w.sum()


def risk_parity_weights(cov: np.ndarray) -> np.ndarray:
    """Calcula los pesos de paridad de riesgo igualitaria clásica (Equal Risk Contribution) vía SLSQP."""
    n = cov.shape[0]
    cov_ = cov + 1e-10 * np.eye(n)

    def objective(w):
        port_var = float(w @ cov_ @ w)
        mrc = cov_ @ w
        rc = w * mrc / max(port_var, 1e-12)
        return float(((rc - 1.0 / n) ** 2).sum())

    constraints = [{'type': 'eq', 'fun': lambda w: w.sum() - 1.0}]
    bounds = [(0.0, 1.0)] * n
    res = sp_minimize(
        objective,
        np.ones(n) / n,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'ftol': 1e-10, 'maxiter': 1000},
    )
    w = np.clip(res.x if res.success else np.ones(n) / n, 0.0, None)
    return w / w.sum()
