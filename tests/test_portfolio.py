"""Pruebas unitarias para los algoritmos de asignación HRP y MVP."""

import pytest
import numpy as np
import pandas as pd

from vae_hrp.portfolio.hrp import hrp_portfolio, build_distance_matrix
from vae_hrp.portfolio.mvp import mvp_weights_from_cov, mvp_weights_long_only


@pytest.fixture
def sample_cov():
    np.random.seed(42)
    n = 10
    A = np.random.randn(n, n)
    cov = A @ A.T + np.eye(n) * 0.1
    cols = [f'Asset_{i}' for i in range(n)]
    return pd.DataFrame(cov, index=cols, columns=cols)


def test_distance_matrix(sample_cov):
    d = build_distance_matrix(sample_cov)
    assert d.shape == sample_cov.shape
    assert np.all(d >= 0.0)
    assert np.all(d <= 1.0)
    np.testing.assert_allclose(np.diag(d), 0.0)


def test_hrp_portfolio(sample_cov):
    w = hrp_portfolio(sample_cov, linkage_method='ward')
    assert isinstance(w, pd.Series)
    assert len(w) == sample_cov.shape[0]
    assert np.isclose(w.sum(), 1.0, atol=1e-5)
    assert (w >= 0.0).all()  # HRP garantiza no negatividad por construcción


def test_mvp_weights_from_cov(sample_cov):
    w = mvp_weights_from_cov(sample_cov.values)
    assert len(w) == sample_cov.shape[0]
    assert np.isclose(w.sum(), 1.0, atol=1e-5)


def test_mvp_weights_long_only(sample_cov):
    w = mvp_weights_long_only(sample_cov.values)
    assert len(w) == sample_cov.shape[0]
    assert np.isclose(w.sum(), 1.0, atol=1e-5)
    assert (w >= -1e-8).all()
