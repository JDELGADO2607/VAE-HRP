"""Pruebas unitarias para cálculo de métricas financieras."""

import pytest
import numpy as np
import pandas as pd

from vae_hrp.metrics.performance import (
    compute_sd,
    compute_sr,
    compute_total_return,
    compute_max_drawdown,
    compute_turnover,
    compute_ena,
    summarise_portfolio,
)


def test_financial_metrics_consistency():
    dates = pd.date_range('2022-01-01', periods=252, freq='B')
    # Serie determinista con retorno diario constante del 0.05%
    daily_r = pd.Series(0.0005, index=dates)

    sd = compute_sd(daily_r)
    tor = compute_total_return(daily_r)
    md = compute_max_drawdown(daily_r)

    assert sd < 1e-12  # Sin varianza (tolerancia de precisión flotante)
    assert tor > 0.10  # ~ (1 + 0.0005)^252 - 1 ≈ 13.4%
    assert md == 0.0   # Nunca cae

    # Probar con serie volátil
    np.random.seed(42)
    volatile_r = pd.Series(np.random.normal(0.0004, 0.01, size=252), index=dates)
    sr = compute_sr(volatile_r, risk_free_rate=0.0462)
    assert isinstance(sr, float)


def test_turnover_and_ena():
    w1 = np.array([0.5, 0.5])
    w2 = np.array([0.2, 0.8])
    weights_hist = [w1, w2]

    # Turnover: |0.2 - 0.5| + |0.8 - 0.5| = 0.3 + 0.3 = 0.6
    turnover = compute_turnover(weights_hist)
    assert np.isclose(turnover, 0.6)

    # ENA para w1: 1 / (0.25 + 0.25) = 2.0
    # ENA para w2: 1 / (0.04 + 0.64) = 1 / 0.68 ≈ 1.470588
    # Media: (2.0 + 1.470588) / 2 ≈ 1.735294
    ena = compute_ena(weights_hist)
    expected_ena = (2.0 + (1.0 / (0.2**2 + 0.8**2))) / 2.0
    assert np.isclose(ena, expected_ena)
