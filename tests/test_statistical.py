"""Pruebas unitarias para métricas de significancia estadística y Deflated Sharpe Ratio."""

import pytest
import numpy as np
import pandas as pd

from vae_hrp.metrics.statistical import (
    compute_moments,
    probabilistic_sharpe_ratio,
    deflated_sharpe_ratio,
    jobson_korkie_test,
)


def test_compute_moments():
    np.random.seed(42)
    daily_returns = pd.Series(np.random.normal(0.0005, 0.01, size=500))
    m = compute_moments(daily_returns, risk_free_rate=0.0462)

    assert m['T'] == 500
    assert isinstance(m['skewness'], float)
    assert isinstance(m['kurtosis'], float)
    assert m['kurtosis'] > 1.0  # Pearson normal ~ 3.0


def test_probabilistic_sharpe_ratio():
    # Estrategia con Sharpe muy alto (1.5 anual) con 1000 días -> PSR > 0.99
    psr_very_high = probabilistic_sharpe_ratio(
        observed_sr_ann=1.50,
        benchmark_sr_ann=0.0,
        n_samples=1000,
        skewness=0.0,
        kurtosis=3.0,
    )
    assert psr_very_high > 0.99

    # Estrategia con Sharpe moderado (0.80 anual) -> PSR > 0.90
    psr_high = probabilistic_sharpe_ratio(
        observed_sr_ann=0.80,
        benchmark_sr_ann=0.0,
        n_samples=1000,
        skewness=0.0,
        kurtosis=3.0,
    )
    assert psr_high > 0.90

    # Estrategia idéntica al benchmark -> PSR = 0.50
    psr_neutral = probabilistic_sharpe_ratio(
        observed_sr_ann=0.0,
        benchmark_sr_ann=0.0,
        n_samples=1000,
        skewness=0.0,
        kurtosis=3.0,
    )
    assert np.isclose(psr_neutral, 0.5, atol=1e-3)


def test_deflated_sharpe_ratio_penalizes_trials():
    trials = [0.1, 0.2, 0.3, 0.4, 0.5, 0.55]
    best_sr = 0.55

    # DSR con 6 ensayos
    dsr_6, sr_star_6 = deflated_sharpe_ratio(
        observed_sr_ann=best_sr,
        trials_ann_srs=trials,
        n_samples=1000,
        skewness=-0.2,
        kurtosis=4.0,
    )

    # Con 50 ensayos con la misma varianza, el umbral requerido debe ser mayor
    trials_many = trials * 8
    dsr_many, sr_star_many = deflated_sharpe_ratio(
        observed_sr_ann=best_sr,
        trials_ann_srs=trials_many,
        n_samples=1000,
        skewness=-0.2,
        kurtosis=4.0,
    )

    assert sr_star_many > sr_star_6
    assert dsr_many < dsr_6


def test_jobson_korkie_test():
    np.random.seed(42)
    # Portafolios correlacionados (comparten componente de mercado)
    market = np.random.normal(0.0005, 0.01, size=1000)
    # Estrategia A añade alfa positivo sistemático con alta correlación
    r_a = pd.Series(market + np.random.normal(0.0003, 0.002, size=1000))
    # Estrategia B sin alfa
    r_b = pd.Series(market + np.random.normal(0.0, 0.002, size=1000))

    res = jobson_korkie_test(r_a, r_b, risk_free_rate=0.0462)
    assert res['SR_A'] > res['SR_B']
    assert res['diff_SR'] > 0
    assert res['p_value_one_sided'] < 0.01
