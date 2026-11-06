"""Pruebas de significancia estadística y Deflated Sharpe Ratio (DSR) de Bailey & López de Prado (2014)."""

from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import pandas as pd
from scipy.stats import norm

from vae_hrp.config import TRADING_DAYS, RISK_FREE_RATE


def compute_moments(
    daily_returns: pd.Series | np.ndarray,
    risk_free_rate: float = RISK_FREE_RATE,
    trading_days: int = TRADING_DAYS,
) -> Dict[str, float]:
    """Calcula los momentos estandarizados de los retornos en exceso (media, desv, asimetría, curtosis)."""
    returns = np.asarray(daily_returns, dtype=float)
    T = len(returns)
    if T < 3:
        return {'T': T, 'mean_daily': 0.0, 'std_daily': 0.0, 'sr_daily': 0.0,
                'sr_ann': 0.0, 'skewness': 0.0, 'kurtosis': 3.0}

    rf_daily = risk_free_rate / trading_days
    excess_r = returns - rf_daily

    mean_d = float(np.mean(excess_r))
    std_d = float(np.std(excess_r, ddof=1))
    sr_daily = mean_d / std_d if std_d > 1e-12 else 0.0
    sr_ann = sr_daily * np.sqrt(trading_days)

    diff = excess_r - mean_d
    m3 = float(np.mean(diff ** 3))
    m4 = float(np.mean(diff ** 4))
    skewness = m3 / (std_d ** 3) if std_d > 1e-12 else 0.0
    kurtosis = m4 / (std_d ** 4) if std_d > 1e-12 else 3.0  # Pearson (normal = 3)

    return {
        'T': T,
        'mean_daily': mean_d,
        'std_daily': std_d,
        'sr_daily': sr_daily,
        'sr_ann': sr_ann,
        'skewness': skewness,
        'kurtosis': kurtosis,
    }


def probabilistic_sharpe_ratio(
    observed_sr_ann: float,
    benchmark_sr_ann: float = 0.0,
    n_samples: int = 1385,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    trading_days: int = TRADING_DAYS,
) -> float:
    """Calcula el Probabilistic Sharpe Ratio (PSR) de Bailey & López de Prado (2012).
    
    Evalúa la probabilidad de que el Sharpe poblacional sea estrictamente mayor a benchmark_sr_ann,
    corrigiendo por asimetría, curtosis (Pearson) y tamaño de muestra.
    """
    if n_samples <= 1:
        return 0.5

    sr_daily = observed_sr_ann / np.sqrt(trading_days)
    sr_star_daily = benchmark_sr_ann / np.sqrt(trading_days)

    denom_sq = (1.0 - skewness * sr_daily + ((kurtosis - 1.0) / 4.0) * (sr_daily ** 2)) / (n_samples - 1)
    if denom_sq <= 1e-14:
        return 1.0 if sr_daily > sr_star_daily else 0.0

    z = (sr_daily - sr_star_daily) / np.sqrt(denom_sq)
    return float(norm.cdf(z))


def deflated_sharpe_ratio(
    observed_sr_ann: float,
    trials_ann_srs: List[float] | np.ndarray,
    n_samples: int = 1385,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    trading_days: int = TRADING_DAYS,
) -> Tuple[float, float]:
    """Calcula el Deflated Sharpe Ratio (DSR) de Bailey & López de Prado (2014).
    
    Ajusta el benchmark SR0* para penalizar el sesgo de selección derivado de haber
    evaluado N combinaciones de hiperparámetros o estrategias.
    
    Args:
        observed_sr_ann: Sharpe ratio anualizado del portafolio óptimo seleccionado.
        trials_ann_srs: Lista o array con los Sharpe ratios anualizados de todas las N alternativas evaluadas.
        n_samples: Longitud de la serie temporal fuera de muestra (días).
        skewness: Asimetría de los retornos del portafolio seleccionado.
        kurtosis: Curtosis de Pearson de los retornos (normal = 3).
        trading_days: Días hábiles anuales (252).
        
    Returns:
        dsr: Probabilidad [0, 1] de que el Sharpe sea genuinamente positivo tras deflactar por N ensayos.
        sr_star_ann: El umbral de Sharpe anualizado mínimo requerido para superar el azar.
    """
    srs = np.asarray(trials_ann_srs, dtype=float)
    N = len(srs)
    if N <= 1:
        sr_star_ann = 0.0
    else:
        # Varianza muestral de los Sharpes evaluados (en escala anualizada)
        var_ann = float(np.var(srs, ddof=1))
        euler_mascheroni = 0.57721566490153286
        z1 = float(norm.ppf(1.0 - 1.0 / N))
        z2 = float(norm.ppf(1.0 - 1.0 / (N * np.e)))
        # Umbral crítico de Sharpe anualizado
        sr_star_ann = np.sqrt(var_ann) * ((1.0 - euler_mascheroni) * z1 + euler_mascheroni * z2)

    dsr = probabilistic_sharpe_ratio(
        observed_sr_ann=observed_sr_ann,
        benchmark_sr_ann=sr_star_ann,
        n_samples=n_samples,
        skewness=skewness,
        kurtosis=kurtosis,
        trading_days=trading_days,
    )
    return dsr, float(sr_star_ann)


def jobson_korkie_test(
    returns_a: pd.Series | np.ndarray,
    returns_b: pd.Series | np.ndarray,
    risk_free_rate: float = RISK_FREE_RATE,
    trading_days: int = TRADING_DAYS,
) -> Dict[str, float]:
    """Prueba de hipótesis de Jobson & Korkie (1981) con corrección de Memmel (2003).
    
    Contrasta la hipótesis nula H0: SR_A == SR_B contra H1: SR_A != SR_B (o unilateral).
    """
    ra = np.asarray(returns_a, dtype=float)
    rb = np.asarray(returns_b, dtype=float)
    T = min(len(ra), len(rb))
    ra, rb = ra[:T], rb[:T]

    rf_daily = risk_free_rate / trading_days
    ea = ra - rf_daily
    eb = rb - rf_daily

    mu_a, mu_b = np.mean(ea), np.mean(eb)
    var_a, var_b = np.var(ea, ddof=1), np.var(eb, ddof=1)
    std_a, std_b = np.sqrt(var_a), np.sqrt(var_b)
    cov_ab = np.cov(ea, eb)[0, 1]

    sr_a_ann = (mu_a / std_a) * np.sqrt(trading_days)
    sr_b_ann = (mu_b / std_b) * np.sqrt(trading_days)
    diff_sr = sr_a_ann - sr_b_ann

    # Corrección de Memmel (2003) para la varianza asintótica de la diferencia
    theta = (
        2.0 * (1.0 - (cov_ab / (std_a * std_b)))
        + 0.5 * (
            (mu_a ** 2 / var_a) + (mu_b ** 2 / var_b)
            - 2.0 * (mu_a * mu_b / (std_a * std_b)) * (cov_ab ** 2 / (var_a * var_b))
        )
    )
    se_diff_daily = np.sqrt(theta / T) if theta > 0 else 1e-8
    se_diff_ann = se_diff_daily * np.sqrt(trading_days)

    z_stat = (diff_sr / se_diff_ann) if se_diff_ann > 1e-12 else 0.0
    p_value_two_sided = 2.0 * (1.0 - norm.cdf(abs(z_stat)))
    p_value_one_sided = 1.0 - norm.cdf(z_stat)

    return {
        'T': T,
        'SR_A': sr_a_ann,
        'SR_B': sr_b_ann,
        'diff_SR': diff_sr,
        'z_stat': z_stat,
        'p_value_two_sided': p_value_two_sided,
        'p_value_one_sided': p_value_one_sided,
    }
