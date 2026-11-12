"""Métricas financieras de evaluación de portafolios fuera de muestra (OOS)."""

from typing import List, Dict, Any
import numpy as np
import pandas as pd

from vae_hrp.config import TRADING_DAYS, RISK_FREE_RATE


def compute_sd(daily_returns: pd.Series, trading_days: int = TRADING_DAYS) -> float:
    """Desviación estándar anualizada del portafolio: sigma * sqrt(252)."""
    return float(daily_returns.std() * np.sqrt(trading_days))


def compute_sr(
    daily_returns: pd.Series,
    risk_free_rate: float = RISK_FREE_RATE,
    trading_days: int = TRADING_DAYS,
) -> float:
    """Sharpe Ratio anualizado compuesto: ((1 + R_oos)^(252/T) - 1 - rf) / SD_anualizada."""
    if len(daily_returns) == 0:
        return 0.0
    ann_ret = float((1.0 + daily_returns).prod() ** (trading_days / len(daily_returns)) - 1.0)
    ann_vol = compute_sd(daily_returns, trading_days=trading_days)
    if ann_vol < 1e-12:
        return 0.0
    return float((ann_ret - risk_free_rate) / ann_vol)


def compute_total_return(daily_returns: pd.Series) -> float:
    """Retorno acumulado total en el período: prod(1 + r_t) - 1."""
    return float((1.0 + daily_returns).prod() - 1.0)


def compute_max_drawdown(daily_returns: pd.Series) -> float:
    """Máximo Drawdown (caída máxima pico a valle en la curva acumulada)."""
    cum = (1.0 + daily_returns).cumprod()
    peak = cum.cummax()
    return float(((cum - peak) / peak).min())


def compute_turnover(weights_history: List[np.ndarray]) -> float:
    """Rotación media inter-trimestral de pesos: (1/(K-1)) * sum_k sum_i |w_i,k - w_i,k-1|."""
    if len(weights_history) < 2:
        return float('nan')
    diffs = [
        np.sum(np.abs(weights_history[k] - weights_history[k - 1]))
        for k in range(1, len(weights_history))
    ]
    return float(np.mean(diffs))


def compute_ena(weights_history: List[np.ndarray]) -> float:
    """Número Efectivo de Activos medio vía índice Herfindahl: ENA = 1 / sum(w_i^2)."""
    enas = [1.0 / np.sum(w ** 2) for w in weights_history if np.sum(w ** 2) > 1e-12]
    return float(np.mean(enas)) if enas else float('nan')


def summarise_portfolio(
    label: str,
    oos_returns: pd.Series,
    weights_history: List[np.ndarray],
    risk_free_rate: float = RISK_FREE_RATE,
) -> Dict[str, Any]:
    """Genera un diccionario consolidado con las seis métricas auditadas."""
    return {
        'Portfolio': label,
        'SD': compute_sd(oos_returns),
        'SR': compute_sr(oos_returns, risk_free_rate=risk_free_rate),
        'TOR': compute_total_return(oos_returns),
        'MD': compute_max_drawdown(oos_returns),
        'Turnover': compute_turnover(weights_history),
        'ENA': compute_ena(weights_history),
    }
