"""Pruebas unitarias para carga y limpieza de datos."""

import pytest
import numpy as np
import pandas as pd

from vae_hrp.data.loader import load_price_data, prepare_log_returns
from vae_hrp.config import DEFAULT_LOCAL_DATA_PATH


def test_load_local_price_data():
    prices = load_price_data(DEFAULT_LOCAL_DATA_PATH)
    assert isinstance(prices, pd.DataFrame)
    assert prices.shape[0] >= 1700
    assert prices.shape[1] == 56  # 56 activos brutos


def test_prepare_log_returns():
    prices = load_price_data(DEFAULT_LOCAL_DATA_PATH)
    returns, dropped = prepare_log_returns(prices)
    
    assert isinstance(returns, pd.DataFrame)
    assert returns.shape[1] == 53  # 53 activos filtrados
    assert returns.isna().sum().sum() == 0  # Cero NaNs
    assert set(dropped) == {'PEI CB EQUITY', 'NUAM CI EQUITY', 'IPCHBC1 PE Equity'}
    assert returns.shape[0] == prices.shape[0] - 1
