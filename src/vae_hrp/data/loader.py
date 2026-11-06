"""Carga y preparación de datos del mercado accionario NUAM."""

import os
from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np
import pandas as pd

from vae_hrp.config import DEFAULT_LOCAL_DATA_PATH, FALLBACK_REMOTE_DATA_URL


def load_price_data(
    file_path_or_url: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Carga los precios diarios brutos desde archivo local o URL remota.
    
    Prioriza el archivo local si existe; en caso contrario, utiliza la URL de respaldo.
    """
    target = file_path_or_url or DEFAULT_LOCAL_DATA_PATH
    
    # 1. Intentar cargar desde ruta local
    if isinstance(target, (str, Path)) and os.path.exists(target):
        data = pd.read_excel(target, index_col='Dates')
    else:
        # 2. Intentar cargar desde URL remota
        url = str(target) if isinstance(target, str) and target.startswith('http') else FALLBACK_REMOTE_DATA_URL
        data = pd.read_excel(url, index_col='Dates')

    data.index = pd.to_datetime(data.index)
    return data


def prepare_log_returns(
    prices_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[str]]:
    """Calcula los retornos logarítmicos continuos y descarta activos con datos faltantes.
    
    Args:
        prices_df: DataFrame indexado por fecha con precios de cierre.
        
    Returns:
        returns_df: DataFrame con retornos logarítmicos limpios.
        dropped_cols: Lista de nombres de columnas descartadas por contener valores nulos.
    """
    nan_columns = prices_df.columns[prices_df.isna().any()].tolist()
    clean_prices = prices_df.drop(columns=nan_columns)
    
    log_returns = np.log(clean_prices / clean_prices.shift(1))
    log_returns = log_returns.dropna(how='all')
    log_returns = log_returns.ffill()
    
    return log_returns, nan_columns
