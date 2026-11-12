"""Preprocesamiento causal de retornos sin sesgo de anticipación (look-ahead bias)."""

from typing import Tuple
import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd


def fit_preprocess(windows: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Estima los estadísticos robustos de winsorización MAD únicamente sobre la partición de entrenamiento.
    
    Args:
        windows: Tensor [N, S, A] correspondiente a las ventanas de entrenamiento.
        
    Returns:
        median: Tensor [1, 1, A] con la mediana por activo.
        mad: Tensor [1, 1, A] con la desviación absoluta mediana acotada inferiormente por 1e-8.
    """
    N, S, A = windows.shape
    flat = windows.reshape(N * S, A)
    median = torch.quantile(flat, 0.5, dim=0)
    abs_dev = torch.abs(flat - median.unsqueeze(0))
    mad = torch.quantile(abs_dev, 0.5, dim=0)
    
    median = median.unsqueeze(0).unsqueeze(0)
    mad = mad.unsqueeze(0).unsqueeze(0)
    return median, torch.clamp(mad, min=1e-8)


def apply_preprocess(
    windows: torch.Tensor,
    median: torch.Tensor,
    mad: torch.Tensor,
) -> torch.Tensor:
    """Aplica winsorización robusta a +/- 5 MAD y estandarización Z-score intra-ventana.
    
    Args:
        windows: Tensor [*, S, A] de ventanas a transformar.
        median: Tensor [1, 1, A] con la mediana estimada en entrenamiento.
        mad: Tensor [1, 1, A] con la MAD estimada en entrenamiento.
        
    Returns:
        Tensor [*, S, A] estandarizado con media cero y varianza unitaria por ventana.
    """
    clipped = torch.max(
        torch.min(windows, median + 5 * mad),
        median - 5 * mad
    )
    mean = torch.mean(clipped, dim=-2, keepdim=True)
    std = torch.std(clipped, dim=-2, keepdim=True)
    return (clipped - mean) / torch.clamp(std, min=1e-8)


def build_loaders(
    df: pd.DataFrame,
    S: int = 60,
    batch_size: int = 32,
    val_frac: float = 0.15,
    gamma: float = 1.0,
) -> Tuple[DataLoader, DataLoader, torch.Tensor, torch.Tensor]:
    """Segmenta las ventanas rodantes cronológicamente, aplica preprocesamiento causal y genera DataLoaders.
    
    Args:
        df: DataFrame de retornos logarítmicos del período histórico hasta el rebalanceo.
        S: Longitud de la ventana deslizante (días hábiles).
        batch_size: Tamaño de lote para el DataLoader.
        val_frac: Fracción final reservada para validación causal.
        gamma: Factor de decaimiento temporal exponencial (1.0 = uniforme, <1.0 = más peso a lo reciente).
        
    Returns:
        train_loader: DataLoader para entrenamiento con pesos temporales.
        val_loader: DataLoader para validación.
        tensor_X_full: Concatenación de ventanas de entrenamiento y validación transformadas.
        raw_wins: Ventanas brutas sin transformar.
    """
    raw = torch.tensor(df.values, dtype=torch.float32)
    raw_wins = raw.unfold(0, S, 1).transpose(1, 2)  # [N, S, A]

    N = raw_wins.shape[0]
    n_val = max(1, int(N * val_frac))
    n_tr = N - n_val

    # 1. Partición cronológica estricta antes de cualquier estimación
    train_raw = raw_wins[:n_tr]
    val_raw = raw_wins[n_tr:]

    # 2. Ajuste de estadísticos únicamente en entrenamiento
    median_tr, mad_tr = fit_preprocess(train_raw)

    # 3. Aplicación de estadísticos a ambas particiones (cero fuga de validación)
    tensor_X_train = apply_preprocess(train_raw, median_tr, mad_tr)
    tensor_X_val = apply_preprocess(val_raw, median_tr, mad_tr)

    # 4. Objetivos de reconstrucción idénticos a las entradas
    tensor_y_train = tensor_X_train.clone()
    tensor_y_val = tensor_X_val.clone()

    # 5. Ponderación temporal exponencial en entrenamiento
    if gamma < 1.0:
        exponents = torch.arange(n_tr - 1, -1, -1, dtype=torch.float32)
        w_train = gamma ** exponents
        w_train = (w_train / w_train.sum() * n_tr)
    else:
        w_train = torch.ones(n_tr)

    train_ds = TensorDataset(tensor_X_train, tensor_y_train, w_train)
    val_ds = TensorDataset(tensor_X_val, tensor_y_val, torch.ones(n_val))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    tensor_X_full = torch.cat([tensor_X_train, tensor_X_val], dim=0)
    return train_loader, val_loader, tensor_X_full, raw_wins
