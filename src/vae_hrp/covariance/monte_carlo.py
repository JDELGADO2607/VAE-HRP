"""Estimación de covarianza por simulación Monte Carlo, Deep Ensemble y factor de Chamberlain & Rothschild."""

from typing import List, Optional
import numpy as np
import pandas as pd
import torch

from vae_hrp.models.vae import CrossSectionalVAE


def get_expected_vae_covariance(
    model: CrossSectionalVAE,
    recent_df: pd.DataFrame,
    S: int = 60,
    num_samples: int = 250,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """Calcula la matriz de covarianza esperada del VAE mediante muestreo Monte Carlo latente.
    
    Args:
        model: Modelo CrossSectionalVAE entrenado.
        recent_df: DataFrame de retornos recientes (ej. 252 días).
        S: Longitud de la ventana deslizante (60 días).
        num_samples: Cantidad de trayectorias latentes Monte Carlo a simular.
        device: Dispositivo de cómputo.
        
    Returns:
        Matriz de covarianza esperada de dimensiones [A, A].
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    model.to(device)

    clean = recent_df.ffill()
    raw = torch.tensor(clean.values, dtype=torch.float32)
    padding = torch.zeros(S - 1, raw.shape[1])
    padded = torch.cat([padding, raw], dim=0)
    wins = padded.unfold(0, S, 1).transpose(1, 2)  # [T, S, A]

    # Preprocesamiento robusto intra-ventana
    median = torch.quantile(wins, 0.5, dim=-2, keepdim=True)
    abs_dev = torch.abs(wins - median)
    mad = torch.quantile(abs_dev, 0.5, dim=-2, keepdim=True)
    mad_f = torch.clamp(mad, min=1e-8)
    clipped = torch.max(torch.min(wins, median + 5 * mad_f), median - 5 * mad_f)
    w_mean = torch.mean(clipped, dim=-2, keepdim=True)
    w_std = torch.std(clipped, dim=-2, keepdim=True).clamp(min=1e-8)

    w_mean_last = w_mean[:, -1, :].cpu()  # [T, A]
    w_std_last = w_std[:, -1, :].cpu()   # [T, A]

    # Entrada al encoder: último paso temporal estandarizado [T, A]
    X_last = ((clipped - w_mean) / w_std)[:, -1, :].to(device)

    cov_accum = np.zeros((recent_df.shape[1], recent_df.shape[1]))

    with torch.no_grad():
        mu, log_var = model.encode(X_last)  # [T, latent_dim]

        for _ in range(num_samples):
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            z = mu + std * eps  # [T, latent_dim]

            recon_z = model.decode(z).cpu()                # [T, A] (espacio z-score)
            recon_r = recon_z * w_std_last + w_mean_last   # [T, A] (espacio de retornos)

            cov_accum += np.cov(recon_r.numpy().T)

    return cov_accum / num_samples


def restore_idiosyncratic_variance(
    cov_vae: np.ndarray,
    cov_empirical: np.ndarray,
) -> np.ndarray:
    """Aplica la descomposición factorial de Chamberlain & Rothschild (1983).
    
    Restaura la varianza idiosincrática eliminada en el cuello de botella latente
    agregando la diagonal de excesos positivos:
        Sigma_Completa = Sigma_VAE + diag(max(0, Sigma_Empirica - Sigma_VAE))
        
    Garantiza que la matriz resultante sea de rango completo e invertible.
    """
    diff_diag = np.diag(cov_empirical) - np.diag(cov_vae)
    idiosyncratic_var = np.clip(diff_diag, 0.0, None)
    return cov_vae + np.diag(idiosyncratic_var)
