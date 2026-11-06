"""Arquitectura del Autoencoder Variacional Transversal (Cross-Sectional VAE)."""

from typing import Tuple
import torch
import torch.nn as nn


class CrossSectionalVAE(nn.Module):
    """Autoencoder Variacional Transversal para modelado no lineal de factores latentes.
    
    Comprime vectores de retornos transversales de dimensión `num_assets` en un espacio
    latente de dimensión `latent_dim`, forzando la captura de factores sistemáticos comunes
    y filtrando el ruido idiosincrático.
    """
    def __init__(
        self,
        num_assets: int = 53,
        latent_dim: int = 8,
        dropout_rate: float = 0.15,
    ):
        super().__init__()
        self.num_assets = num_assets
        self.latent_dim = latent_dim
        self.dropout = nn.Dropout(p=dropout_rate)

        # Encoder: Proyección no lineal hacia representación oculta
        self.encoder_hidden = nn.Sequential(
            nn.Linear(num_assets, 32),
            nn.Tanh(),
        )
        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_log_var = nn.Linear(32, latent_dim)

        # Decoder: Reconstrucción desde el espacio latente hacia el espacio de activos
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.Tanh(),
            nn.Linear(32, num_assets),
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Codifica los retornos de entrada retornando la media y el log-varianza latentes."""
        h = self.encoder_hidden(self.dropout(x))
        return self.fc_mu(h), self.fc_log_var(h)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Aplica el truco de reparametrización z = mu + sigma * epsilon durante el entrenamiento."""
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + std * eps
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decodifica un vector latente hacia la reconstrucción de retornos."""
        return self.decoder(z)

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Paso hacia adelante completo: codificación, muestreo latente y reconstrucción."""
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decode(z)
        return recon, mu, log_var


def compute_vae_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    log_var: torch.Tensor,
    beta: float = 1e-3,
    sample_weights: torch.Tensor = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Calcula la pérdida ELBO negativa: MSE de reconstrucción + beta * Divergencia KL.
    
    Args:
        recon: Tensor reconstruido por el decoder.
        target: Tensor objetivo original.
        mu: Media latente.
        log_var: Logaritmo de varianza latente.
        beta: Ponderación de regularización KL.
        sample_weights: Ponderación temporal de observaciones (opcional).
        
    Returns:
        total_loss: Pérdida combinada.
        recon_loss: Pérdida MSE de reconstrucción.
        kld: Divergencia de Kullback-Leibler.
    """
    if sample_weights is not None:
        # Ponderación temporal de cada ventana en el lote
        recon_loss = ((recon - target) ** 2 * sample_weights[:, None, None]).mean()
    else:
        recon_loss = ((recon - target) ** 2).mean()

    # KL Divergence: -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
    kld = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    total_loss = recon_loss + beta * kld
    return total_loss, recon_loss, kld
