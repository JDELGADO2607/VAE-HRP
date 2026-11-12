"""Pruebas unitarias para la arquitectura CrossSectionalVAE y función de pérdida."""

import pytest
import torch

from vae_hrp.models.vae import CrossSectionalVAE, compute_vae_loss


def test_cross_sectional_vae_forward():
    model = CrossSectionalVAE(num_assets=53, latent_dim=8, dropout_rate=0.1)
    model.eval()

    # Entrada de un lote de ventanas: [B, S, A]
    B, S, A = 4, 60, 53
    x = torch.randn(B, S, A)

    recon, mu, log_var = model(x)

    assert recon.shape == (B, S, A)
    assert mu.shape == (B, S, 8)
    assert log_var.shape == (B, S, 8)
    assert not torch.isnan(recon).any()


def test_compute_vae_loss():
    recon = torch.randn(4, 60, 53)
    target = torch.randn(4, 60, 53)
    mu = torch.randn(4, 60, 8)
    log_var = torch.zeros(4, 60, 8)
    weights = torch.ones(4)

    total_loss, recon_loss, kld = compute_vae_loss(
        recon, target, mu, log_var, beta=1e-3, sample_weights=weights
    )

    assert total_loss.item() > 0.0
    assert recon_loss.item() > 0.0
    assert kld.item() >= 0.0
