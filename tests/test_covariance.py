"""Pruebas unitarias para restauración factorial de covarianza."""

import pytest
import numpy as np

from vae_hrp.covariance.monte_carlo import restore_idiosyncratic_variance


def test_restore_idiosyncratic_variance():
    np.random.seed(42)
    n = 53
    # Simular una covarianza de rango bajo k=8
    V = np.random.randn(n, 8)
    cov_vae = V @ V.T
    
    # Covarianza empírica con ruido adicional
    cov_empirical = cov_vae + np.diag(np.random.uniform(0.1, 0.5, size=n))

    restored = restore_idiosyncratic_variance(cov_vae, cov_empirical)

    # 1. Verificar dimensiones
    assert restored.shape == (n, n)
    
    # 2. Verificar simetría
    np.testing.assert_allclose(restored, restored.T, atol=1e-8)

    # 3. Verificar semidefinición positiva (todos los autovalores >= 0)
    eigenvalues = np.linalg.eigvalsh(restored)
    assert np.all(eigenvalues > -1e-8)

    # 4. Verificar rango completo
    rank = np.linalg.matrix_rank(restored)
    assert rank == n
