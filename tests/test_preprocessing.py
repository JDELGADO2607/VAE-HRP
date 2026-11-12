"""Pruebas unitarias para preprocesamiento causal."""

import pytest
import torch
import numpy as np
import pandas as pd

from vae_hrp.preprocessing.causal import fit_preprocess, apply_preprocess, build_loaders


def test_fit_and_apply_preprocess_shapes():
    N, S, A = 10, 60, 53
    sample_windows = torch.randn(N, S, A)
    
    median, mad = fit_preprocess(sample_windows)
    assert median.shape == (1, 1, A)
    assert mad.shape == (1, 1, A)
    assert torch.all(mad >= 1e-8)

    processed = apply_preprocess(sample_windows, median, mad)
    assert processed.shape == (N, S, A)
    assert not torch.isnan(processed).any()


def test_build_loaders():
    dates = pd.date_range('2021-01-01', periods=300, freq='B')
    assets = [f'Asset_{i}' for i in range(53)]
    synthetic_returns = pd.DataFrame(
        np.random.normal(0, 0.01, size=(300, 53)),
        index=dates,
        columns=assets,
    )

    train_loader, val_loader, full_tensor, raw_wins = build_loaders(
        synthetic_returns,
        S=60,
        batch_size=16,
        val_frac=0.15,
        gamma=0.995,
    )

    assert len(train_loader) > 0
    assert len(val_loader) > 0
    assert full_tensor.shape[1] == 60
    assert full_tensor.shape[2] == 53
