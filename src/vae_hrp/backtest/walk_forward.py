"""Orquestador del protocolo Walk-Forward expansivo con rebalanceo trimestral."""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import torch
import optuna
from tqdm import tqdm

from vae_hrp.config import (
    TRADING_DAYS,
    WINDOW_LENGTH_S,
    DEFAULT_ENSEMBLE_SIZE,
    DEFAULT_MC_SAMPLES,
    DEFAULT_LATENT_DIM,
    TEMPORAL_DECAY_GAMMA,
)
from vae_hrp.preprocessing.causal import build_loaders
from vae_hrp.models.vae import CrossSectionalVAE
from vae_hrp.training.trainer import train_vae, make_vae_objective
from vae_hrp.covariance.monte_carlo import (
    get_expected_vae_covariance,
    restore_idiosyncratic_variance,
)
from vae_hrp.portfolio.hrp import hrp_portfolio
from vae_hrp.portfolio.mvp import mvp_weights_from_cov


def run_walk_forward(
    returns_df: pd.DataFrame,
    min_train_quarters: int = 5,
    n_trials: int = 15,
    n_ensemble: int = DEFAULT_ENSEMBLE_SIZE,
    mc_samples: int = DEFAULT_MC_SAMPLES,
    latent_dim: int = DEFAULT_LATENT_DIM,
    gamma: float = TEMPORAL_DECAY_GAMMA,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Ejecuta el protocolo walk-forward completo sobre el DataFrame de retornos.
    
    Args:
        returns_df: DataFrame con 53 columnas de retornos logarítmicos limpios.
        min_train_quarters: Número inicial de trimestres para el primer período de entrenamiento (5).
        n_trials: Ensayos de optimización bayesiana en Optuna por fold.
        n_ensemble: Cantidad de modelos del ensamble profundo por fold (20).
        mc_samples: Cantidad de muestras Monte Carlo por miembro del ensamble (250).
        latent_dim: Dimensión del cuello de botella latente (8).
        gamma: Decaimiento temporal exponencial de observaciones pasadas (0.995).
        device: Dispositivo PyTorch.
        verbose: Si es True, imprime progreso por trimestre.
        
    Returns:
        Diccionario con series OOS concatenadas, historiales de pesos y métricas por fold.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_assets = returns_df.shape[1]
    quarters = pd.date_range(returns_df.index[0], returns_df.index[-1], freq='QS')

    records = {
        'mvp_emp':   {'returns': [], 'weights': []},
        'hrp_emp':   {'returns': [], 'weights': []},
        'vae_mvp':   {'returns': [], 'weights': []},
        'vae_hrp':   {'returns': [], 'weights': []},
        'eq_weight': {'returns': [], 'weights': []},
    }
    fold_summaries = []

    pbar = tqdm(
        range(min_train_quarters, len(quarters) - 1),
        desc='Quarterly Walk-Forward',
        disable=not verbose,
    )

    for q_idx in pbar:
        test_start = quarters[q_idx]
        test_end = quarters[q_idx + 1]

        train_df = returns_df.loc[:test_start].iloc[:-1]
        test_df = returns_df.loc[test_start:test_end]

        if len(test_df) == 0:
            continue

        recent_train_raw = train_df.iloc[-TRADING_DAYS:] if len(train_df) >= TRADING_DAYS else train_df

        # 1. Loaders y preprocesamiento causal sobre ventana expansiva
        train_loader, val_loader, _, _ = build_loaders(
            train_df,
            S=WINDOW_LENGTH_S,
            batch_size=32,
            val_frac=0.15,
            gamma=gamma,
        )

        # 2. Búsqueda bayesiana de hiperparámetros con Optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(
            direction='minimize',
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
            sampler=optuna.samplers.TPESampler(seed=42 + q_idx),
        )
        study.optimize(
            make_vae_objective(
                train_loader,
                val_loader,
                num_assets=num_assets,
                latent_dim=latent_dim,
                epochs=50,
                patience=5,
                device=device,
            ),
            n_trials=n_trials,
        )
        best_p = study.best_params

        # 3. Entrenamiento del Deep Ensemble de M modelos
        ensemble_covs = []
        for m in range(n_ensemble):
            torch.manual_seed(1000 + m)
            member_model = CrossSectionalVAE(
                num_assets=num_assets,
                latent_dim=latent_dim,
                dropout_rate=best_p['dropout_rate'],
            )
            train_vae(
                member_model,
                train_loader,
                val_loader,
                epochs=100,
                lr=best_p['lr'],
                weight_decay=best_p['weight_decay'],
                beta=best_p['beta'],
                patience=10,
                device=device,
            )

            # Estimación Monte Carlo por miembro
            member_cov = get_expected_vae_covariance(
                member_model,
                recent_train_raw,
                S=WINDOW_LENGTH_S,
                num_samples=mc_samples,
                device=device,
            )
            ensemble_covs.append(member_cov)

        # 4. Promedio del ensamble y restauración factorial de Chamberlain & Rothschild
        cov_tr_vae_mc = np.mean(ensemble_covs, axis=0)
        cov_tr_raw = np.cov(recent_train_raw.values.T)
        cov_tr_vae_fr = restore_idiosyncratic_variance(cov_tr_vae_mc, cov_tr_raw)

        cov_vae_df = pd.DataFrame(
            cov_tr_vae_fr,
            index=recent_train_raw.columns,
            columns=recent_train_raw.columns,
        )
        cov_raw_df = pd.DataFrame(
            cov_tr_raw,
            index=recent_train_raw.columns,
            columns=recent_train_raw.columns,
        )

        # 5. Cálculo de ponderaciones para los cinco portafolios
        w_mvp_emp = mvp_weights_from_cov(cov_tr_raw)

        w_hrp_emp_s = hrp_portfolio(cov_raw_df)
        w_hrp_emp = w_hrp_emp_s.reindex(recent_train_raw.columns).fillna(0.0).values
        w_hrp_emp = w_hrp_emp / w_hrp_emp.sum()

        w_vae_mvp = mvp_weights_from_cov(cov_tr_vae_fr)

        w_vae_hrp_s = hrp_portfolio(cov_vae_df)
        w_vae_hrp = w_vae_hrp_s.reindex(recent_train_raw.columns).fillna(0.0).values
        w_vae_hrp = w_vae_hrp_w = w_vae_hrp / w_vae_hrp.sum()

        w_eq = np.ones(num_assets) / num_assets

        # 6. Evaluación OOS
        test_np = test_df.values
        oos_mvp_emp = pd.Series(test_np @ w_mvp_emp, index=test_df.index, name='MVP-Emp')
        oos_hrp_emp = pd.Series(test_np @ w_hrp_emp, index=test_df.index, name='HRP-Emp')
        oos_vae_mvp = pd.Series(test_np @ w_vae_mvp, index=test_df.index, name='VAE-MVP')
        oos_vae_hrp = pd.Series(test_np @ w_vae_hrp, index=test_df.index, name='VAE-HRP')
        oos_eq = pd.Series(test_np @ w_eq, index=test_df.index, name='EqualWeight')

        for key, oos, w in [
            ('mvp_emp',   oos_mvp_emp, w_mvp_emp),
            ('hrp_emp',   oos_hrp_emp, w_hrp_emp),
            ('vae_mvp',   oos_vae_mvp, w_vae_mvp),
            ('vae_hrp',   oos_vae_hrp, w_vae_hrp),
            ('eq_weight', oos_eq,      w_eq),
        ]:
            records[key]['returns'].append(oos)
            records[key]['weights'].append(w)

        fold_row = {
            'quarter': test_start,
            'sd_mvp_emp': float(oos_mvp_emp.std() * np.sqrt(TRADING_DAYS)),
            'sd_hrp_emp': float(oos_hrp_emp.std() * np.sqrt(TRADING_DAYS)),
            'sd_vae_mvp': float(oos_vae_mvp.std() * np.sqrt(TRADING_DAYS)),
            'sd_vae_hrp': float(oos_vae_hrp.std() * np.sqrt(TRADING_DAYS)),
            'sd_eq_weight': float(oos_eq.std() * np.sqrt(TRADING_DAYS)),
            'best_val_loss': float(study.best_value),
        }
        fold_summaries.append(fold_row)

    # 7. Concatenación final de retornos OOS
    oos_stitched = {
        key: pd.concat(records[key]['returns']).sort_index()
        for key in records
    }

    return {
        'oos_returns': oos_stitched,
        'records': records,
        'fold_summaries': pd.DataFrame(fold_summaries),
    }
