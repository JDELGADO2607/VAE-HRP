"""Experimento de ablación: Grilla de dimensionalidad latente (k_z) y enlaces de clustering (linkage).

Evalúa combinaciones de k_z in {4, 6, 8, 10, 12, 16} con enlaces {'ward', 'average', 'complete', 'single'}
utilizando bisección recursiva pura en HRP, con persistencia y cálculo del Deflated Sharpe Ratio (DSR).
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from vae_hrp.config import (
    TRADING_DAYS,
    RISK_FREE_RATE,
    WINDOW_LENGTH_S,
    TEMPORAL_DECAY_GAMMA,
)
from vae_hrp.preprocessing.causal import build_loaders
from vae_hrp.models.vae import CrossSectionalVAE
from vae_hrp.training.trainer import train_vae
from vae_hrp.covariance.monte_carlo import (
    get_expected_vae_covariance,
    restore_idiosyncratic_variance,
)
from vae_hrp.portfolio.hrp import hrp_portfolio
from vae_hrp.portfolio.mvp import mvp_weights_from_cov
from vae_hrp.metrics.performance import summarise_portfolio
from vae_hrp.metrics.statistical import (
    compute_moments,
    deflated_sharpe_ratio,
    jobson_korkie_test,
)


def get_cache_dir() -> Path:
    """Retorna y asegura la existencia del directorio de caché de matrices."""
    cache_dir = Path("artifacts") / "covariances"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def compute_or_load_covariances(
    returns_df: pd.DataFrame,
    latent_dims: List[int] = [4, 6, 8, 10, 12, 16],
    min_train_quarters: int = 5,
    n_ensemble: int = 5,
    mc_samples: int = 150,
    epochs: int = 35,
    batch_size: int = 64,
    device: Optional[torch.device] = None,
    force_recompute: bool = False,
) -> Dict[str, Any]:
    """Genera y cachea las matrices de covarianza por fold para cada dimensión latente y para la empírica."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cache_dir = get_cache_dir()
    num_assets = returns_df.shape[1]
    quarters = pd.date_range(returns_df.index[0], returns_df.index[-1], freq='QS')
    valid_q_indices = list(range(min_train_quarters, len(quarters) - 1))

    # Estructura de almacenamiento: cov_cache[kz][q_idx] -> pd.DataFrame
    vae_cov_cache: Dict[int, Dict[int, pd.DataFrame]] = {kz: {} for kz in latent_dims}
    emp_cov_cache: Dict[int, pd.DataFrame] = {}
    test_splits: Dict[int, pd.DataFrame] = {}

    print(f"--- FASE 1: Caché de Matrices de Covarianza ({len(valid_q_indices)} trimestres, k_z={latent_dims}) ---")

    for q_idx in tqdm(valid_q_indices, desc="Procesando Trimestres"):
        test_start = quarters[q_idx]
        test_end = quarters[q_idx + 1]

        train_df = returns_df.loc[:test_start].iloc[:-1]
        test_df = returns_df.loc[test_start:test_end]
        if len(test_df) == 0:
            continue

        test_splits[q_idx] = test_df
        recent_train_raw = train_df.iloc[-TRADING_DAYS:] if len(train_df) >= TRADING_DAYS else train_df

        # 1. Covarianza empírica (252 días)
        emp_file = cache_dir / f"cov_emp_q{q_idx}.npy"
        if emp_file.exists() and not force_recompute:
            cov_tr_raw = np.load(emp_file)
        else:
            cov_tr_raw = np.cov(recent_train_raw.values.T)
            np.save(emp_file, cov_tr_raw)

        emp_cov_cache[q_idx] = pd.DataFrame(
            cov_tr_raw, index=recent_train_raw.columns, columns=recent_train_raw.columns
        )

        # 2. Covarianzas VAE para cada dimensión kz
        train_loader, val_loader, _, _ = build_loaders(
            train_df, S=WINDOW_LENGTH_S, batch_size=batch_size, val_frac=0.15, gamma=TEMPORAL_DECAY_GAMMA
        )

        for kz in latent_dims:
            vae_file = cache_dir / f"cov_vae_kz{kz}_q{q_idx}.npy"
            if vae_file.exists() and not force_recompute:
                cov_fr = np.load(vae_file)
            else:
                ensemble_covs = []
                for m in range(n_ensemble):
                    torch.manual_seed(1000 + m + kz * 100)
                    model = CrossSectionalVAE(num_assets=num_assets, latent_dim=kz, dropout_rate=0.15)
                    train_vae(
                        model,
                        train_loader,
                        val_loader,
                        epochs=epochs,
                        lr=1e-3,
                        weight_decay=1e-5,
                        beta=2e-4,
                        patience=5,
                        device=device,
                    )
                    m_cov = get_expected_vae_covariance(
                        model, recent_train_raw, S=WINDOW_LENGTH_S, num_samples=mc_samples, device=device
                    )
                    ensemble_covs.append(m_cov)

                cov_mc = np.mean(ensemble_covs, axis=0)
                cov_fr = restore_idiosyncratic_variance(cov_mc, cov_tr_raw)
                np.save(vae_file, cov_fr)

            vae_cov_cache[kz][q_idx] = pd.DataFrame(
                cov_fr, index=recent_train_raw.columns, columns=recent_train_raw.columns
            )

    return {
        'vae_cov_cache': vae_cov_cache,
        'emp_cov_cache': emp_cov_cache,
        'test_splits': test_splits,
        'valid_q_indices': valid_q_indices,
    }


def run_ablation_grid(
    returns_df: pd.DataFrame,
    latent_dims: List[int] = [4, 6, 8, 10, 12, 16],
    linkages: List[str] = ['ward', 'average', 'complete', 'single'],
    min_train_quarters: int = 5,
    n_ensemble: int = 5,
    mc_samples: int = 150,
    epochs: int = 35,
    force_recompute: bool = False,
) -> Dict[str, Any]:
    """Ejecuta la grilla de ablación completa sobre todas las combinaciones (k_z, linkage) con bisección pura."""
    cov_data = compute_or_load_covariances(
        returns_df=returns_df,
        latent_dims=latent_dims,
        min_train_quarters=min_train_quarters,
        n_ensemble=n_ensemble,
        mc_samples=mc_samples,
        epochs=epochs,
        force_recompute=force_recompute,
    )

    vae_cov_cache = cov_data['vae_cov_cache']
    emp_cov_cache = cov_data['emp_cov_cache']
    test_splits = cov_data['test_splits']
    valid_q_indices = cov_data['valid_q_indices']

    print(f"\n--- FASE 2: Asignación HRP con Bisección Recursiva sobre Grilla ---")

    results_list = []
    oos_returns_dict = {}

    # 1. Evaluar VAE-HRP sobre la grilla kz x linkage
    for kz in latent_dims:
        for link in linkages:
            strat_label = f"VAE-HRP (kz={kz}, {link})"
            q_returns = []
            weights_hist = []

            for q_idx in valid_q_indices:
                cov_df = vae_cov_cache[kz][q_idx]
                test_df = test_splits[q_idx]

                w_s = hrp_portfolio(cov_df, linkage_method=link)
                w = w_s.reindex(test_df.columns).fillna(0.0).values
                w = w / w.sum()

                ret_oos = pd.Series(test_df.values @ w, index=test_df.index)
                q_returns.append(ret_oos)
                weights_hist.append(w)

            full_oos = pd.concat(q_returns).sort_index()
            oos_returns_dict[strat_label] = full_oos

            summary = summarise_portfolio(strat_label, full_oos, weights_hist)
            moments = compute_moments(full_oos)
            summary['kz'] = kz
            summary['linkage'] = link
            summary['model_type'] = 'VAE-HRP'
            summary['skewness'] = moments['skewness']
            summary['kurtosis'] = moments['kurtosis']
            results_list.append(summary)

    # 2. Evaluar Classical HRP sobre los 4 linkages
    for link in linkages:
        strat_label = f"Classical HRP ({link})"
        q_returns = []
        weights_hist = []

        for q_idx in valid_q_indices:
            cov_df = emp_cov_cache[q_idx]
            test_df = test_splits[q_idx]

            w_s = hrp_portfolio(cov_df, linkage_method=link)
            w = w_s.reindex(test_df.columns).fillna(0.0).values
            w = w / w.sum()

            ret_oos = pd.Series(test_df.values @ w, index=test_df.index)
            q_returns.append(ret_oos)
            weights_hist.append(w)

        full_oos = pd.concat(q_returns).sort_index()
        oos_returns_dict[strat_label] = full_oos

        summary = summarise_portfolio(strat_label, full_oos, weights_hist)
        moments = compute_moments(full_oos)
        summary['kz'] = None
        summary['linkage'] = link
        summary['model_type'] = 'Classical HRP'
        summary['skewness'] = moments['skewness']
        summary['kurtosis'] = moments['kurtosis']
        results_list.append(summary)

    # 3. Evaluar Portafolios de Control (VAE-MVP kz=8, Classical MVP, 1/N)
    for model_name, cov_dict, mtype in [
        ("VAE Minimum Variance (kz=8)", vae_cov_cache[8], "VAE-MVP"),
        ("Classical Minimum Variance", emp_cov_cache, "Classical MVP"),
    ]:
        q_returns = []
        weights_hist = []
        for q_idx in valid_q_indices:
            cov_df = cov_dict[q_idx]
            test_df = test_splits[q_idx]
            w = mvp_weights_from_cov(cov_df.values)
            ret_oos = pd.Series(test_df.values @ w, index=test_df.index)
            q_returns.append(ret_oos)
            weights_hist.append(w)

        full_oos = pd.concat(q_returns).sort_index()
        oos_returns_dict[model_name] = full_oos
        summary = summarise_portfolio(model_name, full_oos, weights_hist)
        moments = compute_moments(full_oos)
        summary['kz'] = 8 if 'kz=8' in model_name else None
        summary['linkage'] = 'N/A'
        summary['model_type'] = mtype
        summary['skewness'] = moments['skewness']
        summary['kurtosis'] = moments['kurtosis']
        results_list.append(summary)

    # 4. Equiponderado (1/N)
    q_returns = []
    weights_hist = []
    num_assets = returns_df.shape[1]
    w_eq = np.ones(num_assets) / num_assets
    for q_idx in valid_q_indices:
        test_df = test_splits[q_idx]
        ret_oos = pd.Series(test_df.values @ w_eq, index=test_df.index)
        q_returns.append(ret_oos)
        weights_hist.append(w_eq)

    full_oos = pd.concat(q_returns).sort_index()
    oos_returns_dict["1/N Equal Weight"] = full_oos
    summary = summarise_portfolio("1/N Equal Weight", full_oos, weights_hist)
    moments = compute_moments(full_oos)
    summary['kz'] = None
    summary['linkage'] = 'N/A'
    summary['model_type'] = 'Benchmark 1/N'
    summary['skewness'] = moments['skewness']
    summary['kurtosis'] = moments['kurtosis']
    results_list.append(summary)

    results_df = pd.DataFrame(results_list)

    # --- FASE 3: Cálculo del Deflated Sharpe Ratio (DSR) sobre toda la grilla ---
    all_srs = results_df['SR'].values
    T_samples = len(next(iter(oos_returns_dict.values())))

    dsr_values = []
    sr_star_values = []
    for _, row in results_df.iterrows():
        dsr_val, sr_star = deflated_sharpe_ratio(
            observed_sr_ann=row['SR'],
            trials_ann_srs=all_srs,
            n_samples=T_samples,
            skewness=row['skewness'],
            kurtosis=row['kurtosis'],
        )
        dsr_values.append(dsr_val)
        sr_star_values.append(sr_star)

    results_df['DSR'] = dsr_values
    results_df['SR_Star'] = sr_star_values

    # Guardar resultados en CSV
    output_csv = Path("artifacts") / "ablation_results.csv"
    results_df.to_csv(output_csv, index=False)
    print(f"\n✓ Resultados consolidados guardados en {output_csv}")

    return {
        'results_df': results_df,
        'oos_returns_dict': oos_returns_dict,
    }
