"""VAE-HRP: Paridad de Riesgo Jerárquica con Autoencoders Variacionales en el Mercado NUAM."""

__version__ = "0.1.0"

from vae_hrp.config import (
    TRADING_DAYS,
    RISK_FREE_RATE,
    WINDOW_LENGTH_S,
    NUM_ASSETS,
    DEFAULT_LATENT_DIM,
)
from vae_hrp.data.loader import load_price_data, prepare_log_returns
from vae_hrp.preprocessing.causal import fit_preprocess, apply_preprocess, build_loaders
from vae_hrp.models.vae import CrossSectionalVAE, compute_vae_loss
from vae_hrp.training.trainer import train_vae, make_vae_objective
from vae_hrp.covariance.monte_carlo import (
    get_expected_vae_covariance,
    restore_idiosyncratic_variance,
)
from vae_hrp.portfolio.hrp import hrp_portfolio, hrp_empirical
from vae_hrp.portfolio.mvp import mvp_weights_from_cov, mvp_weights_long_only
from vae_hrp.metrics.performance import summarise_portfolio
from vae_hrp.backtest.walk_forward import run_walk_forward

__all__ = [
    "__version__",
    "TRADING_DAYS",
    "RISK_FREE_RATE",
    "WINDOW_LENGTH_S",
    "NUM_ASSETS",
    "DEFAULT_LATENT_DIM",
    "load_price_data",
    "prepare_log_returns",
    "fit_preprocess",
    "apply_preprocess",
    "build_loaders",
    "CrossSectionalVAE",
    "compute_vae_loss",
    "train_vae",
    "make_vae_objective",
    "get_expected_vae_covariance",
    "restore_idiosyncratic_variance",
    "hrp_portfolio",
    "hrp_empirical",
    "mvp_weights_from_cov",
    "mvp_weights_long_only",
    "summarise_portfolio",
    "run_walk_forward",
]
