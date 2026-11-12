"""Configuración global y constantes financieras del proyecto VAE-HRP."""

from pathlib import Path

# Constantes del Mercado Financiero
TRADING_DAYS: int = 252
RISK_FREE_RATE: float = 0.0462  # 4.62% anual

# Hiperparámetros de Entrada y Ventanas
WINDOW_LENGTH_S: int = 60       # Longitud de ventana de entrada al VAE (días hábiles)
NUM_ASSETS: int = 53            # Acciones finales tras depuración
DEFAULT_LATENT_DIM: int = 8     # Dimensión del espacio latente
TEMPORAL_DECAY_GAMMA: float = 0.995  # Ponderación exponencial de observaciones pasadas

# Configuración de Entrenamiento y Ensamble
DEFAULT_BATCH_SIZE: int = 32
DEFAULT_ENSEMBLE_SIZE: int = 20
DEFAULT_MC_SAMPLES: int = 250
DEFAULT_EPOCHS: int = 100
DEFAULT_PATIENCE: int = 10

# Rutas Predeterminadas
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_LOCAL_DATA_PATH = PROJECT_ROOT / "Data" / "NUAM_CleanData (1).xlsx"
FALLBACK_REMOTE_DATA_URL = (
    "https://raw.githubusercontent.com/JDELGADO2607/VAE-HRP/"
    "9f700c7b66784b2bfa29b7399bdfa32950bb712d/NUAM_CleanData%20(1).xlsx"
)
