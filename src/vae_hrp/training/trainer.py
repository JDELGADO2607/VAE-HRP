"""Entrenamiento del VAE con parada temprana y soporte para Optuna."""

from typing import Optional, Callable
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import optuna

from vae_hrp.models.vae import CrossSectionalVAE, compute_vae_loss


def train_vae(
    model: CrossSectionalVAE,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    beta: float = 1e-3,
    patience: int = 5,
    trial: Optional[optuna.Trial] = None,
    device: Optional[torch.device] = None,
) -> float:
    """Ejecuta el ciclo de entrenamiento del VAE con parada temprana sobre pérdida de validación.
    
    Args:
        model: Instancia de CrossSectionalVAE.
        train_loader: DataLoader con (bx, by, bw).
        val_loader: DataLoader de validación con (bx, by, bw).
        epochs: Número máximo de épocas.
        lr: Tasa de aprendizaje para Adam.
        weight_decay: Regularización L2 para Adam.
        beta: Coeficiente de la divergencia KL.
        patience: Épocas sin mejora antes de detenerse.
        trial: Trial activo de Optuna (para poda de MedianPruner).
        device: Dispositivo de cómputo (CUDA o CPU).
        
    Returns:
        best_val: Mejor pérdida de validación alcanzada.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val = float('inf')
    best_state = None
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        for bx, by, bw in train_loader:
            bx, by, bw = bx.to(device), by.to(device), bw.to(device)
            optimizer.zero_grad()

            recon, mu, log_var = model(bx)
            loss, _, _ = compute_vae_loss(recon, by, mu, log_var, beta=beta, sample_weights=bw)

            loss.backward()
            optimizer.step()

        # Evaluación en validación con la misma función de pérdida (MSE + beta * KL)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by, _ in val_loader:
                bx, by = bx.to(device), by.to(device)
                recon, mu, log_var = model(bx)
                loss_val, _, _ = compute_vae_loss(recon, by, mu, log_var, beta=beta, sample_weights=None)
                val_loss += loss_val.item()

        avg_val = val_loss / max(1, len(val_loader))

        if avg_val < best_val:
            best_val = avg_val
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if trial is not None:
            trial.report(float(avg_val), epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        if epochs_no_improve >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return best_val


def make_vae_objective(
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_assets: int = 53,
    latent_dim: int = 8,
    epochs: int = 50,
    patience: int = 5,
    device: Optional[torch.device] = None,
) -> Callable[[optuna.Trial], float]:
    """Crea la función objetivo de Optuna para optimizar hiperparámetros del VAE."""
    def objective(trial: optuna.Trial) -> float:
        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
        beta = trial.suggest_float('beta', 1e-5, 1e-2, log=True)
        dropout_rate = trial.suggest_float('dropout_rate', 0.05, 0.40)

        model = CrossSectionalVAE(
            num_assets=num_assets,
            latent_dim=latent_dim,
            dropout_rate=dropout_rate,
        )
        val_loss = train_vae(
            model,
            train_loader,
            val_loader,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            beta=beta,
            patience=patience,
            trial=trial,
            device=device,
        )
        return val_loss

    return objective
