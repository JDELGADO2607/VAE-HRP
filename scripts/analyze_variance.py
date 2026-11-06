import sys
import os
sys.path.insert(0, os.path.abspath("src"))

import numpy as np
import pandas as pd
from vae_hrp.data.loader import load_price_data, prepare_log_returns

prices = load_price_data("Data/NUAM_CleanData (1).xlsx")
returns, dropped = prepare_log_returns(prices)
T, N = returns.shape
print(f"Dataset: T={T} días, N={N} activos (descartados con NaN: {len(dropped)})")

# 1. PCA on correlation matrix
corr = returns.corr().values
eigenvalues, _ = np.linalg.eigh(corr)
eigenvalues = np.sort(eigenvalues)[::-1]
explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
cum_explained = np.cumsum(explained_variance_ratio)

print("\n=== 1. Descomposición Espectral (PCA) sobre Matriz de Correlación NUAM ===")
print("k  | Autovalor (lambda) | Varianza (%) | Varianza Acumulada (%)")
print("---|--------------------|--------------|-----------------------")
for k in range(1, 17):
    print(f"{k:2d} | {eigenvalues[k-1]:18.4f} | {explained_variance_ratio[k-1]*100:11.2f}% | {cum_explained[k-1]*100:20.2f}%")

# 2. Marchenko-Pastur on T=1771, N=53
q = T / N
lambda_plus = (1.0 + np.sqrt(1.0 / q))**2
lambda_minus = (1.0 - np.sqrt(1.0 / q))**2
significant_mp = np.sum(eigenvalues > lambda_plus)

print(f"\n=== 2. Teoría de Matrices Aleatorias (Marchenko-Pastur) ===")
print(f"q = T/N = {q:.2f}")
print(f"Límite superior de ruido puro Marchenko-Pastur (lambda+): {lambda_plus:.4f}")
print(f"Autovalores que superan el ruido aleatorio (lambda_i > lambda+): {significant_mp}")
for i in range(min(significant_mp + 4, N)):
    status = "SISTEMÁTICO (Señal)" if eigenvalues[i] > lambda_plus else "RUIDO ALEATORIO (Bulk)"
    print(f"  Factor {i+1:2d}: lambda = {eigenvalues[i]:.4f} -> {status}")

# 3. Criterio Kaiser-Guttman (lambda > 1.0)
kg_factors = np.sum(eigenvalues > 1.0)
print(f"\nCriterio de Kaiser-Guttman (lambda > 1.0): {kg_factors} factores")

# 4. Análisis en Ventanas Rodantes de 252 días (T_w=252, N=53)
print("\n=== 3. Dinámica Espectral en Ventanas Rodantes de 252 Días (T_w=252, N=53) ===")
q_w = 252 / N
lambda_plus_w = (1.0 + np.sqrt(1.0 / q_w))**2
print(f"q_w = 252/53 = {q_w:.2f}")
print(f"Límite lambda+_w para ventana anual = {lambda_plus_w:.4f}")

roll_sig_mp = []
roll_cum_k = {4: [], 6: [], 8: [], 10: [], 12: [], 16: []}

for t_idx in range(252, T, 63):
    w_ret = returns.iloc[t_idx-252:t_idx]
    w_corr = w_ret.corr().values
    w_eig = np.sort(np.linalg.eigvalsh(w_corr))[::-1]
    roll_sig_mp.append(np.sum(w_eig > lambda_plus_w))
    w_cum = np.cumsum(w_eig / np.sum(w_eig))
    for k in roll_cum_k:
        roll_cum_k[k].append(w_cum[k-1])

print(f"Factores significativos promedio (> lambda+): {np.mean(roll_sig_mp):.2f} (min: {np.min(roll_sig_mp)}, max: {np.max(roll_sig_mp)})")
print("Varianza explicada promedio en ventanas rodantes:")
for k, vals in roll_cum_k.items():
    print(f"  k={k:2d}: {np.mean(vals)*100:.2f}% (std: {np.std(vals)*100:.2f}%)")
