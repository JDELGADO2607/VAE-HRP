"""Script de ejecución de la grilla de ablación (k_z x linkage) con DSR y persistencia."""

import sys
from pathlib import Path
import pandas as pd

from vae_hrp.data.loader import load_price_data, prepare_log_returns
from vae_hrp.experiments.ablation import run_ablation_grid
from vae_hrp.metrics.statistical import jobson_korkie_test

sys.stdout.reconfigure(encoding='utf-8')


def main():
    print("=" * 90)
    print("INICIANDO ESTUDIO DE ABLACIÓN VAE-HRP (k_z x Linkage) CON DEFLATED SHARPE RATIO (DSR)")
    print("=" * 90)

    # 1. Cargar datos
    print("Cargando y preparando retornos logarítmicos...")
    prices = load_price_data()
    returns_df, dropped = prepare_log_returns(prices)
    print(f"Dimensiones de retornos: {returns_df.shape} (53 activos, {len(dropped)} excluidos)")

    # 2. Configurar grilla
    # Dimensiones latentes: 4, 6, 8, 10, 12, 16
    latent_dims = [4, 6, 8, 10, 12, 16]
    # Enlaces jerárquicos: ward, average, complete, single
    linkages = ['ward', 'average', 'complete', 'single']

    print(f"Grilla de dimensiones latentes k_z : {latent_dims}")
    print(f"Grilla de enlaces jerárquicos     : {linkages}")
    print(f"Total configuraciones VAE-HRP     : {len(latent_dims) * len(linkages)} (más 4 HRP clásicos y benchmarks)")

    # 3. Ejecutar grilla (con caché automática de matrices en artifacts/covariances/)
    exp = run_ablation_grid(
        returns_df=returns_df,
        latent_dims=latent_dims,
        linkages=linkages,
        min_train_quarters=5,
        n_ensemble=3,       # 3 miembros del ensamble por fold para balance óptimo de velocidad y estabilidad
        mc_samples=100,     # 100 muestras Monte Carlo por miembro
        epochs=25,          # 25 épocas por miembro (batch_size=64)
        force_recompute=False,
    )

    results_df = exp['results_df']
    oos_dict = exp['oos_returns_dict']

    print("\n" + "=" * 105)
    print("TABLA RESUMEN DE ABLACIÓN — DESEMPEÑO FUERA DE MUESTRA Y DEFLATED SHARPE RATIO (DSR)")
    print("=" * 105)
    cols_to_show = ['Portfolio', 'SD', 'SR', 'TOR', 'MD', 'Turnover', 'ENA', 'DSR', 'SR_Star']
    pd.set_option('display.max_columns', 15)
    pd.set_option('display.width', 1000)
    pd.set_option('display.float_format', lambda x: f'{x:.4f}')
    print(results_df[cols_to_show].to_string(index=False))

    # 4. Pruebas por pares Jobson-Korkie
    print("\n" + "=" * 90)
    print("PRUEBAS DE SIGNIFICANCIA POR PARES (JOBSON-KORKIE CON CORRECCIÓN DE MEMMEL)")
    print("=" * 90)

    for link in ['ward', 'average']:
        vae_key = f"VAE-HRP (kz=8, {link})"
        cl_key = f"Classical HRP ({link})"
        if vae_key in oos_dict and cl_key in oos_dict:
            jk = jobson_korkie_test(oos_dict[vae_key], oos_dict[cl_key])
            print(f"\nContraste: {vae_key} vs. {cl_key}")
            print(f"  SR VAE-HRP : {jk['SR_A']:.4f}")
            print(f"  SR Clásico : {jk['SR_B']:.4f}")
            print(f"  Diferencia : {jk['diff_SR']:.4f}")
            print(f"  Estadístico Z : {jk['z_stat']:.4f}")
            print(f"  p-valor (unilateral H1: VAE > Clásico) : {jk['p_value_one_sided']:.4f}")
            print(f"  p-valor (bilateral)                    : {jk['p_value_two_sided']:.4f}")


if __name__ == "__main__":
    main()
