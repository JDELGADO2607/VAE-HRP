# HANDOFF.md — Estado de Entrega y Continuidad del Proyecto

Este documento detalla el estado exacto del repositorio **VAE-HRP** al momento de esta entrega, inventaría los artefactos existentes, documenta la deuda técnica y define la hoja de ruta priorizada para los siguientes pasos de desarrollo.

---

## 1. Resumen Ejecutivo del Estado del Repositorio

- **Rama Activa**: `cetus` (derivada de `origin/main`).
- **Estado de Trabajo**: Limpio (Working tree clean).
- **Entorno Validado**: Python 3.10 ejecutado en Windows con dependencias completas (`torch 2.13.0`, `optuna 5.0.0`, `pandas 2.3.3`, `scipy 1.15.3`, `scikit-learn 1.7.2`, `openpyxl 3.1.5`, `matplotlib 3.10.9`, `seaborn 0.13.2`, `pymupdf 1.28.2`).
- **Artefactos Principales Validados**:
  1. `Data/NUAM_CleanData (1).xlsx`: Base de datos histórica limpia con 1772 filas y 56 activos originales (1771 retornos diarios para 53 activos filtrados sin NaNs).
  2. `Notebook/VAE_documented.ipynb`: Cuaderno Jupyter con código completo, documentación celda por celda y la ejecución íntegra del backtest walk-forward (22 trimestres OOS completados con figuras y métricas reproducidas).
  3. `Paper/Version_Identificada_Asignacion_Portafolios_NUAM (1).pdf`: Artículo de investigación formal (25 páginas) con la formulación teórica, análisis empírico y tablas de resultados.
  4. `LICENSE`: Apache License 2.0.

---

## 2. Inventario de Hallazgos y Deuda Técnica

### 2.1 Ruta de Datos Remota vs. Local
- **Situación**: La celda 5 del notebook define:
  ```python
  DATA_PATH = 'https://raw.githubusercontent.com/JDELGADO2607/VAE-HRP/9f700c7b66784b2bfa29b7399bdfa32950bb712d/NUAM_CleanData%20(1).xlsx'
  ```
- **Riesgo**: Dependencia de conexión externa y disponibilidad del enlace GitHub raw.
- **Acción Requerida**: Implementar fallback local hacia `Data/NUAM_CleanData (1).xlsx` en caso de fallo de red o ejecución offline.

### 2.2 Nomenclatura Interna Heredada (`dae_only` / `dae_hrp`)
- **Situación**: En fases tempranas de investigación el modelo fue concebido como un Autoencoder con Denoising (DAE); al evolucionar a Autoencoder Variacional (VAE), los diccionarios internos del notebook mantuvieron las llaves `'dae_only'` y `'dae_hrp'`.
- **Mitigación Actual**: Mapeado en `PORTFOLIO_META` hacia `A. VAE Minimum Variance` y `B. VAE-HRP`.
- **Acción Requerida**: En la modularización a Python, usar explícitamente `vae_mvp` y `vae_hrp`.

### 2.3 Ausencia de Modularidad y Suite de Pruebas
- **Situación**: Toda la lógica algorítmica reside en un único archivo Jupyter Notebook de más de 3,200 líneas. No existen módulos Python independientes (`.py`) ni tests con `pytest`.
- **Riesgo**: Dificultad para automatizar corridas, realizar integración continua o reutilizar componentes en producción o investigación extendida.

### 2.4 Costo Computacional y Falta de Checkpointing
- **Situación**: La ejecución completa del walk-forward realiza:
  - 22 pliegues trimestrales.
  - 15 ensayos de optimización bayesiana Optuna por pliegue.
  - Entrenamiento de un Deep Ensemble de 20 redes VAE por pliegue.
  - Muestreo Monte Carlo de 250 trayectorias por modelo.
  - Tiempo total estimado: ~1.5 a 2 horas en CPU.
- **Acción Requerida**: Implementar un sistema de guardado en disco (`checkpoints/` o `artifacts/`) para matrices de covarianza y pesos trimestrales generados, permitiendo reanalizar o modificar algoritmos de asignación (HRP vs MVP) en segundos sin tener que reentrenar las redes neuronales.

### 2.5 Ausencia de `.gitignore` en la Raíz
- **Situación**: No existía archivo `.gitignore`, lo que dejaba el repositorio expuesto al rastreo accidental de carpetas de caché de Python (`__pycache__`), puntos de control de Jupyter (`.ipynb_checkpoints`) y archivos de sistema.
- **Acción Tomada**: Se provee `.gitignore` estructurado para proyectos de Python y Machine Learning.

---

## 3. Hoja de Ruta y Estado de Avance

```
[Fase 1: Contexto e Higiene] (COMPLETADA)
  ├── AGENTS.md, CONTEXT.md, HANDOFF.md, ARCHITECTURE.md, DATA.md, README.md
  └── .gitignore

[Fase 2: Modularización del Código] (COMPLETADA)
  ├── Estructura src/vae_hrp/
  │     ├── config.py         (constantes TRADING_DAYS=252, rf=0.0462, S=60, etc.)
  │     ├── data/             (cargador local con fallback remoto, retornos log)
  │     ├── preprocessing/    (winsorización MAD causal, normalización z-score)
  │     ├── models/           (arquitectura CrossSectionalVAE en PyTorch)
  │     ├── training/         (loop de entrenamiento, parada temprana, Optuna)
  │     ├── covariance/       (Monte Carlo, ensamble profundo, factor de Chamberlain)
  │     ├── portfolio/        (optimizador MVP con restricciones, algoritmo HRP)
  │     ├── metrics/          (métricas financieras auditadas: SD, SR, TOR, MD, Turnover, ENA)
  │     └── backtest/         (ejecutor walk-forward con decaimiento temporal y métricas)
  ├── Cuaderno canónico preservado íntegro: Notebook/VAE_documented.ipynb
  └── pyproject.toml formal

[Fase 3: Pruebas y Verificación Automatizada] (COMPLETADA)
  ├── tests/test_data.py          (100% pasando: carga local, 53 activos limpios)
  ├── tests/test_preprocessing.py (100% pasando: ajuste sin fuga, forma de ventanas)
  ├── tests/test_vae.py           (100% pasando: forward pass, pérdida ELBO)
  ├── tests/test_covariance.py    (100% pasando: rango completo, simetría, PSD)
  ├── tests/test_portfolio.py     (100% pasando: HRP y MVP sum(w)==1, w>=0)
  └── tests/test_metrics.py       (100% pasando: consistencia numérica y cálculo ENA/Turnover)
  └── Resultado de ejecución: 13 passed in 16.5s

[Fase 4: Checkpointing y Aceleración] (PENDIENTE)
  ├── Guardar matrices de covarianza por fold en formatos eficientes (.parquet / .npz)
  └── Permitir re-optimización instantánea de estrategias alternativas (ej. HERC)
```

---

## 4. Guía Rápida para el Siguiente Agente

Para retomar el trabajo de inmediato:
1. **Consulta de Reglas**: Lee [AGENTS.md](file:///C:/Users/User/orca/workspaces/VAE-HRP/cetus/AGENTS.md) para conocer las restricciones causales y numéricas.
2. **Consulta de Dominio**: Lee [CONTEXT.md](file:///C:/Users/User/orca/workspaces/VAE-HRP/cetus/CONTEXT.md) para comprender la formulación matemática y el universo NUAM.
3. **Consulta de Datos**: Lee [DATA.md](file:///C:/Users/User/orca/workspaces/VAE-HRP/cetus/DATA.md) para verificar el mapeo de los 53 activos.
4. **Ejecución Local**: Utiliza siempre el comando `py -3.10 <comando>` en esta estación de trabajo.
5. **Pruebas Automatizadas**: Para correr la suite completa:
   ```powershell
   $env:PYTHONPATH="src"; py -3.10 -m pytest tests -v
   ```

---

## 5. Estado de los Tres Aspectos Debatidos y Resueltos

### Aspecto 1: Justificación de la Dimensión Latente ($k_z = 8$) — RESUELTO
- Se ejecutó una grilla de ablación factorial completa para $k_z \in \{4, 6, 8, 10, 12, 16\}$ combinada con los 4 métodos de enlace (\textit{Ward, Average, Complete, Single}) sobre 22 trimestres OOS.
- **Hallazgo Empírico**: $k_z = 8$ ofrece la máxima estabilidad topológica transversal (Sharpes homogéneos entre $0.5269$ y $0.5558$ en todos los enlaces). A partir de $k_z = 16$, el rendimiento decae sistemáticamente a $0.47 - 0.52$, confirmando que dimensiones superiores memorizan ruido idiosincrático.

### Aspecto 2: Expansión de la Tabla 5 — RESUELTO
- Se generó el documento LaTeX formal [`Paper/Reporte_Tablas_Ablacion_DSR.tex`](file:///C:/Users/User/orca/workspaces/VAE-HRP/cetus/Paper/Reporte_Tablas_Ablacion_DSR.tex) con la Tabla 5 expandida reportando las 5 estrategias completas (VAE-MVP, VAE-HRP, MVP Clásica, HRP Clásico, 1/N) y la matriz completa de 31 estrategias.

### Aspecto 3: Significancia Estadística y Deflated Sharpe Ratio — RESUELTO
- Implementado en [`src/vae_hrp/metrics/statistical.py`](file:///C:/Users/User/orca/workspaces/VAE-HRP/cetus/src/vae_hrp/metrics/statistical.py).
- Para $N = 31$ estrategias evaluadas, el umbral del azar es $SR_0^* = 0.2082$. VAE-HRP supera ampliamente este umbral alcanzando un $DSR = 0.8006$ (vs. $0.7887$ de HRP Clásico y $0.3462$ de MVP Clásica).
- Integrado al cuaderno canónico [`Notebook/VAE_documented.ipynb`](file:///C:/Users/User/orca/workspaces/VAE-HRP/cetus/Notebook/VAE_documented.ipynb) en la Sección 18.
4. **Ejecución Local**: Utiliza siempre el comando `py -3.10 <comando>` en esta estación de trabajo.
