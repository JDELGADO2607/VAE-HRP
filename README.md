# VAE-HRP: Paridad de Riesgo Jerárquica con Autoencoders Variacionales en el Mercado NUAM

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Optuna](https://img.shields.io/badge/Optuna-5.0.0-informational.svg)](https://optuna.org/)

Este repositorio contiene la implementación canónica, base de datos y artículo científico de la metodología **VAE-HRP** (*Variational Autoencoder - Hierarchical Risk Parity*), aplicada a la optimización robusta de portafolios de inversión en el mercado accionario integrado **MSCI NUAM** (Bolsas de Santiago de Chile, Colombia y Lima).

---

## 📌 Resumen de la Investigación

La optimización tradicional de media-varianza (Markowitz, 1952) colapsa en mercados emergentes debido a la inestabilidad de la matriz de covarianza muestral y a la magnificación de errores en su inversión. La metodología de **Paridad de Riesgo Jerárquica (HRP)** (López de Prado, 2016) soluciona la inversión de matrices pero sigue dependiendo de covarianzas empíricas lineales contaminadas por ruido.

Este proyecto propone una extensión profunda no lineal:
1. **Autoencoder Variacional Transversal (Cross-Sectional VAE)**: Comprime los retornos transversales de 53 acciones en un espacio latente de baja dimensión ($k=8$) regularizado mediante la divergencia de Kullback-Leibler.
2. **Deep Ensemble + Simulación Monte Carlo**: Entrena 20 redes independientes por trimestre y realiza 250 simulaciones estocásticas para estimar la covarianza de ensamble esperada.
3. **Restauración Factorial de Rango Completo**: Restaura la varianza idiosincrática de la diagonal mediante la formulación de Chamberlain y Rothschild (1983).
4. **Validación Walk-Forward Causal**: Rebalanceo trimestral a lo largo de 22 trimestres fuera de muestra (2020 a 2026) con decaimiento temporal exponencial ($\gamma = 0.995$) y optimización bayesiana en Optuna sin fuga de datos.

### Resultados Clave Fuera de Muestra (OOS)

| Portafolio | Sharpe Ratio ($r_f=4.62\%$) | Retorno Total (TOR) | Desv. Estándar (SD) | Rotación (Turnover) | ENA |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **B. VAE-HRP (Propuesto)** | **0.5210** | **73.12%** | **10.56%** | **0.4139** | **19.65** |
| **D. HRP Clásico** | 0.4938 | 71.07% | 10.68% | 0.4823 | 21.69 |
| **A. VAE Mínima Varianza** | 0.2781 | 50.36% | **10.11%** | 0.4903 | 9.29 |
| **C. Mínima Varianza Clásica** | -0.0026 | 29.10% | 10.84% | 0.9460 | 6.12 |
| **E. Equiponderado (1/N)** | 0.4070 | 70.54% | 12.81% | 0.0000 | 53.00 |

---

## 📂 Estructura del Repositorio

```
VAE-HRP/ (rama: cetus)
├── Data/
│   └── NUAM_CleanData (1).xlsx    # Precios diarios (2019-2026, 53 activos filtrados)
├── Notebook/
│   └── VAE_documented.ipynb       # Cuaderno Jupyter integral con backtest completo
├── Paper/
│   └── Version_Identificada_Asignacion_Portafolios_NUAM (1).pdf # Paper académico (Univ. del Rosario)
├── AGENTS.md                      # Directrices y normas de ingeniería para agentes de IA
├── CONTEXT.md                     # Marco teórico formal, formulación matemática y resultados
├── HANDOFF.md                     # Estado de entrega, inventario de deuda técnica y roadmap
├── ARCHITECTURE.md                # Flujos de datos, arquitectura VAE y diagramas Mermaid
├── DATA.md                        # Diccionario de activos, desglose geográfico y sectorial
├── README.md                      # Portada y visión general (este archivo)
└── LICENSE                        # Licencia Apache 2.0
```

---

## 🗺️ Mapa de Documentación

- **Para Agentes y Desarrolladores**: Consultar [AGENTS.md](file:///C:/Users/User/orca/workspaces/VAE-HRP/cetus/AGENTS.md) para conocer las invariantes operativas y de codificación.
- **Para Fundamentos Teóricos y Matemáticos**: Consultar [CONTEXT.md](file:///C:/Users/User/orca/workspaces/VAE-HRP/cetus/CONTEXT.md).
- **Para Estado Actual y Roadmap de Desarrollo**: Consultar [HANDOFF.md](file:///C:/Users/User/orca/workspaces/VAE-HRP/cetus/HANDOFF.md).
- **Para Arquitectura de Módulos y Modelos**: Consultar [ARCHITECTURE.md](file:///C:/Users/User/orca/workspaces/VAE-HRP/cetus/ARCHITECTURE.md).
- **Para el Catálogo de Datos y Activos**: Consultar [DATA.md](file:///C:/Users/User/orca/workspaces/VAE-HRP/cetus/DATA.md).

---

## 🚀 Inicio Rápido y Requisitos

### Requisitos del Sistema
- Python 3.10+ (con compatibilidad probada en Windows y Linux).
- Memoria RAM recomendada: $\ge 16$ GB.
- GPU (CUDA) opcional pero acelerará los 20 modelos por fold.

### Instalación de Dependencias
```bash
pip install torch optuna pandas numpy scipy scikit-learn openpyxl matplotlib seaborn pymupdf
```

### Ejecución del Cuaderno
Abrir y ejecutar [`Notebook/VAE_documented.ipynb`](file:///C:/Users/User/orca/workspaces/VAE-HRP/cetus/Notebook/VAE_documented.ipynb) en JupyterLab, VS Code o Google Colab.

---

## 📄 Referencia Académica

Si utilizas este código o metodología en investigaciones académicas, por favor cita el artículo correspondiente:

> **Ramírez-Carrillo, G., Carrillo-García, Ó., Arenas-Contreras, S., & Delgado, J. M.** (2026). *Asignación Robusta de Portafolios en el Mercado NUAM: Una Extensión de la Paridad de Riesgo Jerárquica basada en Autoencoders Variacionales*. Documento de Investigación, Facultad de Economía, Universidad del Rosario.
