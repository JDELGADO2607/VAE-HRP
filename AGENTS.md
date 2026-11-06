# AGENTS.md — Protocolo Operativo para Agentes y Desarrolladores

Este documento establece las directrices operativas, restricciones arquitectónicas, estándares de codificación y reglas de interacción para cualquier agente de inteligencia artificial (o ingeniero de software) que opere en el repositorio **VAE-HRP** (rama `cetus`).

---

## 1. Identidad y Misión del Repositorio

- **Proyecto**: *VAE-HRP: Paridad de Riesgo Jerárquica con Aprendizaje de Variedad mediante Autoencoder Variacional en el Mercado Integrado NUAM (Chile, Colombia y Perú)*.
- **Origen Académico**: Investigación desarrollada por Gonzalo Ramírez-Carrillo, Óscar Carrillo-García, Smaily Arenas-Contreras y Juan M. Delgado (Facultad de Economía, Universidad del Rosario, 2026).
- **Objetivo Técnico**: Extracción de representaciones latentes no lineales mediante Autoencoders Variacionales (VAE) para construir matrices de covarianza de rango completo libres de ruido idiosincrático y alimentar algoritmos de Paridad de Riesgo Jerárquica (HRP) y Mínima Varianza (MVP) en un protocolo *walk-forward* causal.

---

## 2. Entorno y Ejecución Técnica

### 2.1 Sistema Operativo e Intérprete Python
- **Sistema Operativo**: Windows 11 / Windows Server.
- **Intérprete Principal**: Python 3.10 instalado en el sistema (`C:\Users\User\AppData\Local\Programs\Python\Python310\python.exe` o invocable vía `py -3.10`).
- **Gestor de Paquetes**: `uv` o `pip`.
- **Bibliotecas Críticas**:
  - `torch` (PyTorch 2.13.0+cpu o versión CUDA si está disponible).
  - `optuna` (5.0.0+) para optimización bayesiana de hiperparámetros.
  - `scipy` (1.15.3+) para agrupamiento jerárquico (`linkage`, `dendrogram`) y optimización (`minimize`).
  - `scikit-learn` (1.7.2+) para utilidades de modelado.
  - `pandas` (2.3.3+) y `numpy` (2.2.6+) para cálculo vectorial y manejo de series de tiempo.
  - `openpyxl` para lectura de datos en formato Excel (`.xlsx`).
  - `matplotlib` y `seaborn` para visualizaciones financieras.

### 2.2 Comandos de Ejecución Estándar
- Para ejecutar scripts en este entorno Windows:
  ```powershell
  py -3.10 <script.py>
  ```
- Para verificar dependencias instaladas:
  ```powershell
  py -3.10 -m pip list
  ```

---

## 3. Invariantes del Dominio y Reglas No Negociables

### Invariante 1: Causalidad Estricta y Cero Sesgo de Anticipación (Look-Ahead Bias)
1. **Aislamiento por Fold**: Ningún cálculo en el fold $t$ (entrenamiento o validación) puede incluir precios, retornos o estadísticas de la ventana de prueba fuera de muestra (OOS) $t$.
2. **Preprocesamiento Causal**:
   - Las estadísticas de winsorización MAD (mediana y MAD) se estiman **únicamente** sobre la partición de entrenamiento de cada fold (`_fit_preprocess`).
   - La partición de validación se transforma aplicando las estadísticas del conjunto de entrenamiento (`_apply_preprocess`).
   - Dentro de cada ventana rodante ($S=60$ días), la estandarización $Z$-score se realiza sobre las observaciones de dicha ventana con factor de estabilidad $\epsilon = 10^{-8}$.
   - Nunca debe reintroducirse relleno con ceros (*zero-padding*) sintético al calcular ventanas iniciales.

### Invariante 2: Integridad Matemática de las Fórmulas Clave
1. **Pérdida del VAE (ELBO Negativo)**:
   $$\mathcal{L}(\phi, \theta) = \text{MSE}(X, \hat{X}) + \beta \cdot \text{KL}\left(q_\phi(Z|X) \parallel \mathcal{N}(0, I_k)\right)$$
   donde:
   $$\text{KL} = -0.5 \sum \left(1 + \log(\sigma^2) - \mu^2 - \sigma^2\right)$$
   Tanto el entrenamiento como la parada temprana en validación deben evaluar la misma función de pérdida (MSE para reconstrucción + divergencia KL ponderada por $\beta$).
2. **Reconstrucción Factorial de Covarianza (Chamberlain & Rothschild, 1983)**:
   Dado que el decoder proyecta desde un espacio latente de $k < n$ dimensiones, la covarianza del VAE tiene rango $k$. Para garantizar una matriz de covarianza de rango completo e invertible:
   $$\Sigma_{\text{Completa}} = \Sigma_{\text{VAE}} + \text{diag}\left(\max\left(0, \Sigma_{\text{Empírica}} - \Sigma_{\text{VAE}}\right)\right)$$
   La varianza idiosincrática eliminada en la compresión se restaura en la diagonal.
3. **Métrica de Distancia de HRP**:
   $$d_{i,j} = \sqrt{\frac{1 - \rho_{i,j}}{2}}$$
   donde $\rho_{i,j}$ es la correlación derivada de la matriz de covarianza correspondiente. El enlace jerárquico predeterminado es `average`.

### Invariante 3: Mapeo Coherente de Nombres de Portafolios
El código del notebook original utiliza etiquetas internas heredadas (`dae_only`, `dae_hrp`). Todo nuevo desarrollo, refactorización o reporte debe mapear consistentemente a la nomenclatura oficial del paper:
- `A. VAE Minimum Variance` (o `VAE-MVP`): Mínima varianza con covarianza VAE restaurada.
- `B. VAE-HRP`: Paridad de Riesgo Jerárquica con covarianza VAE restaurada.
- `C. Classical Minimum Variance` (o `MVP Clásica`): Mínima varianza empírica rodante (252 días).
- `D. Classical HRP` (o `HRP Clásica`): Paridad de Riesgo Jerárquica con covarianza empírica rodante (252 días).
- `E. 1/N Equal Weight` (o `Equiponderado`): Asignación uniforme $w_i = 1/53$.

### Invariante 4: Métricas Financieras Oficiales
- Días hábiles anuales: $TRADING\_DAYS = 252$.
- Tasa libre de riesgo anual: $r_f = 0.0462$ ($4.62\%$).
- Métricas auditadas:
  - **SD**: Desviación estándar anualizada ($\sigma \times \sqrt{252}$).
  - **SR**: Sharpe Ratio anualizado $\left(\frac{(1 + r_{oos})^{252/T} - 1 - r_f}{\text{SD}}\right)$.
  - **TOR**: Retorno acumulado total $\prod (1 + r_{oos}) - 1$.
  - **MD**: Máximo drawdown $\min \left(\frac{C_t - \max_{s \le t} C_s}{\max_{s \le t} C_s}\right)$.
  - **Turnover**: Rotación media inter-trimestral $\frac{1}{K-1} \sum_{k=2}^K \sum_{i=1}^n |w_{i,k} - w_{i,k-1}|$.
  - **ENA**: Número Efectivo de Activos $\frac{1}{K} \sum_{k=1}^K \frac{1}{\sum_{i=1}^n w_{i,k}^2}$.

---

## 4. Estructura del Repositorio

```
VAE-HRP/ (rama: cetus)
├── Data/
│   └── NUAM_CleanData (1).xlsx    # Precios diarios de cierre (1772 observaciones, 56 columnas brutas)
├── Notebook/
│   └── VAE_documented.ipynb       # Notebook canónico documentado con corrida de 22 trimestres OOS
├── Paper/
│   └── Version_Identificada_Asignacion_Portafolios_NUAM (1).pdf # Artículo científico completo
├── AGENTS.md                      # Directrices y reglas de ingeniería para agentes (este archivo)
├── CONTEXT.md                     # Marco teórico, matemático y financiero del problema NUAM
├── HANDOFF.md                     # Estado de entrega actual, deuda técnica y roadmap inmediato
├── ARCHITECTURE.md                # Diagramas de flujo y arquitectura de módulos y datos
├── DATA.md                        # Diccionario de activos (53 acciones), sectores y países
├── README.md                      # Portada y guía general del repositorio
└── LICENSE                        # Licencia del proyecto
```

---

## 5. Instrucciones para Tareas Futuras

Cuando se solicite al agente implementar cambios o continuar el proyecto:
1. **Modularización**: Migrar gradualmente el código del notebook monolítico hacia un paquete Python estructurado (ej. `src/vae_hrp/`) con separación de responsabilidades:
   - `src/vae_hrp/data/`: Carga, limpieza y cálculo de retornos logarítmicos.
   - `src/vae_hrp/preprocessing/`: MAD winsorization, z-scoring causal y creación de tensores.
   - `src/vae_hrp/models/`: Arquitectura PyTorch del `CrossSectionalVAE`.
   - `src/vae_hrp/training/`: Loop de entrenamiento con parada temprana y optimizador Optuna.
   - `src/vae_hrp/covariance/`: Simulación Monte Carlo, Deep Ensemble y factor de Chamberlain & Rothschild.
   - `src/vae_hrp/portfolio/`: Optimizadores MVP y algoritmo de HRP (quasi-diagonalización y bipartición).
   - `src/vae_hrp/backtest/`: Orquestador walk-forward expansivo y cálculo de métricas.
2. **Pruebas Automatizadas**: Crear pruebas unitarias con `pytest` en `tests/` para verificar invariantes numéricas (sumas de pesos $\sum w_i = 1$, no negatividad $w_i \ge 0$, simetría y semidefinición positiva de covarianzas reconstruidas).
3. **Persistencia y Caché**: Guardar modelos entrenados por fold y matrices de covarianza intermedias para evitar reentrenamientos costosos cuando sólo se evalúan nuevas estrategias de portafolio.
4. **Git Discipline**: Crear ramas descriptivas para nuevas funcionalidades, no modificar directamente datos crudos y mantener commits atómicos con mensajes en formato estándar (`feat:`, `fix:`, `refactor:`, `docs:`).
