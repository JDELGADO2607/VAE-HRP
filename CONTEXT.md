# CONTEXT.md — Marco Teórico, Metodológico y Financiero

Este documento detalla el trasfondo económico, la formulación matemática y el diseño experimental del marco **VAE-HRP** aplicado al mercado accionario integrado **MSCI NUAM**.

---

## 1. El Problema de Investigación y el Mercado NUAM

### 1.1 Contexto del Mercado Integrado NUAM
El holding regional **NUAM** (Holding Bursátil Regional) integra las tres principales bolsas de valores de la región andina:
- **Bolsa de Comercio de Santiago (Chile)**
- **Bolsa de Valores de Colombia (Colombia)**
- **Bolsa de Valores de Lima (Perú)**

A diferencia de iniciativas previas como el **MILA** (Mercado Integrado Latinoamericano) —que operó únicamente mediante convenios de enrutamiento con fragmentación tributaria, regulatoria y operativa—, **NUAM** establece una fusión corporativa y tecnológica con una infraestructura única de negociación, compensación y liquidación.

### 1.2 Retos para la Asignación de Activos
1. **Heterogeneidad Monetaria y Cambiaria**: Cotizaciones en Pesos Chilenos (CLP), Pesos Colombianos (COP), Soles Peruanos (PEN) y Dólares Estadounidenses (USD).
2. **Asimetría de Liquidez y Profundidad**: Mercado chileno con mayor profundidad financiera frente a mercados con menor flotante libre y presencia transaccional.
3. **Quiebres Estructurales**: Período histórico (2019–2026) caracterizado por choques sistémicos extremos:
   - Protestas sociales en Chile y Colombia (2019–2021).
   - Choque global y parada súbita por COVID-19 (2020).
   - Choques inflacionarios globales y alzas de tasas de interés de bancos centrales (2022–2023).
   - Ciclos políticos locales y reconfiguraciones macroeconómicas.

### 1.3 La Falla de la Optimización Tradicional (Markowitz MPT / MVO)
El marco clásico de Media-Varianza de Harry Markowitz (1952) busca ponderaciones óptimas resolviendo:
$$\min_w w^T \Sigma w \quad \text{s.t.} \quad \sum_{i=1}^n w_i = 1, \quad w_i \ge 0$$

**Limitación crítica**: Cuando el número de activos $n$ se aproxima o supera el número de observaciones temporales $T$ ($n \approx T$), la matriz de covarianza muestral $\Sigma$ se vuelve mal condicionada o casi singular (número de condición elevado).
- La inversión de la matriz $\Sigma^{-1}$ actúa como un **maximizador de errores de estimación** (Michaud, 1989; López de Prado, 2016).
- Pequeñas perturbaciones estadísticas en los datos empíricos generan fluctuaciones gigantescas en las ponderaciones calculadas, rotación excesiva (*turnover* prohibitivo) y colapso del desempeño fuera de muestra (*out-of-sample*, OOS).

---

## 2. Paridad de Riesgo Jerárquica (HRP)

Introducido por Marcos López de Prado (2016), el algoritmo **HRP** reemplaza la inversión de la matriz de covarianza por un enfoque basado en teoría de grafos y agrupamiento jerárquico no supervisado.

### 2.1 Fases del Algoritmo HRP Clásico

```
Matriz de Covarianza Σ 
       │
       ▼
1. Matriz de Distancia d(i,j) = √[(1 - ρ(i,j))/2]
       │
       ▼
2. Agrupamiento Jerárquico (Árbol/Dendrograma con enlace 'average')
       │
       ▼
3. Cuasi-Diagonalización (Reordenamiento de filas y columnas)
       │
       ▼
4. Bipartición Recursiva (Asignación top-down por paridad inversa de varianza)
```

1. **Definición de Distancia**:
   $$d_{i,j} = \sqrt{\frac{1 - \rho_{i,j}}{2}} \in [0, 1]$$
   donde $\rho_{i,j}$ es el coeficiente de correlación de Pearson.
2. **Agrupamiento de Clústeres (Tree Clustering)**:
   Se construye una matriz de enlace (*linkage matrix*) mediante aglomeración jerárquica con método de vinculación promedio (*average linkage*).
3. **Cuasi-Diagonalización**:
   Se reordenan los activos siguiendo el recorrido del árbol, de modo que activos altamente correlacionados queden contiguos a lo largo de la diagonal principal.
4. **Bipartición Recursiva (Recursive Bisection)**:
   Para cada bifurcación entre dos subclústeres $C_1$ y $C_2$, se calcula la varianza ponderada del clúster:
   $$V_k = w_k^T \Sigma_k w_k \quad \text{donde} \quad w_k = \frac{\text{diag}(\Sigma_k)^{-1}}{\text{Tr}(\text{diag}(\Sigma_k)^{-1})}$$
   El factor de división del capital entre ambos clústeres es:
   $$\alpha_1 = 1 - \frac{V_1}{V_1 + V_2}, \quad \alpha_2 = \frac{V_1}{V_1 + V_2}$$
   El capital se reparte de forma recursiva hasta llegar a los activos individuales.

### 2.2 La Brecha de Investigación
Aunque HRP no invierte matrices, sigue dependiendo de una matriz empírica $\Sigma$ construida a partir de correlaciones lineales de Pearson sobre retornos históricos. En mercados emergentes como NUAM, las covarianzas empíricas sufren de:
- Contaminación por ruido idiosincrático transitorio.
- Incapacidad para modelar dependencias no lineales y dinámicas de cola gruesa.
- Inestabilidad temporal ante cambios de régimen.

---

## 3. La Solución Propuesta: VAE-HRP

La metodología propuesta utiliza un **Autoencoder Variacional Transversal (Cross-Sectional VAE)** como un **modelo factorial no lineal de variables latentes estocásticas**.

### 3.1 Arquitectura del CrossSectionalVAE
Para un vector de retornos estandarizados $x_t \in \mathbb{R}^n$ en el instante $t$:
1. **Perturbación Estocástica (Input Dropout)**:
   Con probabilidad $p_{\text{drop}} \in [0.05, 0.40]$, ciertas entradas se anulan aleatoriamente durante el entrenamiento para inducir representaciones invariantes a ruido idiosincrático.
2. **Encoder**:
   $$h = \tanh(W_e x + b_e) \quad (h \in \mathbb{R}^{32})$$
   $$\mu = W_\mu h + b_\mu \quad (\mu \in \mathbb{R}^k)$$
   $$\log \sigma^2 = W_\sigma h + b_\sigma \quad (\log \sigma^2 \in \mathbb{R}^k)$$
   donde la dimensión latente se fija en $k = 8$ (o se busca entre 8 y 16).
3. **Truco de Reparametrización (Reparameterization Trick)**:
   $$z = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I_k)$$
4. **Decoder**:
   $$h_d = \tanh(W_d z + b_d) \quad (h_d \in \mathbb{R}^{32})$$
   $$\hat{x} = W_o h_d + b_o \quad (\hat{x} \in \mathbb{R}^n)$$

### 3.2 Función de Pérdida (Negativo del ELBO)
$$\mathcal{L}(\phi, \theta) = \frac{1}{n} \sum_{i=1}^n (x_i - \hat{x}_i)^2 + \beta \cdot \text{KL}\left(q_\phi(z|x) \parallel \mathcal{N}(0, I_k)\right)$$
donde la divergencia de Kullback-Leibler cerrada para gaussianas es:
$$\text{KL} = -\frac{1}{2} \sum_{j=1}^k \left(1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2\right)$$
El coeficiente $\beta \in [10^{-5}, 10^{-2}]$ modula el equilibrio entre fidelidad de reconstrucción y regularización del espacio latente.

---

## 4. Reconstrucción de Covarianza y Ensamble

### 4.1 Simulación Monte Carlo de Covarianza Esperada
Dado que el decoder es una función no lineal, la covarianza de los retornos reconstruidos bajo la distribución posterior no se puede obtener analíticamente. Se extraen $S = 250$ muestras latentes:
$$z^{(s)} \sim \mathcal{N}(\mu, \sigma^2), \quad s = 1, \dots, S$$
Cada muestra se decodifica, se des-estandariza y se calcula su matriz de covarianza muestral:
$$\hat{\Sigma}_{\text{MC}} = \frac{1}{S} \sum_{s=1}^S \text{Cov}(\hat{r}^{(s)})$$

### 4.2 Deep Ensemble ($M = 20$ modelos)
Para erradicar la variabilidad proveniente de inicializaciones estocásticas de la red:
- Se entrenan $M = 20$ modelos VAE independientes con semillas aleatorias distintas sobre los mejores hiperparámetros de Optuna.
- La matriz del ensamble es el promedio:
  $$\hat{\Sigma}_{\text{VAE}} = \frac{1}{M} \sum_{m=1}^M \hat{\Sigma}_{\text{MC}}^{(m)}$$

### 4.3 Restauración de Varianza Idiosincrática (Chamberlain & Rothschild, 1983)
Dado que $k = 8 \ll n = 53$, $\hat{\Sigma}_{\text{VAE}}$ tiene rango como máximo 8 y es singular. Se aplica la descomposición factorial clásica:
$$\Sigma_{\text{Completa}} = \hat{\Sigma}_{\text{VAE}} + \text{diag}\left(\max\left(0, \Sigma_{\text{Empírica}} - \hat{\Sigma}_{\text{VAE}}\right)\right)$$
- **Componente Sistemática**: Proporcionada por $\hat{\Sigma}_{\text{VAE}}$ (bajo rango, filtrada de ruido).
- **Componente Específica**: La diagonal que preserva la varianza individual no explicada por los factores latentes.
- **Resultado**: Matriz simétrica, estrictamente definida positiva y con número de condición controlado.

---

## 5. Diseño Experimental y Protocolo Walk-Forward

### 5.1 Ventanas y Rebalanceo
- **Ventana de Entrada al VAE**: $S = 60$ días de negociación (captura dinámicas de corto plazo sin sobreajuste).
- **Ventana de Entrenamiento del VAE**: Expansiva (acumula todo el historial histórico disponible hasta el inicio del trimestre OOS) con decaimiento temporal exponencial:
  $$w_t = \gamma^{T-t}, \quad \gamma = 0.995$$
- **Ventana de Modelos Base Empíricos**: Ventana rodante fija de 252 días (1 año bursátil) sin ponderación temporal.
- **Rebalanceo**: Trimestral (trimestres naturales).
- **Período OOS**: 22 trimestres consecutivos (desde el 01 de octubre de 2020 hasta el 13 de abril de 2026).

### 5.2 Optimización Bayesiana de Hiperparámetros (Optuna)
- **Algoritmo**: Tree-structured Parzen Estimator (TPE).
- **Podador**: `MedianPruner` (warmup de 10 épocas, mínimo 5 ensayos).
- **Iteraciones**: 15 ensayos por fold.
- **Espacio de Búsqueda**:
  - `lr`: $[1 \times 10^{-4}, 1 \times 10^{-2}]$ (log-uniforme)
  - `weight_decay`: $[1 \times 10^{-6}, 1 \times 10^{-3}]$ (log-uniforme)
  - `beta`: $[1 \times 10^{-5}, 1 \times 10^{-2}]$ (log-uniforme)
  - `pdrop`: $[0.05, 0.40]$ (uniforme)

---

## 6. Resultados Empíricos Fuera de Muestra (OOS)

Consolidación de las 22 ventanas trimestrales evaluadas de forma causal:

| Portafolio | Desv. Estándar (SD) | Sharpe Ratio (SR, $r_f=4.62\%$) | Retorno Total (TOR) | Máximo Drawdown (MD) | Rotación (Turnover) | ENA (Nº Efectivo Activos) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **B. VAE-HRP (Propuesto)** | **0.1056** | **0.5210** | **0.7312** | **-0.2232** | **0.4139** | **19.65** |
| **D. Classical HRP (Base)** | 0.1068 | 0.4938 | 0.7107 | -0.2116 | 0.4823 | 21.69 |
| **A. VAE Minimum Variance** | **0.1011** | 0.2781 | 0.5036 | -0.2800 | 0.4903 | 9.29 |
| **C. Classical Min Variance** | 0.1084 | -0.0026 | 0.2910 | -0.3102 | 0.9460 | 6.12 |
| **E. 1/N Equal Weight** | 0.1281 | 0.4070 | 0.7054 | -0.2676 | 0.0000 | 53.00 |

### 6.1 Principales Conclusiones Empíricas
1. **VAE-HRP vs. HRP Clásico**:
   - El Sharpe Ratio se incrementa de **0.4938** a **0.5210** (+5.5%).
   - El Retorno Total aumenta de **71.07%** a **73.12%**.
   - La rotación del portafolio disminuye de **0.4823** a **0.4139** (-14.2%), lo cual reduce drásticamente los costos de transacción e impacto de mercado.
2. **Patología de Markowitz Evidenciada**:
   - La Mínima Varianza Clásica presenta el peor desempeño ajustado por riesgo (Sharpe de -0.0026), una concentración extrema (ENA = 6.12 activos) y una rotación insostenible del 94.6% trimestral.
   - En contraste, el VAE con restauración factorial regulariza la matriz de covarianza, permitiendo que la Mínima Varianza VAE reduzca la desviación estándar al **10.11%** (la menor de todo el estudio) y recorte el turnover a la mitad (0.4903).
3. **Robustez y Estabilidad**:
   - VAE-HRP combina la mejor ratio Sharpe y el menor turnover entre todas las estrategias dinámicas evaluadas, confirmando la hipótesis de que la extracción de representaciones latentes estocásticas no lineales mejora la topología jerárquica de asignación en mercados emergentes integrados.
