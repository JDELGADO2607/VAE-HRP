# DATA.md — Catálogo y Pipeline de Datos del Mercado NUAM

Este documento describe en detalle las fuentes de datos, los criterios de filtrado y depuración, el universo de activos modelados, la clasificación sectorial y geográfica, y las transformaciones causales aplicadas en el marco **VAE-HRP**.

---

## 1. Fuente y Características Generales

- **Archivo Fuente**: [`Data/NUAM_CleanData (1).xlsx`](file:///C:/Users/User/orca/workspaces/VAE-HRP/cetus/Data/NUAM_CleanData%20(1).xlsx)
- **Universo de Referencia**: Acciones del índice **MSCI NUAM** (mercados bursátiles de Santiago de Chile, Colombia y Lima).
- **Frecuencia**: Diaria (precios de cierre oficiales en moneda local ajustados por dividendos y splits).
- **Horizonte Temporal**: 28 de junio de 2019 al 13 de abril de 2026 (1,772 registros de fechas).
- **Dimensión Inicial**: 1 columna de fechas (`Dates`) + 56 series de precios de activos.

---

## 2. Criterios de Inclusión y Limpieza de Datos

Para evitar sesgos de supervivencia o distorsiones numéricas durante el backtest walk-forward de 2020 a 2026:
1. **Regla de Exclusión por Valores Faltantes**: Cualquier activo con vacíos o ausencias de cotización en cualquier punto del horizonte de estudio fue excluido de la muestra.
2. **Activos Eliminados (3 títulos)**:
   - `PEI CB EQUITY` (Patrimonio Estrategias Inmobiliarias, Colombia) — Ausencia de historial completo.
   - `NUAM CI EQUITY` (Holding Bursátil Regional NUAM, Chile) — Emisión reciente sin historial retrospectivo previo a 2023.
   - `IPCHBC1 PE Equity` (Inversiones Portafolio / Credicorp, Perú) — Series discontinuas en 2019–2020.
3. **Muestra Final Modelada**: **53 activos** sin ningún dato faltante (`Remaining NaNs: 0`) a lo largo de 1,771 retornos diarios calculados.

---

## 3. Composición Geográfica y Sectorial (53 Activos)

### 3.1 Distribución Geográfica

| País | Mercado Bursátil | Sufijo Ticker | Nº de Acciones | Participación (%) |
| :--- | :--- | :---: | :---: | :---: |
| **Chile** | Bolsa de Comercio de Santiago (BCS) | `CI EQUITY` | 34 | 64.2% |
| **Colombia** | Bolsa de Valores de Colombia (BVC) | `CB EQUITY` | 12 | 22.6% |
| **Perú** | Bolsa de Valores de Lima (BVL) | `PE EQUITY` | 7 | 13.2% |
| **Total** | **Holding NUAM** | — | **53** | **100.0%** |

### 3.2 Distribución Sectorial

| Sector GICS | Chile (34) | Colombia (12) | Perú (7) | Total Acciones |
| :--- | :--- | :--- | :--- | :---: |
| **Financiero** | Banco de Chile, BSAN, BCI, Itaú Chile, Quiñenco, ILC, Banco BICE | Grupo Cibest, Cibest Pref, Grupo Sura, Sura Pref, Grupo Aval, Davivienda Pref, Corficolombiana | Credicorp (BAP), IFS | **16** |
| **Materiales** | SQM/B, CMPC, CAP, Oro Blanco | Grupo Argos, Cementos Argos | Southern Copper, Buenaventura, Volcan | **9** |
| **Servicios Públicos (Utilities)** | Enel Américas, Enel Chile, Colbún, Aguas Andinas, Engie Energía Chile (ECL), IAM | ISA, Celsia | — | **8** |
| **Consumo Básico (Staples)** | Cencosud, Embotelladora Andina B, CCU, SMU, Viña Concha y Toro | — | Alicorp | **6** |
| **Industrial** | LATAM Airlines, Empresas Copec, Vapores, SalfaCorp | — | Ferreycorp | **5** |
| **Consumo Discrecional** | Falabella, Ripley, Forus | — | — | **3** |
| **Bienes Raíces (Real Estate)** | Mallplaza, Parque Arauco, Cencosud Shopping | — | — | **3** |
| **Servicios de Comunicación** | Entel Chile | — | — | **1** |
| **Energía** | — | Ecopetrol | — | **1** |
| **Tecnologías de Información** | Sonda | — | — | **1** |
| **Total General** | **34** | **12** | **7** | **53** |

---

## 4. Catálogo Detallado de los 53 Tickers Modelados

| Nº | Ticker Bloomberg / Archivo | Empresa Emisora | País | Sector |
| :---: | :--- | :--- | :---: | :--- |
| 01 | `BAP PE Equity` | Credicorp Ltd. | Perú | Financiero |
| 02 | `SCCO PE EQUITY` | Southern Copper Corporation | Perú | Materiales |
| 03 | `CHILE CI EQUITY` | Banco de Chile | Chile | Financiero |
| 04 | `SQM/B CI EQUITY` | Sociedad Química y Minera de Chile (Serie B) | Chile | Materiales |
| 05 | `PFCIBEST CB EQUITY` | Grupo Cibest (Preferencial) | Colombia | Financiero |
| 06 | `CENCOSUD CI EQUITY` | Cencosud S.A. | Chile | Consumo Básico |
| 07 | `BSAN CI EQUITY` | Banco Santander Chile | Chile | Financiero |
| 08 | `LTM CI EQUITY` | LATAM Airlines Group S.A. | Chile | Industrial |
| 09 | `BCI CI EQUITY` | Banco de Crédito e Inversiones | Chile | Financiero |
| 10 | `CIBEST CB EQUITY` | Grupo Cibest (Ordinaria) | Colombia | Financiero |
| 11 | `COPEC CI EQUITY` | Empresas Copec S.A. | Chile | Industrial |
| 12 | `FALAB CI EQUITY` | Falabella S.A. | Chile | Consumo Discrecional |
| 13 | `ECOPETL CB EQUITY` | Ecopetrol S.A. | Colombia | Energía |
| 14 | `BVN PE EQUITY` | Compañía de Minas Buenaventura S.A.A. | Perú | Materiales |
| 15 | `ISA CB EQUITY` | Interconexión Eléctrica S.A. E.S.P. | Colombia | Servicios Públicos |
| 16 | `ENELAM CI EQUITY` | Enel Américas S.A. | Chile | Servicios Públicos |
| 17 | `CMPC CI EQUITY` | Empresas CMPC S.A. | Chile | Materiales |
| 18 | `ENELCHIL CI EQUITY` | Enel Chile S.A. | Chile | Servicios Públicos |
| 19 | `MALLPLAZ CI EQUITY` | Plaza S.A. (Mallplaza) | Chile | Bienes Raíces |
| 20 | `GRUPOARG CB EQUITY` | Grupo Argos S.A. | Colombia | Materiales |
| 21 | `CEMARGOS CB EQUITY` | Cementos Argos S.A. | Colombia | Materiales |
| 22 | `ANDINAB CI EQUITY` | Embotelladora Andina S.A. (Serie B) | Chile | Consumo Básico |
| 23 | `PARAUCO CI EQUITY` | Parque Arauco S.A. | Chile | Bienes Raíces |
| 24 | `QUINENC CI EQUITY` | Quiñenco S.A. | Chile | Financiero |
| 25 | `IFS PE EQUITY` | Intercorp Financial Services Inc. | Perú | Financiero |
| 26 | `COLBUN CI EQUITY` | Colbún S.A. | Chile | Servicios Públicos |
| 27 | `PFGRUPSUR CB EQUITY` | Grupo Sura (Preferencial) | Colombia | Financiero |
| 28 | `CENCOMAL CI EQUITY` | Cencosud Shopping S.A. | Chile | Bienes Raíces |
| 29 | `ITAUCL CI EQUITY` | Itaú Corpbanca Chile | Chile | Financiero |
| 30 | `CCU CI EQUITY` | Compañía Cervecerías Unidas S.A. | Chile | Consumo Básico |
| 31 | `AGUAS/A CI EQUITY` | Aguas Andinas S.A. (Serie A) | Chile | Servicios Públicos |
| 32 | `VAPORES CI EQUITY` | Compañía Sud Americana de Vapores | Chile | Industrial |
| 33 | `SMU CI EQUITY` | SMU S.A. (Unimarc) | Chile | Consumo Básico |
| 34 | `CONCHA CI EQUITY` | Viña Concha y Toro S.A. | Chile | Consumo Básico |
| 35 | `GRUPOSUR CB EQUITY` | Grupo Sura (Ordinaria) | Colombia | Financiero |
| 36 | `AVAL CB EQUITY` | Grupo Aval Acciones y Valores S.A. | Colombia | Financiero |
| 37 | `PFDAVVND CB EQUITY` | Banco Davivienda S.A. (Preferencial) | Colombia | Financiero |
| 38 | `ECL CI EQUITY` | Engie Energía Chile S.A. | Chile | Servicios Públicos |
| 39 | `FERREYC1 PE EQUITY` | Ferreycorp S.A.A. | Perú | Industrial |
| 40 | `CELSIA CB EQUITY` | Celsia S.A. E.S.P. | Colombia | Servicios Públicos |
| 41 | `ENTEL CI EQUITY` | Empresa Nacional de Telecomunicaciones (Entel) | Chile | Servicios Comunicación |
| 42 | `ALICORC1 PE EQUITY` | Alicorp S.A.A. | Perú | Consumo Básico |
| 43 | `RIPLEY CI EQUITY` | Ripley Corp S.A. | Chile | Consumo Discrecional |
| 44 | `IAM CI EQUITY` | Inversiones Aguas Metropolitanas S.A. | Chile | Servicios Públicos |
| 45 | `ILC CI EQUITY` | Inversiones La Construcción S.A. | Chile | Financiero |
| 46 | `CAP CI EQUITY` | CAP S.A. | Chile | Materiales |
| 47 | `CORFICOL CB EQUITY` | Corporación Financiera Colombiana S.A. | Colombia | Financiero |
| 48 | `SALFACOR CI EQUITY` | SalfaCorp S.A. | Chile | Industrial |
| 49 | `FORUS CI EQUITY` | Forus S.A. | Chile | Consumo Discrecional |
| 50 | `SONDA CI EQUITY` | Sonda S.A. | Chile | Tecnologías Información |
| 51 | `OROB CI EQUITY` | Sociedad de Inversiones Oro Blanco S.A. | Chile | Materiales |
| 52 | `VOLCABC1 PE EQUITY` | Volcan Compañía Minera S.A.A. | Perú | Materiales |
| 53 | `BICE CI EQUITY` | Banco BICE / BICEcorp S.A. | Chile | Financiero |

---

## 5. Pipeline de Transformación y Preprocesamiento Causal

```
Precios Brutos P(t) [1772 x 53]
       │
       ▼
1. Retornos Logarítmicos: r(t) = ln( P(t) / P(t-1) ) [1771 x 53]
       │
       ▼
2. Segmentación en Ventanas Rodantes de Longitud S = 60 días
       │
       ▼
3. Winsorización MAD: clip(r, Mediana ± 5 * MAD)
   (Estadísticas ajustadas ÚNICAMENTE en datos de entrenamiento)
       │
       ▼
4. Estandarización Intra-Ventana: (r - μ) / (σ + 1e-8)
       │
       ▼
5. Ponderación Temporal Exponencial: w(t) = γ^(T-t), γ = 0.995
   (Asignación de pesos en el DataLoader de entrenamiento)
```

- **Ventana $S=60$**: Unidad atómica de entrada al VAE que capta regímenes dinámicos sin sobreajustar ruido de ultra-alta frecuencia.
- **Factor $\gamma = 0.995$**: En el entrenamiento expansivo, las observaciones antiguas reciben un peso ponderado menor, facilitando la adaptación del VAE a los cambios de régimen macroeconómico.
