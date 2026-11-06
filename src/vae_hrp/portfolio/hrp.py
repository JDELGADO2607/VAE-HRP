"""Algoritmo de Paridad de Riesgo Jerárquica (Hierarchical Risk Parity - HRP)."""

from typing import List
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform, pdist


def build_distance_matrix(
    cov_matrix: pd.DataFrame,
    use_abs: bool = False,
) -> np.ndarray:
    """Calcula la matriz de distancia basada en correlación: d = sqrt((1 - rho)/2)."""
    vols = np.sqrt(np.maximum(np.diag(cov_matrix.values), 1e-12))
    corr_clip = (cov_matrix.values / np.outer(vols, vols)).clip(-1.0, 1.0)
    
    if use_abs:
        d_corr = np.sqrt(np.maximum(1.0 - np.abs(corr_clip), 0.0))
    else:
        d_corr = np.sqrt(np.maximum(0.5 * (1.0 - corr_clip), 0.0))
        
    np.fill_diagonal(d_corr, 0.0)
    return d_corr


def get_quasi_diag(link: np.ndarray) -> List[int]:
    """Reordena los activos según la jerarquía del dendrograma (cuasi-diagonalización)."""
    link = link.astype(int)
    sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
    num_items = link[-1, 3]
    
    while sort_ix.max() >= num_items:
        sort_ix.index = pd.RangeIndex(0, sort_ix.shape[0] * 2, 2)
        df0 = sort_ix[sort_ix >= num_items]
        i, j = df0.index, df0.values - num_items
        sort_ix[i] = link[j, 0]
        df1 = pd.Series(link[j, 1], index=i + 1)
        sort_ix = pd.concat([sort_ix, df1]).sort_index()
        sort_ix.index = pd.RangeIndex(sort_ix.shape[0])
        
    return sort_ix.tolist()


def getIVP(cov_matrix: pd.DataFrame) -> np.ndarray:
    """Calcula las ponderaciones de paridad de varianza inversa intra-clúster."""
    epsilon = 1e-8
    ivp = 1.0 / (np.diag(cov_matrix.values) + epsilon)
    return ivp / ivp.sum()


def getClusterVar(cov_matrix: pd.DataFrame, items: List[str]) -> float:
    """Calcula la varianza agregada de un clúster ponderada por varianza inversa."""
    sub_cov = cov_matrix.loc[items, items].astype(float)
    w_ = getIVP(sub_cov).reshape(-1, 1)
    return float(np.dot(np.dot(w_.T, sub_cov.values), w_).item())


def getRecBipart(cov_matrix: pd.DataFrame, ordered_items: List[str]) -> pd.Series:
    """Aplica la bipartición recursiva asignando capital top-down por paridad inversa de riesgo."""
    w = pd.Series(1.0, index=ordered_items)
    cItems = [ordered_items]
    
    while len(cItems) > 0:
        cItems = [
            i[j:k]
            for i in cItems
            for j, k in ((0, len(i) // 2), (len(i) // 2, len(i)))
            if len(i) > 1
        ]
        for i in range(0, len(cItems), 2):
            c0, c1 = cItems[i], cItems[i + 1]
            var0 = getClusterVar(cov_matrix, c0)
            var1 = getClusterVar(cov_matrix, c1)
            denom = var0 + var1
            alpha = 0.5 if denom < 1e-10 else 1.0 - var0 / denom
            w.loc[c0] *= alpha
            w.loc[c1] *= 1.0 - alpha
            
    return w


def hrp_portfolio(
    cov_matrix: pd.DataFrame,
    linkage_method: str = 'ward',
    use_abs_corr: bool = False,
) -> pd.Series:
    """Genera los pesos óptimos de portafolio mediante Paridad de Riesgo Jerárquica."""
    d_mat = build_distance_matrix(cov_matrix, use_abs=use_abs_corr)
    d_condensed = squareform(d_mat, checks=False)
    link = linkage(d_condensed, method=linkage_method)
    sort_ix = get_quasi_diag(link)
    ordered = cov_matrix.columns[sort_ix].tolist()
    w = getRecBipart(cov_matrix, ordered)
    return w / w.sum()


# Alias compatible con la nomenclatura del notebook
hrp_empirical = hrp_portfolio
