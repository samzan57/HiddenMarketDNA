# src/diagnostics.py

import numpy as np
import pandas as pd


def explained_variance_table(pca_engine) -> pd.DataFrame:
    """
    Retourne un tableau de variance expliquée et cumulée.
    """
    var = pca_engine.explained_variance()

    table = pd.DataFrame({
        "Explained Variance": var,
        "Cumulative Variance": var.cumsum()
    })

    return table


def factor_contributions(pca_engine) -> pd.DataFrame:
    """
    Contribution (au sens variance) de chaque actif à chaque composante.
    """
    loadings = pca_engine.get_eigen_portfolios()

    contributions = loadings ** 2
    contributions = contributions.div(contributions.sum(axis=0), axis=1)

    return contributions


def market_dominance_ratio(pca_engine) -> float:
    """
    Mesure la dominance du facteur marché (PC1).
    """
    var = pca_engine.explained_variance()
    return var.iloc[0]


def factor_orthogonality(factors: pd.DataFrame) -> pd.DataFrame:
    """
    Matrice de corrélation empirique des facteurs.
    Doit être proche de l'identité.
    """
    return factors.corr()


def summary_report(pca_engine, factors: pd.DataFrame) -> dict:
    """
    Synthèse des diagnostics PCA.
    """
    report = {
        "explained_variance": explained_variance_table(pca_engine),
        "market_dominance": market_dominance_ratio(pca_engine),
        "orthogonality": factor_orthogonality(factors),
        "contributions": factor_contributions(pca_engine)
    }

    return report
