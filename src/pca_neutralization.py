# src/factors.py

import numpy as np
import pandas as pd


def project_on_factors(returns: pd.DataFrame, pca_engine) -> pd.DataFrame:
    """
    Projette les rendements sur les facteurs PCA.
    """
    return pca_engine.transform(returns)


def reconstruct_returns(
    factors: pd.DataFrame,
    pca_engine,
    n_components: int = None
) -> pd.DataFrame:
    """
    Reconstruit les rendements à partir des facteurs PCA.
    """
    if not pca_engine.fitted:
        raise RuntimeError("PCAEngine doit être ajusté")

    if n_components is None:
        n_components = factors.shape[1]

    W = pca_engine.get_eigen_portfolios().iloc[:, :n_components].values
    F = factors.iloc[:, :n_components].values

    X_reconstructed = F @ W.T

    # Retour dans l'espace original
    X_reconstructed = pca_engine.scaler.inverse_transform(X_reconstructed)

    return pd.DataFrame(
        X_reconstructed,
        index=factors.index,
        columns=pca_engine.columns_
    )


def neutralize_factors(
    returns: pd.DataFrame,
    factors: pd.DataFrame,
    pca_engine,
    n_components: int = 1
) -> pd.DataFrame:
    """
    Supprime l'exposition aux n premiers facteurs PCA.
    """
    reconstructed = reconstruct_returns(
        factors,
        pca_engine,
        n_components=n_components
    )

    residuals = returns - reconstructed
    return residuals


def extract_residual_signal(
    returns: pd.DataFrame,
    factors: pd.DataFrame,
    pca_engine,
    n_components: int = 1
) -> pd.DataFrame:
    """
    Résidu après neutralisation des facteurs dominants.
    """
    return neutralize_factors(
        returns,
        factors,
        pca_engine,
        n_components=n_components
    )
