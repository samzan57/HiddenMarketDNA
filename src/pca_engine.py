# src/pca_engine.py

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class PCAEngine:
    """
    Moteur PCA pour l'analyse factorielle des rendements.
    """

    def __init__(self, n_components=None):
        self.n_components = n_components

        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components)

        self.fitted = False

    def fit(self, returns: pd.DataFrame):
        """
        Ajuste la PCA sur les rendements.
        """
        if not isinstance(returns, pd.DataFrame):
            raise TypeError("returns doit être un pandas DataFrame")

        X = returns.values

        X_scaled = self.scaler.fit_transform(X)
        self.pca.fit(X_scaled)

        self.fitted = True
        self.columns_ = returns.columns
        self.index_ = returns.index

        return self

    def transform(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Projette les rendements sur les composantes principales.
        """
        if not self.fitted:
            raise RuntimeError("PCAEngine doit être ajusté avant transform()")

        X = returns.values
        X_scaled = self.scaler.transform(X)
        factors = self.pca.transform(X_scaled)

        factor_names = [f"PC{i+1}" for i in range(factors.shape[1])]

        return pd.DataFrame(
            factors,
            index=returns.index,
            columns=factor_names
        )

    def get_eigen_portfolios(self) -> pd.DataFrame:
        """
        Retourne les portefeuilles propres (loadings).
        """
        if not self.fitted:
            raise RuntimeError("PCAEngine doit être ajusté avant extraction")

        loadings = self.pca.components_.T

        factor_names = [f"PC{i+1}" for i in range(loadings.shape[1])]

        return pd.DataFrame(
            loadings,
            index=self.columns_,
            columns=factor_names
        )

    def explained_variance(self) -> pd.Series:
        """
        Variance expliquée par composante.
        """
        if not self.fitted:
            raise RuntimeError("PCAEngine doit être ajusté")

        return pd.Series(
            self.pca.explained_variance_ratio_,
            index=[f"PC{i+1}" for i in range(len(self.pca.explained_variance_ratio_))]
        )
