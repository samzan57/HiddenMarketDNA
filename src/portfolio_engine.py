import numpy as np
import pandas as pd


class PortfolioEngine:
    """
    Construction de portefeuilles à partir de facteurs PCA.

    Stratégie implémentée :
    - Neutralisation du facteur de marché (PC1)
    - Exploitation de la dispersion sectorielle (PC2)
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        loadings: pd.DataFrame,
        target_factor: str = "PC2"
    ):
        """
        Parameters
        ----------
        returns : pd.DataFrame
            Rendements des actifs (index = Date, colonnes = actifs)
        loadings : pd.DataFrame
            Loadings PCA (index = actifs, colonnes = PC1, PC2, ...)
        target_factor : str
            Facteur utilisé pour construire les positions (par défaut PC2)
        """
        self.returns = returns.copy()
        self.loadings = loadings.copy()
        self.target_factor = target_factor

        self._check_inputs()

    def _check_inputs(self):
        if self.target_factor not in self.loadings.columns:
            raise ValueError(f"{self.target_factor} absent des loadings PCA")

        common_assets = self.returns.columns.intersection(self.loadings.index)
        if len(common_assets) < 2:
            raise ValueError("Pas assez d'actifs communs entre returns et loadings")

        self.returns = self.returns[common_assets]
        self.loadings = self.loadings.loc[common_assets]

    def build_market_neutral_weights(self) -> pd.DataFrame:
        """
        Construit des poids market-neutral basés sur le facteur cible.

        Returns
        -------
        pd.DataFrame
            Poids du portefeuille (index = Date, colonnes = actifs)
        """
        factor_exposure = self.loadings[self.target_factor]

        # Normalisation cross-sectionnelle
        weights = factor_exposure / factor_exposure.abs().sum()

        # Répliquer les poids sur toutes les dates
        weights = pd.DataFrame(
            np.tile(weights.values, (len(self.returns), 1)),
            index=self.returns.index,
            columns=self.returns.columns
        )

        return weights

    def compute_portfolio_returns(self, weights: pd.DataFrame) -> pd.Series:
        """
        Calcule les rendements du portefeuille.

        Parameters
        ----------
        weights : pd.DataFrame
            Poids du portefeuille

        Returns
        -------
        pd.Series
            Rendements du portefeuille
        """
        portfolio_returns = (weights * self.returns).sum(axis=1)
        portfolio_returns.name = "Portfolio"
        return portfolio_returns
