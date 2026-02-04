
import numpy as np
import pandas as pd

from src.pca_engine import PCAEngine
from src.portfolio_engine import PortfolioEngine


class RollingPCABacktest:
    """
    Backtest avec ACP recalculée sur fenêtre glissante et détection de régimes via PC1.
    Ajuste dynamiquement l'exposition en fonction du risque.
    Fournit également la série pc1_vol_series pour visualisation des régimes.
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        window: int = 252,
        n_components: int = 3,
        target_factor: str = "PC2",
        risk_scale_high_vol: float = 0.5,  # proportion des poids en marché volatil
        vol_threshold_quantile: float = 0.75  # quantile pour définir marché volatil
    ):
        self.returns = returns
        self.window = window
        self.n_components = n_components
        self.target_factor = target_factor

        self.risk_scale_high_vol = risk_scale_high_vol
        self.vol_threshold_quantile = vol_threshold_quantile

        self.weights_history = []
        self.portfolio_returns = []
        self.pc1_vol_series = None
        self.vol_threshold = None

    def run(self):
        """
        Exécute le backtest rolling PCA avec détection des régimes
        et enregistre la volatilité PC1 pour visualisation.
        """
        dates = self.returns.index

        # 1 Calcul des volatilités de PC1 sur toutes les fenêtres
        pc1_vols_list = []

        for t in range(self.window, len(dates)):
            in_sample = self.returns.iloc[t - self.window:t]

            pca = PCAEngine(n_components=self.n_components)
            pca.fit(in_sample)

            factors = pca.transform(in_sample)
            pc1_vols_list.append(factors["PC1"].std())

        self.vol_threshold = np.quantile(pc1_vols_list, self.vol_threshold_quantile)

        # Convertir en Series pour alignement avec dates out-of-sample
        self.pc1_vol_series = pd.Series(
            pc1_vols_list[:-1],  # exclut dernier, car pas d'out_sample pour ce point
            index=dates[self.window + 1:]
        )

        # 2 Rolling backtest avec ajustement dynamique des poids
        for t in range(self.window, len(dates) - 1):
            in_sample = self.returns.iloc[t - self.window:t]
            out_sample = self.returns.iloc[t + 1]

            pca = PCAEngine(n_components=self.n_components)
            pca.fit(in_sample)
            loadings = pca.get_eigen_portfolios()

            factors = pca.transform(in_sample)
            pc1_vol = factors["PC1"].std()

            # Construction portefeuille
            engine = PortfolioEngine(
                returns=in_sample,
                loadings=loadings,
                target_factor=self.target_factor
            )

            weights = engine.build_market_neutral_weights().iloc[0]

            # Ajustement dynamique des poids
            if pc1_vol > self.vol_threshold:
                weights *= self.risk_scale_high_vol

            # Rendement out-of-sample
            port_ret = np.dot(weights.values, out_sample.values)

            self.weights_history.append(weights)
            self.portfolio_returns.append(port_ret)

        # Conversion en DataFrame / Series
        self.weights_history = pd.DataFrame(
            self.weights_history,
            index=dates[self.window + 1:]
        )

        self.portfolio_returns = pd.Series(
            self.portfolio_returns,
            index=dates[self.window + 1:],
            name="Rolling Portfolio"
        )

    def get_results(self):
        """
        Retourne :
        - rolling_returns : pd.Series
        - rolling_weights : pd.DataFrame
        - pc1_vol_series : pd.Series
        """
        return self.portfolio_returns, self.weights_history, self.pc1_vol_series
