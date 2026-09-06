# src/factor_backtest.py
"""
Backtest multi-facteurs avec détection de régimes avancée (Sprint 2 + 3).

ÉVOLUTION PAR SPRINT :
  Sprint 1 → poids = PC2 loadings | régime = seuil PC1 vol (binaire)
  Sprint 2 → poids = ACP + Momentum + Quality | régime = seuil PC1 vol (binaire)
  Sprint 3 → poids = ACP + Momentum          | régime = HMM + GARCH (continu)

AMÉLIORATION CLÉ DE SPRINT 3 :
  Avant : if pc1_vol > seuil : weights *= 0.5  (tout ou rien)
  Après : weights *= regime_manager.evaluate(in_sample).risk_scale
          → risk_scale varie continûment entre min_scale (0.3) et 1.0
          → moins de faux signaux, moins de turnover, coûts réduits

PARAMÈTRE update_freq :
  HMM + GARCH ne sont refittés que tous les 21 jours (mensuel).
  Entre deux refits, le dernier régime est réutilisé.
  Ce compromis réduit le temps de calcul de ~5x sans perte significative
  de précision (les régimes durent typiquement plusieurs semaines).
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.pca_engine import PCAEngine
from src.factors.momentum import compute_multi_horizon_momentum
from src.factors.quality import compute_quality_score
from src.factors.composite import CompositeSignal, FactorWeights
from src.regimes.regime_manager import RegimeManager
from src.optimization.optimizer import PortfolioOptimizer

logger = logging.getLogger(__name__)


class RollingFactorBacktest:
    """
    Backtest rolling : signal multi-facteurs + régimes HMM/GARCH + optimisation.

    Paramètres
    ----------
    returns : pd.DataFrame
        Rendements log. index=Date, colonnes=actifs.
    window : int
        Taille fenêtre in-sample (252j = 1 an).
    n_components : int
        Composantes PCA à extraire.
    target_factor : str
        Facteur PCA cible ("PC2" = rotation sectorielle).
    factor_weights : FactorWeights, optional
        Poids ACP/Momentum/Quality (défaut : 60/40/0).
    use_advanced_regimes : bool
        True  → HMM + GARCH (Sprint 3)
        False → seuil PC1 vol simple (Sprint 1/2, plus rapide)
    regime_update_freq : int
        Fréquence de refit HMM/GARCH en jours (défaut 21 = mensuel).
    min_risk_scale : float
        Exposition minimale en période de stress extrême (défaut 0.30).
    risk_scale_high_vol : float
        Facteur de réduction si use_advanced_regimes=False (défaut 0.5).
    vol_threshold_quantile : float
        Quantile seuil si use_advanced_regimes=False (défaut 0.75).
    quality_window : int
        Fenêtre du score de qualité (défaut 63j = 3 mois).
    optimizer : PortfolioOptimizer, optional
        Si fourni, optimise les poids au lieu d'utiliser le signal brut.
        None → comportement Sprint 1/2/3 (signal direct, rétrocompatible).
    """

    def __init__(
        self,
        returns:                  pd.DataFrame,
        window:                   int                        = 252,
        n_components:             int                        = 3,
        target_factor:            str                        = "PC2",
        factor_weights:           FactorWeights              = None,
        use_advanced_regimes:     bool                       = True,
        regime_update_freq:       int                        = 21,
        min_risk_scale:           float                      = 0.30,
        risk_scale_high_vol:      float                      = 0.50,
        vol_threshold_quantile:   float                      = 0.75,
        quality_window:           int                        = 63,
        optimizer:                Optional[PortfolioOptimizer] = None,
    ):
        self.returns               = returns
        self.window                = window
        self.n_components          = n_components
        self.target_factor         = target_factor
        self.use_advanced_regimes  = use_advanced_regimes
        self.regime_update_freq    = regime_update_freq
        self.min_risk_scale        = min_risk_scale
        self.risk_scale_high_vol   = risk_scale_high_vol
        self.vol_threshold_quantile = vol_threshold_quantile
        self.quality_window        = quality_window
        self._optimizer            = optimizer

        # Combinateur de signaux factoriels
        self.signal = CompositeSignal(
            weights=factor_weights or FactorWeights(pca=0.60, momentum=0.40, quality=0.00),
            normalize=True,
        )

        # Gestionnaire de régimes (Sprint 3)
        self._regime_manager = RegimeManager(
            min_scale=min_risk_scale,
            update_freq=regime_update_freq,
        )

        # Résultats — remplis après run()
        self._portfolio_returns: pd.Series    = None
        self._weights_history:   pd.DataFrame = None
        self._pc1_vol_series:    pd.Series    = None
        self._regime_series:     pd.Series    = None   # labels de régime par date
        self._risk_scale_series: pd.Series    = None   # risk_scale continu par date
        self._factor_breakdown:  dict         = {}
        self.vol_threshold:      float        = None

    # ------------------------------------------------------------------
    # Exécution
    # ------------------------------------------------------------------

    def run(self) -> None:
        """
        Exécute le backtest rolling.

        Passe 1 : calibre le seuil de vol PC1 (fallback si HMM désactivé)
        Passe 2 : backtest out-of-sample avec signal + régime
        """
        dates = self.returns.index
        n     = len(dates)

        mode = "HMM+GARCH" if self.use_advanced_regimes else "Vol-Threshold"
        logger.info(
            f"[FactorBacktest] Démarrage — {self.returns.shape[1]} actifs, "
            f"{n} jours, fenêtre={self.window}, régime={mode}"
        )

        # Seuil PC1 calibré sur toute la période (uniquement pour le fallback binaire)
        if not self.use_advanced_regimes:
            calib_vols = self._compute_pc1_vols(dates)
            self.vol_threshold = np.quantile(calib_vols, self.vol_threshold_quantile)
        else:
            self.vol_threshold = None

        self._regime_manager.reset()

        # --- Passe 2 : backtest ---
        portfolio_returns = []
        weights_list      = []
        regime_labels     = []
        risk_scales       = []
        pca_contribs      = []
        mom_contribs      = []
        qual_contribs     = []
        pc1_vols_series   = []

        use_quality = self.signal.weights.quality > 0.0
        prev_weights = None

        for t in range(self.window, n - 1):
            in_sample  = self.returns.iloc[t - self.window : t]
            out_sample = self.returns.iloc[t + 1]

            # 1. ACP → loadings PC2
            pca = PCAEngine(n_components=self.n_components)
            pca.fit(in_sample)
            loadings = pca.get_eigen_portfolios()
            pc2_load = loadings[self.target_factor]

            # 2. Signaux factoriels
            mom_scores  = compute_multi_horizon_momentum(in_sample)
            qual_scores = (
                compute_quality_score(in_sample, window=self.quality_window)
                if use_quality
                else pd.Series(0.0, index=in_sample.columns)
            )

            # 3. Signal composite
            breakdown = self.signal.compute_with_breakdown(
                pca_loadings=pc2_load,
                momentum_scores=mom_scores,
                quality_scores=qual_scores,
            )
            signal = breakdown["weights"]

            # 4. Vol PC1 inline (évite une 2e passe séparée)
            pc1_vol = pca.transform(in_sample)["PC1"].std()
            pc1_vols_series.append(pc1_vol)

            # 5. Optimisation des poids (Sprint 4)
            if self._optimizer is not None:
                # Signal factoriel → μ proxy pour l'optimiseur
                weights = self._optimizer.optimize(
                    expected_returns=signal,
                    returns_window=in_sample,
                    prev_weights=prev_weights,
                )
            else:
                weights = signal   # comportement Sprint 1/2/3 direct

            # 6. Régime → risk_scale
            if self.use_advanced_regimes:
                regime_result = self._regime_manager.evaluate(in_sample)
                risk_scale    = regime_result.risk_scale
                label         = regime_result.label
            else:
                risk_scale = self.risk_scale_high_vol if pc1_vol > self.vol_threshold else 1.0
                label      = "stress" if pc1_vol > self.vol_threshold else "calm"

            weights      = weights * risk_scale
            prev_weights = weights   # pour la contrainte de turnover (MVO)

            # 7. Rendement out-of-sample
            aligned = weights.reindex(out_sample.index).fillna(0.0)
            port_ret = float(np.dot(aligned.values, out_sample.values))

            portfolio_returns.append(port_ret)
            weights_list.append(weights)
            regime_labels.append(label)
            risk_scales.append(risk_scale)
            pca_contribs.append(breakdown["pca_contrib"])
            mom_contribs.append(breakdown["mom_contrib"])
            qual_contribs.append(breakdown["qual_contrib"])

        # --- Conversion en pandas ---
        out_index = dates[self.window + 1:]

        self._portfolio_returns = pd.Series(
            portfolio_returns, index=out_index, name="FactorPortfolio"
        )
        self._pc1_vol_series    = pd.Series(pc1_vols_series, index=out_index, name="PC1Vol")
        self._weights_history   = pd.DataFrame(weights_list, index=out_index)
        self._regime_series     = pd.Series(regime_labels, index=out_index, name="Regime")
        self._risk_scale_series = pd.Series(risk_scales,   index=out_index, name="RiskScale")
        self._factor_breakdown  = {
            "pca":      pd.DataFrame(pca_contribs,  index=out_index),
            "momentum": pd.DataFrame(mom_contribs,  index=out_index),
            "quality":  pd.DataFrame(qual_contribs, index=out_index),
        }

        # Stats de régime pour le monitoring
        regime_counts = self._regime_series.value_counts()
        logger.info(
            f"[FactorBacktest] Terminé — {len(self._portfolio_returns)} jours. "
            f"Régimes : {regime_counts.to_dict()}"
        )

    # ------------------------------------------------------------------
    # Accès aux résultats
    # ------------------------------------------------------------------

    def get_results(self):
        """Compatibilité avec RollingPCABacktest — retourne (returns, weights, pc1_vol)."""
        self._check_run()
        return self._portfolio_returns, self._weights_history, self._pc1_vol_series

    def get_regime_series(self) -> pd.Series:
        """Série des labels de régime par date : 'calm' | 'transition' | 'stress'."""
        self._check_run()
        return self._regime_series

    def get_risk_scale_series(self) -> pd.Series:
        """Série des risk_scale continus [min_scale, 1.0] par date."""
        self._check_run()
        return self._risk_scale_series

    def get_factor_breakdown(self) -> dict:
        """Contributions de chaque facteur aux poids du portefeuille."""
        self._check_run()
        return self._factor_breakdown

    def get_annualized_factor_contributions(self) -> pd.Series:
        """Rendement annualisé attribuable à chaque facteur."""
        self._check_run()
        out_returns = self.returns.reindex(self._portfolio_returns.index)
        contribs = {}
        for name, contrib_df in self._factor_breakdown.items():
            factor_rets = (contrib_df * out_returns).sum(axis=1)
            contribs[name] = float(factor_rets.mean() * 252)
        return pd.Series(contribs, name="Annualized contribution")

    # ------------------------------------------------------------------
    # Interne
    # ------------------------------------------------------------------

    def _compute_pc1_vols(self, dates) -> list:
        vols = []
        for t in range(self.window, len(dates)):
            in_sample = self.returns.iloc[t - self.window : t]
            pca = PCAEngine(n_components=self.n_components)
            pca.fit(in_sample)
            vols.append(pca.transform(in_sample)["PC1"].std())
        return vols

    def _check_run(self):
        if self._portfolio_returns is None:
            raise RuntimeError("Appeler run() avant d'accéder aux résultats.")
