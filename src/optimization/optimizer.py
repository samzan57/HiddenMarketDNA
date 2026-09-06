# src/optimization/optimizer.py
"""
Interface unifiée pour les méthodes d'optimisation de portefeuille.

TROIS MÉTHODES DISPONIBLES :

  "equal_weight" — Baseline équipondéré
    Long les N/2 actifs avec le score le plus élevé, short les N/2 avec le plus bas.
    Aucun paramètre, aucune estimation de covariance.
    → Benchmark simple pour valider que l'optimisation apporte de la valeur.

  "risk_parity" — Equal Risk Contribution
    Pondère les positions par l'inverse de leur volatilité.
    Actifs plus risqués → positions plus petites.
    → Moins de drawdowns violents, plus stable en crise.

  "mvo" — Mean-Variance Optimization (Markowitz)
    Maximise l'utilité espérée : w'μ - (λ/2)×w'Σw
    Le signal factoriel joue le rôle de μ.
    → Meilleur ratio rendement/risque théorique, mais plus sensible aux erreurs
    d'estimation de μ (compensé par Ledoit-Wolf sur Σ).

RECOMMANDATION PRATIQUE :
  - Compte demo / premier test → "risk_parity" (robuste, peu de paramètres)
  - Portefeuille institutionnel → "mvo" (plus précis mais besoin de calibration)
  - Validation du signal → "equal_weight" (isole la valeur du signal pur)

COMMENT CHOISIR ?
  Backtest les 3 sur la même période et comparer les Sharpe / Max DD.
  Si "equal_weight" > "mvo", le problème vient de la covariance, pas du signal.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.optimization.constraints import PortfolioConstraints
from src.optimization.mean_variance import MeanVarianceOptimizer
from src.optimization.risk_parity import RiskParityOptimizer

logger = logging.getLogger(__name__)

VALID_METHODS = {"mvo", "risk_parity", "equal_weight"}


class PortfolioOptimizer:
    """
    Sélectionne et délègue à la méthode d'optimisation choisie.

    Paramètres
    ----------
    method : str
        "mvo" | "risk_parity" | "equal_weight"
    constraints : PortfolioConstraints
        Contraintes du portefeuille (partagées par toutes les méthodes).
    risk_aversion : float
        (MVO uniquement) Coefficient d'aversion au risque λ. Défaut 1.0.
    target_vol : float, optional
        (MVO uniquement) Volatilité cible annualisée pour calibrer λ.
    use_ledoit_wolf : bool
        (MVO + Risk Parity) Shrinkage Ledoit-Wolf. Défaut True.
    """

    def __init__(
        self,
        method:          str                  = "risk_parity",
        constraints:     PortfolioConstraints = None,
        risk_aversion:   float                = 1.0,
        target_vol:      Optional[float]      = None,
        use_ledoit_wolf: bool                 = True,
    ):
        if method not in VALID_METHODS:
            raise ValueError(f"method doit être dans {VALID_METHODS}, reçu : {method!r}")

        self.method      = method
        self.constraints = constraints or PortfolioConstraints.long_short_standard()

        if method == "mvo":
            self._impl = MeanVarianceOptimizer(
                risk_aversion=risk_aversion,
                target_vol=target_vol,
                use_ledoit_wolf=use_ledoit_wolf,
            )
        elif method == "risk_parity":
            self._impl = RiskParityOptimizer(use_ledoit_wolf=use_ledoit_wolf)
        else:
            self._impl = None   # equal_weight : pas d'objet dédié

        logger.info(f"[PortfolioOptimizer] Méthode = {method}")

    # ------------------------------------------------------------------
    # Interface principale
    # ------------------------------------------------------------------

    def optimize(
        self,
        expected_returns: pd.Series,
        returns_window:   pd.DataFrame,
        prev_weights:     Optional[pd.Series] = None,
    ) -> pd.Series:
        """
        Calcule les poids optimaux.

        Paramètres
        ----------
        expected_returns : pd.Series
            Signal factoriel (scores normalisés, index = actifs).
            Utilisé comme μ en MVO, ou comme indicateur de direction en risk parity.
        returns_window : pd.DataFrame
            Fenêtre de rendements historiques (pour la covariance).
        prev_weights : pd.Series, optional
            Poids précédents (pour contrainte de turnover en MVO).

        Retourne
        --------
        pd.Series
            Poids du portefeuille (index = actifs).
        """
        if self.method == "equal_weight":
            return self._equal_weight(expected_returns)

        return self._impl.optimize(
            expected_returns=expected_returns,
            returns_window=returns_window,
            constraints=self.constraints,
            prev_weights=prev_weights,
        )

    # ------------------------------------------------------------------
    # Equal-weight baseline
    # ------------------------------------------------------------------

    def _equal_weight(self, signal: pd.Series) -> pd.Series:
        """
        Long les N/2 actifs au meilleur score, short les N/2 au pire.
        Chaque position a le même poids absolu = 1/N.

        C'est le "benchmark signal pur" : 100% de la performance vient
        du signal factoriel, zéro optimisation de portefeuille.
        Si ce benchmark > méthodes sophistiquées → problème d'estimation.
        """
        n       = len(signal)
        ranked  = signal.rank()
        weights = pd.Series(0.0, index=signal.index)

        long_mask  = ranked > n / 2
        short_mask = ranked <= n / 2

        n_long  = long_mask.sum()
        n_short = short_mask.sum()

        if n_long > 0:
            weights[long_mask]  =  1.0 / n_long
        if n_short > 0:
            weights[short_mask] = -1.0 / n_short

        # Market-neutral si demandé (corrige le déséquilibre avec N impair)
        if self.constraints.market_neutral:
            weights = weights - weights.mean()

        # Normalise en levier unitaire (sum(|w|) = 1)
        abs_sum = weights.abs().sum()
        if abs_sum > 1e-10:
            weights = weights / abs_sum

        return weights.clip(
            lower=self.constraints.min_position,
            upper=self.constraints.max_position,
        )
