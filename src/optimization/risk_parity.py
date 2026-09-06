# src/optimization/risk_parity.py
"""
Risk Parity — Equal Risk Contribution (ERC).

PRINCIPE :
  Le portefeuille traditionnel (60/40) alloue 60% en capital aux actions,
  mais les actions contribuent ~90% du risque total (car plus volatiles).
  → Le capital est "équilibré" mais le RISQUE est très concentré.

  Risk Parity corrige ça : chaque actif contribue ÉGALEMENT au risque total.

  Contribution au risque de l'actif i :
    RC_i = w_i × (Σw)_i = w_i × MRC_i
    où MRC_i = (Σw)_i = risque marginal de l'actif i

  ERC : RC_i = RC_j pour tous i, j
    → Résolu numériquement (scipy.optimize)

DEUX MODES :
  1. Long-only (ETFs, retail) :
     ERC classique via scipy — minimise Σ_i Σ_j (RC_i - RC_j)²
     Avec w_i ≥ 0 et Σw_i = 1

  2. Long/Short market-neutral (hedge fund) :
     Approche Bridgewater "All Weather" adaptée au L/S :
     w_i ∝ signal_i / vol_i
     → Chaque position a la même volatilité dollar
     → Les actifs risqués sont sous-pondérés même si leur signal est fort
     → Plus stable qu'ERC pour les portefeuilles avec shorts

POURQUOI C'EST MIEUX QUE LE SIGNAL BRUT ?
  Sans risk parity : si XLE (énergie, vol=40%) et XLU (utilities, vol=12%)
    ont le même score de 0.5, ils reçoivent le même poids.
    Mais XLE contribue 3× plus au risque → concentration cachée.

  Avec risk parity : XLE reçoit un poids 3× plus petit que XLU
    → chaque actif contribue ~également à la volatilité totale
    → drawdowns moins violents en périodes de stress sectoriel.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

from src.optimization.constraints import PortfolioConstraints

logger = logging.getLogger(__name__)


class RiskParityOptimizer:
    """
    Optimiseur Risk Parity — ERC pour long-only, vol-scaling pour L/S.

    Paramètres
    ----------
    use_ledoit_wolf : bool
        Shrinkage de Ledoit-Wolf sur la covariance (défaut True).
    min_weight_floor : float
        Poids minimum imposé avant normalisation pour éviter les zéros
        (long-only uniquement). Défaut 1e-4.
    """

    def __init__(
        self,
        use_ledoit_wolf: bool  = True,
        min_weight_floor: float = 1e-4,
    ):
        self.use_ledoit_wolf  = use_ledoit_wolf
        self.min_weight_floor = min_weight_floor

    # ------------------------------------------------------------------
    # Interface principale
    # ------------------------------------------------------------------

    def optimize(
        self,
        expected_returns: pd.Series,
        returns_window:   pd.DataFrame,
        constraints:      PortfolioConstraints,
        prev_weights:     Optional[pd.Series] = None,
    ) -> pd.Series:
        """
        Calcule les poids risk-parity.

        Paramètres
        ----------
        expected_returns : pd.Series
            Signal factoriel (utilisé comme indicateur de direction en L/S).
        returns_window : pd.DataFrame
            Rendements historiques pour estimer la covariance / volatilités.
        constraints : PortfolioConstraints
            Contraintes du portefeuille.
        prev_weights : pd.Series, optional
            Non utilisé ici (risk parity est stationnaire), gardé pour
            compatibilité avec l'interface unifiée.
        """
        assets = expected_returns.index
        rets   = returns_window[assets].dropna()

        cov  = self._estimate_covariance(rets)
        vols = np.sqrt(np.diag(cov))   # volatilités individuelles annualisées

        if constraints.market_neutral:
            return self._vol_scaled_ls(expected_returns, vols, constraints)
        else:
            return self._erc_long_only(rets, cov, vols, constraints)

    # ------------------------------------------------------------------
    # Mode Long/Short : vol-scaling du signal
    # ------------------------------------------------------------------

    def _vol_scaled_ls(
        self,
        signal:      pd.Series,
        vols:        np.ndarray,
        constraints: PortfolioConstraints,
    ) -> pd.Series:
        """
        Pondère le signal par l'inverse de la volatilité.

        Formule : w_i = signal_i / vol_i
        Puis market-neutralize et normalise en levier unitaire.

        Effet : un actif 3× plus volatil que la moyenne reçoit un poids
        3× plus petit à score égal → contributions au risque équilibrées.
        """
        assets = signal.index
        vol_s  = pd.Series(vols, index=assets)

        # Evite la division par zéro
        vol_s = vol_s.replace(0.0, np.nan).fillna(vol_s.mean()).clip(lower=1e-8)

        weights = signal / vol_s

        # Market-neutral : somme(w) = 0
        weights = weights - weights.mean()

        # Normalise en levier unitaire
        abs_sum = weights.abs().sum()
        if abs_sum > 1e-10:
            weights = weights / abs_sum

        # Contraintes position
        weights = weights.clip(
            lower=constraints.min_position,
            upper=constraints.max_position,
        )
        weights = self._enforce_leverage(weights, constraints.max_leverage)

        return weights

    # ------------------------------------------------------------------
    # Mode Long-Only : ERC classique via scipy
    # ------------------------------------------------------------------

    def _erc_long_only(
        self,
        rets:        pd.DataFrame,
        cov:         np.ndarray,
        vols:        np.ndarray,
        constraints: PortfolioConstraints,
    ) -> pd.Series:
        """
        Equal Risk Contribution via scipy SLSQP.

        Objectif : minimiser Σ_i Σ_j (RC_i - RC_j)²
        où RC_i = w_i × (Σw)_i (contribution au risque de l'actif i)

        Point de départ : inverse-vol (bonne initialisation pour ERC).
        """
        assets = rets.columns
        n      = len(assets)

        # Initialisation : inverse-vol (approximation analytique d'ERC)
        w0 = (1.0 / vols)
        w0 = w0 / w0.sum()
        # Floor pour éviter les zéros au démarrage
        w0 = np.maximum(w0, self.min_weight_floor)
        w0 = w0 / w0.sum()

        def objective(w):
            """Variance des contributions au risque (à minimiser → 0 au ERC)."""
            Sw    = cov @ w
            rc    = w * Sw               # RC_i = w_i × (Σw)_i, non-normalisé
            # Somme des carrés des différences entre toutes les paires
            diff  = rc[:, None] - rc[None, :]  # (N, N)
            return float(np.sum(diff ** 2))

        def grad_objective(w):
            """Gradient analytique pour accélérer SLSQP."""
            Sw = cov @ w
            rc = w * Sw
            diff = rc[:, None] - rc[None, :]   # (N, N)
            # dL/dw_k = 4 Σ_j (RC_k - RC_j) × d(RC_k)/dw_k
            # d(RC_k)/dw_k = (Σw)_k + w_k × (Σ)_{kk}  (simplifié)
            drc_dw = Sw + w * np.diag(cov)
            grad   = 4.0 * diff.sum(axis=1) * drc_dw
            return grad

        scipy_constraints = [
            {"type": "eq", "fun": lambda w: w.sum() - 1.0, "jac": lambda w: np.ones(n)},
        ]
        bounds = [(self.min_weight_floor, constraints.max_position) for _ in range(n)]

        result = minimize(
            fun=objective,
            x0=w0,
            jac=grad_objective,
            method="SLSQP",
            bounds=bounds,
            constraints=scipy_constraints,
            options={"maxiter": 500, "ftol": 1e-10},
        )

        if not result.success:
            logger.debug(f"[RiskParity] Convergence imparfaite : {result.message}. Fallback inverse-vol.")
            # Fallback propre : inverse-volatilité
            w_fallback = 1.0 / vols
            w_fallback /= w_fallback.sum()
            return pd.Series(w_fallback, index=assets)

        weights = pd.Series(result.x, index=assets).clip(lower=0.0)
        weights /= weights.sum()   # re-normalise pour compenser les clips
        weights = self._enforce_leverage(weights, constraints.max_leverage)
        return weights

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _estimate_covariance(self, returns: pd.DataFrame) -> np.ndarray:
        """Ledoit-Wolf shrinkage + régularisation + annualisation."""
        X = returns.values

        if self.use_ledoit_wolf:
            lw = LedoitWolf()
            lw.fit(X)
            cov = lw.covariance_
        else:
            cov = np.cov(X, rowvar=False)

        eps = 1e-6 * np.trace(cov) / len(cov)
        cov += eps * np.eye(len(cov))
        return cov * 252

    @staticmethod
    def _enforce_leverage(weights: pd.Series, max_leverage: float) -> pd.Series:
        total = weights.abs().sum()
        if total > max_leverage:
            weights = weights * (max_leverage / total)
        return weights
