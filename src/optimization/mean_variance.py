# src/optimization/mean_variance.py
"""
Optimisation Mean-Variance de Markowitz avec améliorations institutionnelles.

PROBLÈME DE MARKOWITZ CLASSIQUE :
  Maximiser : w'μ - (λ/2) × w'Σw
  Sujet à    : contraintes de portefeuille

  Où :
    μ  = rendements espérés (vecteur N×1)
    Σ  = matrice de covariance (N×N)
    λ  = aversion au risque (calibré pour cibler un niveau de vol)
    w  = poids du portefeuille (vecteur N×1)

PROBLÈMES DU MVO NAÏF (et nos solutions) :
  1. "Error maximization" : le MVO amplifie les erreurs d'estimation de μ
     → Solution : Ledoit-Wolf shrinkage sur Σ (sklearn)
     → Solution : utiliser nos signaux factoriels comme μ (pas de régression)

  2. Poids extrêmes et instables
     → Solution : contraintes explicites (max_position, max_leverage)

  3. Covariance singulière si N > T
     → Solution : shrinkage + regularization (Σ + ε×I)

SHRINKAGE LEDOIT-WOLF :
  Σ_shrunk = (1-α)×Σ_sample + α×F
  Où F est une cible structurée (matrice diagonale ici).
  α est calculé analytiquement pour minimiser l'erreur quadratique.
  Résultat : covariance plus stable, moins sujette à l'overfitting.

SOLVER : scipy.optimize.minimize avec SLSQP
  SLSQP = Sequential Least Squares Programming → gère les contraintes linéaires
  et quadratiques, rapide pour N ≤ 50 actifs.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

from src.optimization.constraints import PortfolioConstraints

logger = logging.getLogger(__name__)


class MeanVarianceOptimizer:
    """
    Optimiseur Mean-Variance avec shrinkage de covariance et contraintes réelles.

    Paramètres
    ----------
    risk_aversion : float
        Coefficient λ d'aversion au risque.
        Valeur élevée → portefeuille défensif (low vol)
        Valeur basse  → portefeuille agressif (high return)
        Défaut : 1.0 (calibration neutre)
    target_vol : float, optional
        Si fourni, λ est ajusté pour cibler ce niveau de volatilité annualisée.
        Ex: target_vol=0.10 → portefeuille ciblant ~10% de vol annuelle.
    use_ledoit_wolf : bool
        True (défaut) → shrinkage Ledoit-Wolf sur la covariance.
        False → covariance empirique brute (moins stable).
    """

    def __init__(
        self,
        risk_aversion:    float          = 1.0,
        target_vol:       Optional[float] = None,
        use_ledoit_wolf:  bool           = True,
    ):
        self.risk_aversion   = risk_aversion
        self.target_vol      = target_vol
        self.use_ledoit_wolf = use_ledoit_wolf

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
        Calcule les poids optimaux du portefeuille.

        Paramètres
        ----------
        expected_returns : pd.Series
            Rendements espérés par actif (index = actifs).
            En pratique : nos scores factoriels normalisés.
        returns_window : pd.DataFrame
            Fenêtre de rendements historiques pour estimer Σ.
        constraints : PortfolioConstraints
            Contraintes à appliquer.
        prev_weights : pd.Series, optional
            Poids précédents (pour contrainte de turnover si activée).

        Retourne
        --------
        pd.Series
            Poids optimaux (index = actifs).
        """
        assets = expected_returns.index
        n      = len(assets)

        # Aligner les données
        rets = returns_window[assets].dropna()
        mu   = expected_returns.values.astype(float)

        # --- Estimation de la covariance ---
        cov = self._estimate_covariance(rets)

        # --- Aversion au risque : ajustement si target_vol fourni ---
        lam = self._calibrate_lambda(cov, constraints)

        # --- Point de départ pour le solver ---
        if prev_weights is not None:
            w0 = prev_weights.reindex(assets).fillna(0.0).values
        else:
            w0 = np.zeros(n)

        # --- Fonction objectif : -( w'μ - λ/2 × w'Σw ) ---
        # On minimise le négatif = on maximise l'utilité espérée
        def objective(w):
            port_ret = float(w @ mu)
            port_var = float(w @ cov @ w)
            return -(port_ret - (lam / 2) * port_var)

        def grad_objective(w):
            return -(mu - lam * cov @ w)

        # --- Contraintes SLSQP ---
        scipy_constraints = self._build_scipy_constraints(
            n, constraints, prev_weights, assets
        )

        # --- Bornes par actif ---
        bounds = [
            (constraints.min_position, constraints.max_position)
            for _ in range(n)
        ]

        # --- Optimisation ---
        result = minimize(
            fun=objective,
            x0=w0,
            jac=grad_objective,
            method="SLSQP",
            bounds=bounds,
            constraints=scipy_constraints,
            options={"maxiter": 500, "ftol": 1e-9},
        )

        if not result.success:
            logger.debug(
                f"[MVO] Convergence imparfaite : {result.message}. "
                f"Utilisation du meilleur résultat trouvé."
            )

        weights = pd.Series(result.x, index=assets)

        # Post-traitement : clip final + renormalisation
        weights = weights.clip(
            lower=constraints.min_position,
            upper=constraints.max_position,
        )
        weights = self._enforce_leverage(weights, constraints.max_leverage)

        return weights

    # ------------------------------------------------------------------
    # Helpers privés
    # ------------------------------------------------------------------

    def _estimate_covariance(self, returns: pd.DataFrame) -> np.ndarray:
        """
        Estime la matrice de covariance avec Ledoit-Wolf shrinkage.

        Ledoit-Wolf est le standard en gestion quantitative car il produit
        une covariance mieux conditionnée que l'estimateur empirique quand
        T/N est faible (ce qui est souvent le cas en finance).
        """
        X = returns.values

        if self.use_ledoit_wolf:
            lw = LedoitWolf()
            lw.fit(X)
            cov = lw.covariance_
        else:
            cov = np.cov(X, rowvar=False)

        # Régularisation : ajoute ε×I pour garantir définie positive
        # Nécessaire si certains actifs sont très corrélés
        eps = 1e-6 * np.trace(cov) / len(cov)
        cov += eps * np.eye(len(cov))

        # Annualiser (rendements journaliers × 252)
        return cov * 252

    def _calibrate_lambda(
        self,
        cov: np.ndarray,
        constraints: PortfolioConstraints,
    ) -> float:
        """
        Si target_vol est fourni, ajuste λ pour que le portefeuille EW
        atteigne approximativement ce niveau de volatilité.
        Sinon retourne risk_aversion tel quel.
        """
        if self.target_vol is None:
            return self.risk_aversion

        n = cov.shape[0]
        w_ew = np.ones(n) / n
        vol_ew = np.sqrt(float(w_ew @ cov @ w_ew))

        if vol_ew < 1e-8:
            return self.risk_aversion

        # λ proportionnel au ratio vol_cible / vol_ew
        # Plus la vol cible est basse → λ élevé → pénalise plus la variance
        return self.risk_aversion * (vol_ew / self.target_vol) ** 2

    def _build_scipy_constraints(
        self,
        n:            int,
        constraints:  PortfolioConstraints,
        prev_weights: Optional[pd.Series],
        assets:       pd.Index,
    ) -> list:
        """Construit la liste de contraintes pour scipy.optimize.minimize."""
        scipy_constraints = []

        # Contrainte de neutralité marché : sum(w) = 0
        if constraints.market_neutral:
            scipy_constraints.append({
                "type": "eq",
                "fun":  lambda w: np.sum(w),
                "jac":  lambda w: np.ones(n),
            })

        # Contrainte de levier : sum(|w|) ≤ max_leverage
        # Approximée par w_pos - w_neg ≤ max_leverage/2 (car SLSQP ne gère pas |w|)
        # On le gère post-optimisation via _enforce_leverage

        # Contrainte de turnover : ||w - w_prev||_1 ≤ max_turnover
        if constraints.max_turnover is not None and prev_weights is not None:
            w_prev = prev_weights.reindex(assets).fillna(0.0).values
            scipy_constraints.append({
                "type": "ineq",
                # sum(|w - w_prev|) ≤ max_turnover → approximé par norme L2
                "fun":  lambda w: constraints.max_turnover - np.sum(np.abs(w - w_prev)),
            })

        return scipy_constraints

    @staticmethod
    def _enforce_leverage(weights: pd.Series, max_leverage: float) -> pd.Series:
        """
        Normalise les poids pour respecter la contrainte de levier.
        Si sum(|w|) > max_leverage → scaling proportionnel.
        """
        total_leverage = weights.abs().sum()
        if total_leverage > max_leverage:
            weights = weights * (max_leverage / total_leverage)
        return weights
