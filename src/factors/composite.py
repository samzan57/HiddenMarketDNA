# src/factors/composite.py
"""
Combinaison des signaux multi-facteurs en un signal de trading unique.

ARCHITECTURE DU SIGNAL COMPOSITE :

    Signal final = w_pca × ACP(PC2) + w_mom × Momentum + w_qual × Quality

  Où chaque composante est préalablement z-scorée (mise à la même échelle).

POURQUOI CETTE COMBINAISON ?
  - ACP seul (Sprint 1) : capte la structure sectorielle mais ignore la tendance
  - Momentum seul      : fort signal mais très cyclique (drawdowns en retournements)
  - Qualité seule      : conservateur, bon en période de stress
  - Combiné            : les trois se compensent → Sharpe plus stable

CALIBRATION DES POIDS (hyperparamètres) :
  Les poids par défaut (50/30/20) sont basés sur la littérature académique.
  Sprint 4 (optimisation) les calibrera sur données historiques via
  walk-forward optimization pour éviter l'overfitting.

NEUTRALITÉ DE MARCHÉ :
  Le signal final est toujours market-neutral :
  - sum(poids longs) ≈ +1
  - sum(poids courts) ≈ -1
  - sum(tous les poids) ≈ 0  (pas d'exposition directionnelle nette)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional

from src.utils.stats import zscore_cross_sectional as _zscore


@dataclass
class FactorWeights:
    """
    Poids de chaque signal dans le composite.
    Encapsulé dans un dataclass pour faciliter les tests et l'optimisation future.
    """
    pca:      float = 0.50   # ACP (structure latente du marché)
    momentum: float = 0.30   # Momentum cross-sectionnel
    quality:  float = 0.20   # Qualité / stabilité

    def __post_init__(self):
        total = self.pca + self.momentum + self.quality
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Les poids doivent sommer à 1.0, somme actuelle : {total:.4f}\n"
                f"Astuce : si quality=0.0, mettre pca+momentum=1.0"
            )
        # Un facteur à 0 est autorisé (désactivation explicite d'un signal)
        if any(w < 0 for w in [self.pca, self.momentum, self.quality]):
            raise ValueError("Les poids ne peuvent pas être négatifs.")


def _market_neutralize(weights: pd.Series) -> pd.Series:
    """
    Rend les poids market-neutral : soustrait la moyenne pour que
    la somme des poids soit ≈ 0 (pas d'exposition beta nette).
    """
    return weights - weights.mean()


def _normalize_to_unit_leverage(weights: pd.Series) -> pd.Series:
    """
    Normalise les poids pour que la somme des valeurs absolues = 1.
    Convention standard des fonds long/short : levier total = 1.
    Ex: 50% long XLK, 50% short XLE → |+0.5| + |-0.5| = 1.
    """
    abs_sum = weights.abs().sum()
    if abs_sum < 1e-10:
        return pd.Series(0.0, index=weights.index)
    return weights / abs_sum


class CompositeSignal:
    """
    Combine les trois signaux factoriels en poids de portefeuille finaux.

    Paramètres
    ----------
    weights : FactorWeights
        Poids de chaque signal (doivent sommer à 1).
    normalize : bool
        Si True : normalise les poids finaux en levier unitaire (défaut : True).
    """

    def __init__(
        self,
        weights: Optional[FactorWeights] = None,
        normalize: bool = True,
    ):
        self.weights = weights or FactorWeights()
        self.normalize = normalize

    def compute(
        self,
        pca_loadings:     pd.Series,
        momentum_scores:  pd.Series,
        quality_scores:   pd.Series,
    ) -> pd.Series:
        """
        Calcule les poids finaux du portefeuille.

        Paramètres
        ----------
        pca_loadings : pd.Series
            Loadings PC2 issus de l'ACP (index = actifs)
        momentum_scores : pd.Series
            Z-scores du momentum cross-sectionnel (index = actifs)
        quality_scores : pd.Series
            Z-scores de la qualité (index = actifs)

        Retourne
        --------
        pd.Series
            Poids market-neutral du portefeuille (index = actifs)
        """
        # Aligner tous les signaux sur le même univers d'actifs
        assets = pca_loadings.index
        mom  = momentum_scores.reindex(assets).fillna(0.0)
        qual = quality_scores.reindex(assets).fillna(0.0)

        # Z-scorer chaque signal → même unité de mesure
        pca_z  = _zscore(pca_loadings)
        mom_z  = _zscore(mom)
        qual_z = _zscore(qual)

        # Combinaison linéaire pondérée
        composite = (
            self.weights.pca      * pca_z
            + self.weights.momentum * mom_z
            + self.weights.quality  * qual_z
        )

        # Neutralisation marché : somme des poids = 0
        composite = _market_neutralize(composite)

        # Normalisation en levier unitaire
        if self.normalize:
            composite = _normalize_to_unit_leverage(composite)

        return composite

    def compute_with_breakdown(
        self,
        pca_loadings:    pd.Series,
        momentum_scores: pd.Series,
        quality_scores:  pd.Series,
    ) -> dict:
        """
        Variante verbose : retourne aussi la contribution de chaque signal.
        Utile pour le dashboard de monitoring (Sprint 6) et le debug.

        Retourne
        --------
        dict avec clés :
          'weights'    : pd.Series → poids finaux
          'pca_contrib'  : pd.Series → contribution du signal ACP
          'mom_contrib'  : pd.Series → contribution du signal momentum
          'qual_contrib' : pd.Series → contribution du signal qualité
        """
        assets = pca_loadings.index
        mom  = momentum_scores.reindex(assets).fillna(0.0)
        qual = quality_scores.reindex(assets).fillna(0.0)

        pca_z  = _zscore(pca_loadings)
        mom_z  = _zscore(mom)
        qual_z = _zscore(qual)

        pca_contrib  = self.weights.pca      * pca_z
        mom_contrib  = self.weights.momentum * mom_z
        qual_contrib = self.weights.quality  * qual_z

        composite = _market_neutralize(pca_contrib + mom_contrib + qual_contrib)
        if self.normalize:
            composite = _normalize_to_unit_leverage(composite)

        return {
            "weights":      composite,
            "pca_contrib":  pca_contrib,
            "mom_contrib":  mom_contrib,
            "qual_contrib": qual_contrib,
        }
