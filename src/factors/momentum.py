# src/factors/momentum.py
"""
Momentum cross-sectionnel — implémentation du facteur de Jegadeesh & Titman (1993)

PRINCIPE :
  Le momentum dit que les actifs qui ont bien performé dans le passé
  continuent de surperformer à court/moyen terme.
  En version cross-sectorielle : on achète les secteurs les plus forts
  et on vend les plus faibles — relativement à l'univers.

CONVENTION STANDARD EN FINANCE QUANTITATIVE :
  - Lookback : 12 mois (252 jours) → capture la tendance structurelle
  - Skip     :  1 mois  (21 jours) → on ignore le mois récent pour éviter
                le phénomène de "short-term reversal" (retour à la moyenne
                sur 1-4 semaines documenté par Lehmann, 1990)

  Signal = rendement cumulatif de [t-252 → t-21] → z-score cross-sectionnel

RÉSULTATS EMPIRIQUES CONNUS (pour info) :
  - AQR publie un Sharpe ~0.4-0.5 sur ce facteur seul
  - Combiné à d'autres facteurs : amélioration significative
"""

import numpy as np
import pandas as pd

from src.utils.stats import zscore_cross_sectional as _zscore_cross_sectional


def compute_momentum_score(
    returns: pd.DataFrame,
    lookback: int = 252,
    skip: int = 21,
) -> pd.Series:
    """
    Calcule le score de momentum cross-sectionnel sur une fenêtre de rendements.

    Ce score est calculé à chaque pas de temps t à partir des rendements
    historiques [t-lookback, t]. Il est ensuite utilisé comme signal de trading.

    Paramètres
    ----------
    returns : pd.DataFrame
        Fenêtre de rendements log. index=Date, colonnes=actifs.
        Typiquement la fenêtre in-sample du rolling backtest (252 jours).
    lookback : int
        Nombre total de jours à regarder en arrière.
    skip : int
        Nombre de jours récents à ignorer (évite le short-term reversal).
        Si skip=0 : on utilise tous les jours (moins standard).

    Retourne
    --------
    pd.Series
        Z-scores du momentum par actif.
        Positif = forte tendance haussière relative → signal long
        Négatif = faible tendance / baissière → signal short
    """
    n = len(returns)

    # Pas assez de données pour calculer un signal fiable → scores nuls
    if n < lookback:
        return pd.Series(0.0, index=returns.columns)

    # Fenêtre de calcul : de t-lookback à t-skip
    # On exclut les `skip` derniers jours pour éviter le reversal
    if skip > 0 and n > skip:
        window = returns.iloc[-lookback:-skip]
    else:
        window = returns.iloc[-lookback:]

    if len(window) < 21:  # moins d'un mois de données : signal non fiable
        return pd.Series(0.0, index=returns.columns)

    # Rendement cumulatif sur la fenêtre
    # Formule : exp(sum(log_returns)) - 1 = produit des (1 + r_t) - 1
    cum_return = np.exp(window.sum()) - 1

    return _zscore_cross_sectional(cum_return)


def compute_multi_horizon_momentum(
    returns: pd.DataFrame,
    horizons: dict = None,
) -> pd.Series:
    """
    Momentum multi-horizon : combine plusieurs fenêtres temporelles.

    Pourquoi plusieurs horizons ?
    - 1 mois  (21j)  : capture les tendances très court terme
    - 3 mois  (63j)  : tendances de moyen terme
    - 12 mois (252j) : tendances structurelles longues

    Combinaison par moyenne pondérée → signal plus stable que mono-horizon.

    Paramètres
    ----------
    horizons : dict {nom: (lookback, skip, weight)}
        Définition des horizons à combiner.
        Défaut : standard Jegadeesh-Titman 12-1 + court terme 3-1.
    """
    if horizons is None:
        horizons = {
            "mom_3m":  (63,  5,  0.3),   # 3 mois, skip 1 semaine, poids 30%
            "mom_12m": (252, 21, 0.7),   # 12 mois, skip 1 mois,   poids 70%
        }

    composite = pd.Series(0.0, index=returns.columns)

    for name, (lookback, skip, weight) in horizons.items():
        score = compute_momentum_score(returns, lookback=lookback, skip=skip)
        composite += weight * score

    # Re-normaliser le signal combiné
    return _zscore_cross_sectional(composite)
