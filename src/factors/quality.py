# src/factors/quality.py
"""
Facteur de qualité — mesure la "solidité" d'un actif sur une période récente.

PRINCIPE :
  Un actif de "haute qualité" est un actif qui :
    1. A une faible volatilité réalisée  → mouvements stables, prévisibles
    2. A un bon ratio rendement/risque  → Sharpe élevé sur la période récente
    3. A une régularité de ses rendements → peu de jours extrêmes négatifs

  En finance institutionnelle, le facteur Quality est l'un des 5 facteurs
  de Fama-French (2015). Ici on l'adapte aux ETFs sectoriels.

INTUITION ÉCONOMIQUE :
  - XLU (Utilities) a historiquement une faible volatilité = score qualité élevé
  - XLE (Energy) est très volatile (corrélé au pétrole) = score qualité bas
  - Le signal : surpondérer les secteurs "stables" dans le portefeuille
"""

import numpy as np
import pandas as pd

from src.utils.stats import zscore_cross_sectional as _zscore


def compute_quality_score(
    returns: pd.DataFrame,
    window: int = 63,
) -> pd.Series:
    """
    Score de qualité composite basé sur la volatilité et le Sharpe récents.

    On utilise une fenêtre courte (63j = 3 mois) pour capter la qualité
    *récente* — un secteur peut être stable pendant des années puis
    devenir volatile (ex: XLF en 2008, XLE en 2020).

    Paramètres
    ----------
    returns : pd.DataFrame
        Fenêtre de rendements in-sample du backtest.
    window : int
        Nombre de jours récents à utiliser pour calculer la qualité.
        63 jours (3 mois) = compromis entre réactivité et stabilité.

    Retourne
    --------
    pd.Series
        Score de qualité z-scoré.
        Positif = actif stable/qualitatif → signal à surpondérer
        Négatif = actif volatile/risqué  → signal à sous-pondérer
    """
    # Utilise les `window` jours les plus récents de la fenêtre in-sample
    recent = returns.iloc[-window:] if len(returns) >= window else returns

    if len(recent) < 10:
        return pd.Series(0.0, index=returns.columns)

    # --- Composante 1 : Volatilité réalisée annualisée ---
    # Plus la volatilité est basse, meilleure est la qualité
    # → on prend l'opposé pour que "haute qualité = score élevé"
    ann_vol = recent.std() * np.sqrt(252)
    vol_score = _zscore(-ann_vol)   # signe négatif : vol basse = score haut

    # --- Composante 2 : Sharpe ratio récent ---
    # Ratio rendement annualisé / volatilité annualisée (taux sans risque = 0)
    # Un Sharpe élevé = bon rendement ajusté du risque = haute qualité
    ann_ret = recent.mean() * 252
    sharpe = ann_ret / ann_vol.replace(0.0, np.nan)
    sharpe = sharpe.fillna(0.0).clip(-5, 5)  # clip pour éviter les valeurs extrêmes
    sharpe_score = _zscore(sharpe)

    # --- Composante 3 : Downside deviation (semi-écart-type négatif) ---
    # Mesure uniquement les rendements négatifs → pénalise les actifs avec
    # de gros crashs même si la volatilité globale semble acceptable.
    # Formule : sqrt(mean(min(r, 0)^2) × 252)
    downside = recent.clip(upper=0)  # garde seulement les rendements négatifs
    downside_dev = np.sqrt((downside ** 2).mean() * 252)
    downside_score = _zscore(-downside_dev)   # basse downside dev = haute qualité

    # --- Score composite : moyenne pondérée des 3 composantes ---
    # Poids : volatilité 40%, Sharpe 40%, downside 20%
    quality = 0.40 * vol_score + 0.40 * sharpe_score + 0.20 * downside_score

    return _zscore(quality)  # re-normaliser le score final


def compute_stability_score(
    returns: pd.DataFrame,
    short_window: int = 21,
    long_window: int = 126,
) -> pd.Series:
    """
    Score de stabilité : mesure la cohérence du comportement d'un actif.

    Idée : compare la volatilité courte vs longue.
    Si vol_courte << vol_longue → l'actif se calme → signal positif
    Si vol_courte >> vol_longue → l'actif s'emballe → signal négatif

    C'est en fait un signal de "mean reversion de la volatilité" :
    utile pour anticiper les régimes.

    Paramètres
    ----------
    short_window : int
        Fenêtre courte pour la volatilité récente (21j = 1 mois).
    long_window : int
        Fenêtre longue pour la volatilité de référence (126j = 6 mois).
    """
    if len(returns) < long_window:
        return pd.Series(0.0, index=returns.columns)

    vol_short = returns.iloc[-short_window:].std() * np.sqrt(252)
    vol_long  = returns.iloc[-long_window:].std()  * np.sqrt(252)

    # Ratio vol_courte / vol_longue :
    # < 1 → l'actif se calme (favoriser) | > 1 → l'actif s'emballe (éviter)
    vol_ratio = vol_short / vol_long.replace(0.0, np.nan)
    vol_ratio = vol_ratio.fillna(1.0)

    # Score = -log(ratio) : négatif si ratio > 1, positif si ratio < 1
    stability = -np.log(vol_ratio)

    return _zscore(stability)
