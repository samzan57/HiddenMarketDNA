# src/returns.py

import numpy as np
import pandas as pd

def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule les rendements logarithmiques à partir des prix.

    Parameters
    ----------
    prices : pd.DataFrame
        DataFrame de prix, indexé par date, colonnes = actifs

    Returns
    -------
    pd.DataFrame
        DataFrame de rendements log, aligné et nettoyé
    """
    # --- Forcer uniquement les colonnes en float ---
    prices = prices.copy()
    for col in prices.columns:
        prices[col] = pd.to_numeric(prices[col], errors='coerce')

    # Supprimer toutes les lignes contenant NaN
    prices = prices.dropna(how='any')

    # Vérification des prix positifs
    if (prices <= 0).any().any():
        raise ValueError("Les prix doivent être strictement positifs")

    # Calcul des rendements logarithmiques
    log_returns = np.log(prices / prices.shift(1))

    # Supprimer la première ligne NaN créée par le shift
    log_returns = log_returns.dropna(how='any')

    # Forcer le type float64
    log_returns = log_returns.astype(float)

    return log_returns
