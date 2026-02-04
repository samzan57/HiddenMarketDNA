# src/visualization.py

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

sns.set_style("whitegrid")


def plot_portfolio_with_regimes(portfolio_returns: pd.Series, weights: pd.DataFrame, pc1_vol: pd.Series, vol_threshold: float):
    """
    Affiche le rendement cumulatif du portefeuille et les régimes de marché.
    
    Paramètres
    ----------
    portfolio_returns : pd.Series
        Rendements du portefeuille rolling
    weights : pd.DataFrame
        Historique des poids (optionnel pour vérification)
    pc1_vol : pd.Series
        Volatilité du PC1 à chaque pas (pour détecter les régimes)
    vol_threshold : float
        Seuil de volatilité pour considérer un marché comme volatil
    """

    # Rendement cumulatif
    cumulative_returns = (1 + portfolio_returns).cumprod()

    # Détection des régimes
    high_vol_regime = pc1_vol > vol_threshold

    plt.figure(figsize=(12, 5))
    plt.plot(cumulative_returns.index, cumulative_returns.values, label="Portfolio Cumulative Returns", color="blue")

    # Coloration des périodes volatiles
    for start, end in contiguous_regions(high_vol_regime):
        plt.axvspan(cumulative_returns.index[start], cumulative_returns.index[end], color="red", alpha=0.2)

    plt.title("Portfolio Cumulative Returns with Market Regimes")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Returns")
    plt.legend()
    plt.tight_layout()
    plt.show()


def contiguous_regions(condition: pd.Series):
    """
    Transforme une série booléenne en intervalles contigus True.
    Retourne une liste de tuples (start_index, end_index)
    """
    condition = condition.values
    d = np.diff(condition.astype(int))
    idx, = d.nonzero() 

    idx += 1

    if condition[0]:
        idx = np.r_[0, idx]

    if condition[-1]:
        idx = np.r_[idx, condition.size]

    idx.shape = (-1, 2)
    return idx
