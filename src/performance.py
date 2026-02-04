import numpy as np
import pandas as pd


def compute_performance_metrics(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    annualization_factor: int = 252
) -> dict:
    """
    Calcule les métriques de performance d'un portefeuille.
    """
    mean_return = returns.mean() * annualization_factor
    volatility = returns.std() * np.sqrt(annualization_factor)

    sharpe = np.nan
    if volatility > 0:
        sharpe = (mean_return - risk_free_rate) / volatility

    cumulative = (1 + returns).cumprod()
    drawdown = cumulative / cumulative.cummax() - 1
    max_drawdown = drawdown.min()

    return {
        "Annualized Return": mean_return,
        "Annualized Volatility": volatility,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_drawdown
    }
