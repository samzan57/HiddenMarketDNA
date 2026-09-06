# src/utils/stats.py
import numpy as np
import pandas as pd


def zscore_cross_sectional(series: pd.Series) -> pd.Series:
    """Z-score cross-sectionnel. Retourne zéros si std ≈ 0."""
    std = series.std()
    if std < 1e-10:
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std
