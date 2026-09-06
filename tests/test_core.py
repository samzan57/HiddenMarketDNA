"""
Unit tests for the core research pillar: log-return computation
(src/returns.py) and the PCA engine (src/pca_engine.py).

Run with:
    pytest tests/
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.returns import compute_log_returns
from src.pca_engine import PCAEngine


def test_compute_log_returns_matches_manual_calculation():
    prices = pd.DataFrame({"A": [100.0, 105.0, 110.25], "B": [50.0, 49.0, 51.0]})

    log_returns = compute_log_returns(prices)

    expected_A = np.log(105.0 / 100.0)
    expected_B = np.log(49.0 / 50.0)

    assert len(log_returns) == 2  # first row dropped (no prior price)
    assert log_returns["A"].iloc[0] == pytest.approx(expected_A)
    assert log_returns["B"].iloc[0] == pytest.approx(expected_B)


def test_compute_log_returns_rejects_non_positive_prices():
    prices = pd.DataFrame({"A": [100.0, -5.0, 110.0]})

    with pytest.raises(ValueError):
        compute_log_returns(prices)


def test_pca_engine_first_component_dominates_for_correlated_assets():
    # 6 assets that are near-perfect copies of one common trend plus tiny
    # idiosyncratic noise: PC1 should capture almost all the variance,
    # exactly the "one dominant market factor" claim in the README.
    rng = np.random.default_rng(7)
    common_trend = rng.standard_normal(500)
    noise = rng.standard_normal((500, 6)) * 0.01
    returns = pd.DataFrame(common_trend[:, None] + noise, columns=list("ABCDEF"))

    engine = PCAEngine().fit(returns)
    explained = engine.explained_variance()

    assert explained.iloc[0] > 0.95
    assert explained.sum() == pytest.approx(1.0, abs=1e-6)


def test_pca_engine_requires_fit_before_use():
    engine = PCAEngine()
    returns = pd.DataFrame({"A": [0.01, -0.02, 0.03]})

    with pytest.raises(RuntimeError):
        engine.transform(returns)
    with pytest.raises(RuntimeError):
        engine.get_eigen_portfolios()
    with pytest.raises(RuntimeError):
        engine.explained_variance()
