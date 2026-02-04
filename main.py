import sys
import os
# Racine du projet (1 niveau au-dessus de main.py)
sys.path.append(os.path.abspath(".."))

from src.data_loader import DataLoader
from src.returns import compute_log_returns
from src.rolling_backtest import RollingPCABacktest
from src.performance import compute_performance_metrics
from src.visualization import plot_portfolio_with_regimes

# -----------------------------
# Configuration
# -----------------------------
DATA_DIR = "data/raw"
ETF_LIST = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU"]

N_COMPONENTS = 3
TARGET_FACTOR = "PC2"
WINDOW = 252  # taille de la fenêtre pour le rolling PCA
RISK_SCALE_HIGH_VOL = 0.5
VOL_THRESHOLD_QUANTILE = 0.75

# -----------------------------
# Pipeline
# -----------------------------
def main():
    # 1. Chargement des prix
    loader = DataLoader(DATA_DIR)
    prices = loader.load_prices(ETF_LIST)

    # 2. Rendements
    returns = compute_log_returns(prices)

    # 3. Rolling PCA Backtest
    backtest = RollingPCABacktest(
        returns=returns,
        window=WINDOW,
        n_components=N_COMPONENTS,
        target_factor=TARGET_FACTOR,
        risk_scale_high_vol=RISK_SCALE_HIGH_VOL,
        vol_threshold_quantile=VOL_THRESHOLD_QUANTILE
    )

    backtest.run()

    rolling_returns, rolling_weights, pc1_vol = backtest.get_results()

    # 4. Diagnostics de performance
    metrics = compute_performance_metrics(rolling_returns)

    print("==== Résultats Rolling PCA avec régimes ====")
    for k, v in metrics.items():
        print(f"{k}: {v:.3f}")

    # 5. Exemple de poids
    print("\nExemple de poids (premières lignes) :")
    print(rolling_weights.head())

    # 6. Visualisation cumulée avec régimes de marché
    plot_portfolio_with_regimes(
        portfolio_returns=rolling_returns,
        weights=rolling_weights,
        pc1_vol=pc1_vol,
        vol_threshold=backtest.vol_threshold
    )


if __name__ == "__main__":
    main()
