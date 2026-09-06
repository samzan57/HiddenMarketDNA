"""
HiddenMarketDNA — Pipeline principal
=====================================
Backtest :
    python main.py

Live trading (paper IBKR) :
    python -m src.live_trader --dry-run   # test sans envoyer d'ordres
    python -m src.live_trader             # execution reelle

Sprints disponibles :
  Sprint 1 : USE_FACTORS=False -> ACP seul
  Sprint 2 : USE_FACTORS=True, USE_ADVANCED_REGIMES=False -> multi-facteurs
  Sprint 3 : USE_FACTORS=True, USE_ADVANCED_REGIMES=True  -> HMM+GARCH (config finale)
  Sprint 4 : USE_OPTIMIZER=True -> optimisation MVO / Risk Parity / Equal-Weight
  Sprint 5 : Live trading IBKR paper (src/live_trader.py)
"""

import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

from src.data.pipeline import DataPipeline
from src.data.universe import describe_universe
from src.rolling_backtest import RollingPCABacktest
from src.factor_backtest import RollingFactorBacktest
from src.factors.composite import FactorWeights
from src.optimization.optimizer import PortfolioOptimizer
from src.optimization.constraints import PortfolioConstraints
from src.performance import compute_performance_metrics
from src.visualization import plot_portfolio_with_regimes

# =============================================================================
# CONFIGURATION
# =============================================================================

# --- Mode ---
# False = Sprint 1 (ACP seul)
# True  = Sprint 2+3 (multi-facteurs + régimes HMM/GARCH)
USE_FACTORS = True

# --- Régimes avancés (Sprint 3) ---
# True  = HMM + GARCH (précis, ~5-8 min)
# False = seuil vol PC1 simple (rapide, ~2-3 min)
USE_ADVANCED_REGIMES = True
REGIME_UPDATE_FREQ   = 21    # refit HMM/GARCH tous les N jours (21 = mensuel)
MIN_RISK_SCALE       = 0.30  # exposition minimale en stress extrême

# --- Univers ---
UNIVERSE   = "sector_etfs"
START_DATE = "2010-01-01"
END_DATE   = None

# --- Paramètres ACP ---
N_COMPONENTS  = 3
TARGET_FACTOR = "PC2"
WINDOW        = 252

# --- Gestion du risque (fallback si USE_ADVANCED_REGIMES=False) ---
RISK_SCALE_HIGH_VOL    = 0.5
VOL_THRESHOLD_QUANTILE = 0.75

# --- Poids des facteurs ---
FACTOR_WEIGHTS = FactorWeights(
    pca      = 0.60,
    momentum = 0.40,
    quality  = 0.00,   # désactivé : anti-corrélé avec momentum sur secteurs
)

# --- Sprint 4 : Optimisation de portefeuille ---
# False → signal factoriel brut comme poids (Sprint 1/2/3)
# True  → signal passé dans l'optimiseur
USE_OPTIMIZER    = False
OPTIMIZER_METHOD = "mvo"           # "mvo" | "risk_parity" | "equal_weight"

# Contraintes (partagées par MVO et Risk Parity)
# long_short_standard() = levier 2x, max 15% par actif, market-neutral
CONSTRAINTS = PortfolioConstraints.long_short_standard()


# =============================================================================
# PIPELINE
# =============================================================================

def main():

    describe_universe()
    print()

    # --- 1. Données ---
    print(">>> Étape 1 : Chargement des données...")
    pipeline = DataPipeline(cache_dir="data/cache", cache_expiry_hours=24)
    _, returns = pipeline.run_returns(start=START_DATE, end=END_DATE, universe=UNIVERSE)
    pipeline.print_report()
    print(f"    → {returns.shape[1]} actifs × {returns.shape[0]} jours\n")

    # --- 2. Backtest ---
    if USE_FACTORS:
        regime_str = "HMM+GARCH" if USE_ADVANCED_REGIMES else "Vol-Threshold"
        optim_str  = f"Optimizer={OPTIMIZER_METHOD}" if USE_OPTIMIZER else "Signal direct"
        print(f">>> Étape 2 : Rolling Factor Backtest ({regime_str} | {optim_str})...")

        optimizer = (
            PortfolioOptimizer(
                method=OPTIMIZER_METHOD,
                constraints=CONSTRAINTS,
                target_vol=0.06,   # calibre λ pour cibler ~6% vol annuelle
            )
            if USE_OPTIMIZER else None
        )

        backtest = RollingFactorBacktest(
            returns=returns,
            window=WINDOW,
            n_components=N_COMPONENTS,
            target_factor=TARGET_FACTOR,
            factor_weights=FACTOR_WEIGHTS,
            use_advanced_regimes=USE_ADVANCED_REGIMES,
            regime_update_freq=REGIME_UPDATE_FREQ,
            min_risk_scale=MIN_RISK_SCALE,
            risk_scale_high_vol=RISK_SCALE_HIGH_VOL,
            vol_threshold_quantile=VOL_THRESHOLD_QUANTILE,
            optimizer=optimizer,
        )
    else:
        print(">>> Étape 2 : Rolling PCA Backtest (Sprint 1 — ACP seul)...")
        backtest = RollingPCABacktest(
            returns=returns,
            window=WINDOW,
            n_components=N_COMPONENTS,
            target_factor=TARGET_FACTOR,
            risk_scale_high_vol=RISK_SCALE_HIGH_VOL,
            vol_threshold_quantile=VOL_THRESHOLD_QUANTILE,
        )

    backtest.run()
    rolling_returns, rolling_weights, pc1_vol = backtest.get_results()

    # --- 3. Performance ---
    print(">>> Étape 3 : Métriques de performance")
    metrics = compute_performance_metrics(rolling_returns)

    if not USE_FACTORS:
        mode_label = "ACP SEUL (Sprint 1)"
    elif USE_OPTIMIZER:
        mode_label = f"MULTI-FACTEURS + HMM/GARCH + {OPTIMIZER_METHOD.upper()} (Sprint 4)"
    elif USE_ADVANCED_REGIMES:
        mode_label = "MULTI-FACTEURS + HMM/GARCH (Sprint 3)"
    else:
        mode_label = "MULTI-FACTEURS + VOL-THRESHOLD (Sprint 2)"
    print(f"\n{'=' * 45}")
    print(f"   RÉSULTATS — {mode_label}")
    print(f"{'=' * 45}")
    for k, v in metrics.items():
        print(f"  {k:<28} : {v:>8.3f}")
    print(f"{'=' * 45}\n")

    # Afficher la décomposition factorielle (Sprint 2 uniquement)
    if USE_FACTORS and hasattr(backtest, "get_annualized_factor_contributions"):
        contribs = backtest.get_annualized_factor_contributions()
        print("   Attribution par facteur (rendement annualisé):")
        for factor, contrib in contribs.items():
            print(f"     {factor:<12} : {contrib:+.4f}")
        print()

    # --- 4. Visualisation ---
    print(">>> Étape 4 : Génération du graphique...")
    plot_portfolio_with_regimes(
        portfolio_returns=rolling_returns,
        weights=rolling_weights,
        pc1_vol=pc1_vol,
        vol_threshold=backtest.vol_threshold,
    )


if __name__ == "__main__":
    main()
