# src/live_trader.py
"""
Boucle principale du live trader HiddenMarketDNA.

FLUX PAR EXÉCUTION (hebdomadaire, lundi ~9h35 NY) :

  1. Connexion IBKR (TWS paper trading — host/port/client ID via .env)
  2. Récupération compte : valeur nette + positions actuelles
  3. Téléchargement des prix récents (via DataPipeline — yfinance)
  4. Génération du signal : PCA + Momentum + HMM/GARCH
  5. Risk check (levier, position max, drawdown)
  6. Calcul des ordres (delta positions)
  7. Exécution (ordres au marché)
  8. Log du résultat

UTILISATION :
  python -m src.live_trader          # exécution unique
  python -m src.live_trader --dry-run # simulation sans envoyer d'ordres

Pour automatiser : Windows Task Scheduler, tous les lundis à 15h35 UTC (9h35 NY).
"""

import argparse
import logging
import os
import sys
from datetime import datetime

from src.telegram_notify import send as tg

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION LIVE (doit correspondre au backtest Sprint 3)
# =============================================================================

UNIVERSE         = "sector_etfs"
START_DATE       = "2010-01-01"    # historique complet pour PCA/HMM stables
N_COMPONENTS     = 3
TARGET_FACTOR    = "PC2"
WINDOW           = 252

USE_ADVANCED_REGIMES = True
REGIME_UPDATE_FREQ   = 21
MIN_RISK_SCALE       = 0.30

RISK_SCALE_HIGH_VOL    = 0.5
VOL_THRESHOLD_QUANTILE = 0.75

from src.factors.composite import FactorWeights

FACTOR_WEIGHTS = FactorWeights(pca=0.60, momentum=0.40, quality=0.00)

# Risk guard
MAX_LEVERAGE        = 2.0
MAX_SINGLE_POSITION = 0.20
MAX_DRAWDOWN_HALT   = -0.15    # liquide tout si drawdown > 15%

# Ordres
MIN_TRADE_VALUE = 200.0        # ignore les ordres < 200 $


# =============================================================================
# LIVE TRADER
# =============================================================================

class LiveTrader:
    """
    Orchestre signal → risk check → exécution sur IBKR paper.

    Paramètres
    ----------
    dry_run : bool
        Si True, calcule les ordres mais ne les envoie pas.
        Utile pour tester sans risque.
    """

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run

        from src.data.pipeline import DataPipeline
        from src.factor_backtest import RollingFactorBacktest
        from src.execution.ibkr_client import IBKRClient
        from src.execution.order_manager import OrderManager
        from src.execution.risk_guard import RiskGuard

        self._pipeline = DataPipeline(cache_dir="data/cache", cache_expiry_hours=1)
        self._client   = IBKRClient()
        self._orders   = OrderManager(min_trade_value=MIN_TRADE_VALUE)
        self._guard    = RiskGuard(
            max_leverage=MAX_LEVERAGE,
            max_single_position=MAX_SINGLE_POSITION,
            max_drawdown=MAX_DRAWDOWN_HALT,
        )

    # ------------------------------------------------------------------
    # Point d'entrée principal
    # ------------------------------------------------------------------

    def run_once(self) -> None:
        """
        Exécute un cycle complet signal → ordres.
        À appeler une fois par semaine (lundi matin).
        """
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        mode = "[DRY-RUN] " if self.dry_run else ""
        logger.info(f"[LiveTrader] {mode}Démarrage — {ts}")

        # 1. Données
        logger.info("[LiveTrader] Étape 1 : Téléchargement des données...")
        _, returns = self._pipeline.run_returns(
            start=START_DATE, end=None, universe=UNIVERSE
        )
        logger.info(f"[LiveTrader] {returns.shape[1]} actifs × {returns.shape[0]} jours chargés")

        # 2. Signal sur la fenêtre la plus récente
        logger.info("[LiveTrader] Étape 2 : Calcul du signal...")
        weights = self._compute_signal(returns)
        logger.info(f"[LiveTrader] Signal calculé — {len(weights)} actifs")
        for sym, w in weights.sort_values().items():
            logger.info(f"  {sym:<6} : {w:+.4f}")

        # 3. Connexion IBKR + infos compte
        logger.info("[LiveTrader] Étape 3 : Connexion IBKR...")
        with self._client as ib:
            account_value = ib.get_net_liquidation()
            current_pos   = ib.get_positions()
            logger.info(f"[LiveTrader] Compte : ${account_value:,.2f} | {len(current_pos)} positions ouvertes")

            # 4. Risk check
            check = self._guard.check(weights, account_value)
            if not check.approved:
                logger.warning(f"[LiveTrader] Risk check ECHOUE : {check.reason}")
                tg(f"⚠️ HiddenMarketDNA — Risk check bloqué\n{check.reason}")
                if "flat" in check.reason.lower():
                    self._flatten(ib, current_pos, dry_run=self.dry_run)
                return

            # 5. Prix actuels
            logger.info("[LiveTrader] Étape 4 : Récupération des prix...")
            prices = ib.get_snapshot_prices(list(weights.index))
            missing = prices[prices.isna()].index.tolist()
            if missing:
                logger.warning(f"[LiveTrader] Prix manquants : {missing} — exclus des ordres")
                weights = weights.drop(missing, errors="ignore")
                prices  = prices.dropna()

            # 6. Calcul des ordres
            logger.info("[LiveTrader] Étape 5 : Calcul des ordres...")
            orders = self._orders.compute_orders(weights, current_pos, prices, account_value)

            if orders.empty:
                logger.info("[LiveTrader] Aucun ordre à envoyer — portefeuille déjà en place")
                tg(f"ℹ️ HiddenMarketDNA — Aucun ordre\nPortefeuille déjà en place (${account_value:,.0f})")
                return

            # 7. Exécution
            if self.dry_run:
                logger.info("[LiveTrader] DRY-RUN — ordres calculés mais non envoyés :")
                lines = []
                for sym, qty in orders.items():
                    action = "BUY " if qty > 0 else "SELL"
                    logger.info(f"  {action} {abs(qty):>5} {sym}")
                    lines.append(f"  {action} {abs(qty)} {sym}")
                tg(
                    f"🧪 HiddenMarketDNA — DRY-RUN\n"
                    f"Compte : ${account_value:,.0f} | Régime : stress\n"
                    + "\n".join(lines)
                )
            else:
                logger.info(f"[LiveTrader] Étape 6 : Envoi de {len(orders)} ordre(s)...")
                lines = []
                for sym, qty in orders.items():
                    ib.place_market_order(sym, int(qty))
                    action = "BUY " if qty > 0 else "SELL"
                    lines.append(f"  {action} {abs(qty)} {sym}")
                ib.wait_for_fills(timeout=60)
                logger.info("[LiveTrader] Tous les ordres envoyés")
                tg(
                    f"✅ HiddenMarketDNA — Ordres exécutés\n"
                    f"Compte : ${account_value:,.0f}\n"
                    + "\n".join(lines)
                )

        logger.info(f"[LiveTrader] {mode}Cycle terminé")

    # ------------------------------------------------------------------
    # Signal
    # ------------------------------------------------------------------

    def _compute_signal(self, returns) -> "pd.Series":
        """
        Calcule le signal factoriel sur la fenêtre in-sample la plus récente.
        Réutilise exactement la même logique que le backtest Sprint 3.
        """
        import numpy as np
        import pandas as pd
        from src.pca_engine import PCAEngine
        from src.factors.momentum import compute_multi_horizon_momentum
        from src.factors.composite import CompositeSignal
        from src.regimes.regime_manager import RegimeManager

        in_sample = returns.iloc[-WINDOW:]

        # ACP
        pca = PCAEngine(n_components=N_COMPONENTS)
        pca.fit(in_sample)
        loadings = pca.get_eigen_portfolios()
        pc2_load = loadings[TARGET_FACTOR]

        # Momentum
        mom_scores = compute_multi_horizon_momentum(in_sample)

        # Signal composite
        signal_builder = CompositeSignal(weights=FACTOR_WEIGHTS, normalize=True)
        breakdown = signal_builder.compute_with_breakdown(
            pca_loadings=pc2_load,
            momentum_scores=mom_scores,
            quality_scores=pd.Series(0.0, index=in_sample.columns),
        )
        signal = breakdown["weights"]

        # Régime → risk_scale
        regime_mgr = RegimeManager(
            min_scale=MIN_RISK_SCALE,
            update_freq=REGIME_UPDATE_FREQ,
        )
        regime_result = regime_mgr.force_evaluate(in_sample)
        logger.info(
            f"[LiveTrader] Régime : {regime_result.label} | "
            f"risk_scale={regime_result.risk_scale:.2f} | "
            f"stress_score={regime_result.stress_score:.2f}"
        )

        return signal * regime_result.risk_scale

    # ------------------------------------------------------------------
    # Flatten (urgence drawdown)
    # ------------------------------------------------------------------

    def _flatten(self, ib, current_positions, dry_run: bool) -> None:
        """Liquide toutes les positions ouvertes."""
        if current_positions.empty:
            logger.info("[LiveTrader] Flatten : aucune position à liquider")
            return

        logger.warning(f"[LiveTrader] FLATTEN — liquidation de {len(current_positions)} position(s)")
        for sym, qty in current_positions.items():
            if qty == 0:
                continue
            if dry_run:
                action = "BUY " if qty < 0 else "SELL"
                logger.info(f"  [DRY] {action} {abs(qty)} {sym}")
            else:
                ib.place_market_order(sym, int(-qty))

        if not dry_run:
            ib.wait_for_fills(timeout=60)
            logger.warning("[LiveTrader] Flatten terminé — compte en cash")


# =============================================================================
# CLI
# =============================================================================

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    # Silence le flood de positions/portfolio de ib_insync
    logging.getLogger("ib_insync").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(description="HiddenMarketDNA — Live Trader")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Calcule les ordres sans les envoyer (test sans risque)",
    )
    args = parser.parse_args()

    trader = LiveTrader(dry_run=args.dry_run)
    trader.run_once()


if __name__ == "__main__":
    main()
