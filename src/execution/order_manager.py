# src/execution/order_manager.py
"""
Convertit les poids cibles en ordres d'actions à passer.

LOGIQUE :
  1. Poids cibles (float) × valeur nette du compte → valeur $ cible par actif
  2. Valeur $ cible / prix actuel → nombre d'actions cible (arrondi à l'entier)
  3. Delta = cible - position actuelle
  4. Filtre : ignore les deltas < min_trade_value $ (évite les micro-ordres)

EXEMPLE :
  Compte = 100 000 $, poids XLK = +0.15, prix XLK = 200 $
  → valeur cible = 15 000 $
  → actions cibles = 75
  → si position actuelle = 60 → delta = +15 (acheter 15 XLK)
"""

import logging
from typing import Dict

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class OrderManager:
    """
    Convertit les poids en ordres entiers.

    Paramètres
    ----------
    min_trade_value : float
        Valeur minimale d'un ordre en $ (ignore les petits deltas).
        Défaut : 200 $ (évite les frais disproportionnés).
    """

    def __init__(self, min_trade_value: float = 200.0):
        self.min_trade_value = min_trade_value

    # ------------------------------------------------------------------
    # Interface principale
    # ------------------------------------------------------------------

    def compute_orders(
        self,
        target_weights:   pd.Series,
        current_positions: pd.Series,
        prices:            pd.Series,
        account_value:     float,
    ) -> pd.Series:
        """
        Calcule les ordres (delta d'actions entières) à envoyer.

        Paramètres
        ----------
        target_weights : pd.Series
            Poids cibles (index = tickers, valeurs ∈ [-1, 1]).
        current_positions : pd.Series
            Positions actuelles en nombre d'actions.
        prices : pd.Series
            Prix actuels par ticker.
        account_value : float
            Valeur nette du compte en $.

        Returns
        -------
        pd.Series
            Delta d'actions par ticker (int). Positif = achat, négatif = vente.
            Ne contient que les tickers avec delta != 0.
        """
        all_symbols = target_weights.index.union(current_positions.index)

        target_shares  = self._weights_to_shares(target_weights, prices, account_value)
        current_shares = current_positions.reindex(all_symbols).fillna(0.0)

        delta = (target_shares.reindex(all_symbols).fillna(0.0) - current_shares).round().astype(int)

        # Filtre les micro-ordres
        delta_values = delta.abs() * prices.reindex(delta.index).fillna(0.0)
        delta = delta[delta_values >= self.min_trade_value]

        self._log_orders(delta, prices, account_value)
        return delta[delta != 0]

    def target_shares(
        self,
        target_weights: pd.Series,
        prices:         pd.Series,
        account_value:  float,
    ) -> pd.Series:
        """Expose le calcul poids → actions (utile pour les tests)."""
        return self._weights_to_shares(target_weights, prices, account_value)

    # ------------------------------------------------------------------
    # Interne
    # ------------------------------------------------------------------

    def _weights_to_shares(
        self,
        weights:       pd.Series,
        prices:        pd.Series,
        account_value: float,
    ) -> pd.Series:
        """
        Convertit les poids en nombre d'actions entières.

        Formule : shares_i = floor(w_i × NAV / price_i)
        Le floor (pas round) conserve le levier ≤ target.
        """
        aligned_prices = prices.reindex(weights.index)
        valid = aligned_prices > 0

        shares = pd.Series(0, index=weights.index, dtype=int)
        shares[valid] = (
            weights[valid] * account_value / aligned_prices[valid]
        ).apply(np.fix).astype(int)   # truncate vers zéro (pas floor) pour long/short symétrique

        return shares

    def _log_orders(self, delta: pd.Series, prices: pd.Series, account_value: float) -> None:
        if delta.empty:
            logger.info("[OrderManager] Aucun ordre à envoyer (tout dans la tolérance)")
            return

        total_turnover = (delta.abs() * prices.reindex(delta.index).fillna(0.0)).sum()
        logger.info(
            f"[OrderManager] {len(delta)} ordre(s) | "
            f"Turnover estimé : ${total_turnover:,.0f} "
            f"({100 * total_turnover / account_value:.1f}% NAV)"
        )
        for sym, qty in delta.items():
            price  = prices.get(sym, float("nan"))
            value  = abs(qty) * price
            action = "BUY " if qty > 0 else "SELL"
            logger.info(f"  {action} {abs(qty):>5} {sym:<6}  @ ${price:>8.2f}  = ${value:>10,.0f}")
