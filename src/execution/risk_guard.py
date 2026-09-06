# src/execution/risk_guard.py
"""
Garde-fou pre-trade : bloque l'exécution si une contrainte est violée.

CHECKS EFFECTUÉS (dans l'ordre) :

  1. LEVIER TOTAL : sum(|w|) ≤ max_leverage
     Empêche d'envoyer des ordres qui dépassent la capacité du compte.

  2. POSITION MAXIMALE : max(|w_i|) ≤ max_single_position
     Évite la concentration excessive sur un seul actif.

  3. DRAWDOWN LIMITE : si la perte cumulée depuis le pic dépasse max_drawdown,
     le live trader passe en mode "flat" (liquide tout) jusqu'à récupération.
     Protège le compte paper d'une destruction totale.

  4. MARCHÉ OUVERT : vérifie qu'on n'envoie pas d'ordres le week-end ou
     les jours fériés US (basique — NYSE fermé).

USAGE :
  guard = RiskGuard(max_leverage=2.0, max_single_position=0.20, max_drawdown=-0.15)
  result = guard.check(weights, account_value, peak_value)
  if not result.approved:
      print(result.reason)
      # ne pas envoyer les ordres
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd

logger = logging.getLogger(__name__)

# Jours fériés NYSE 2025-2026 (approximatif — dates fixes)
_NYSE_HOLIDAYS = {
    "2025-01-01", "2025-01-20", "2025-02-17", "2025-04-18",
    "2025-05-26", "2025-06-19", "2025-07-04", "2025-09-01",
    "2025-11-27", "2025-12-25",
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03",
    "2026-05-25", "2026-06-19", "2026-07-03", "2026-09-07",
    "2026-11-26", "2026-12-25",
}


@dataclass
class RiskCheckResult:
    approved: bool
    reason:   str   # vide si approuvé


class RiskGuard:
    """
    Valide les poids cibles avant exécution.

    Paramètres
    ----------
    max_leverage : float
        Levier total maximum = sum(|w_i|). Défaut 2.0 (100% long + 100% short).
    max_single_position : float
        Poids absolu maximum par actif. Défaut 0.20 (20%).
    max_drawdown : float
        Drawdown maximum toléré depuis le pic (négatif). Défaut -0.15 (-15%).
        En dessous, on liquide et on attend.
    """

    def __init__(
        self,
        max_leverage:        float = 2.0,
        max_single_position: float = 0.20,
        max_drawdown:        float = -0.15,
    ):
        self.max_leverage        = max_leverage
        self.max_single_position = max_single_position
        self.max_drawdown        = max_drawdown
        self._peak_value:        float = 0.0

    # ------------------------------------------------------------------
    # Interface principale
    # ------------------------------------------------------------------

    def check(
        self,
        weights:       pd.Series,
        account_value: float,
    ) -> RiskCheckResult:
        """
        Valide les poids avant envoi des ordres.

        Paramètres
        ----------
        weights : pd.Series
            Poids cibles (index = tickers).
        account_value : float
            Valeur nette du compte en $.

        Returns
        -------
        RiskCheckResult : approved=True si tout est OK.
        """
        self._update_peak(account_value)

        checks = [
            self._check_market_open(),
            self._check_leverage(weights),
            self._check_position_size(weights),
            self._check_drawdown(account_value),
        ]

        for result in checks:
            if not result.approved:
                logger.warning(f"[RiskGuard] BLOQUE — {result.reason}")
                return result

        logger.info(f"[RiskGuard] OK — compte ${account_value:,.0f} | levier {weights.abs().sum():.2f}x")
        return RiskCheckResult(approved=True, reason="")

    def reset_peak(self, value: float) -> None:
        """Réinitialise le pic (ex: début d'une nouvelle session)."""
        self._peak_value = value

    # ------------------------------------------------------------------
    # Checks individuels
    # ------------------------------------------------------------------

    def _check_market_open(self) -> RiskCheckResult:
        now = datetime.now(timezone.utc)
        if now.weekday() >= 5:   # samedi=5, dimanche=6
            return RiskCheckResult(False, f"Marché fermé — week-end ({now.strftime('%A')})")
        date_str = now.strftime("%Y-%m-%d")
        if date_str in _NYSE_HOLIDAYS:
            return RiskCheckResult(False, f"Marché fermé — jour férié NYSE ({date_str})")
        return RiskCheckResult(True, "")

    def _check_leverage(self, weights: pd.Series) -> RiskCheckResult:
        leverage = weights.abs().sum()
        if leverage > self.max_leverage:
            return RiskCheckResult(
                False,
                f"Levier {leverage:.2f}x > maximum {self.max_leverage}x"
            )
        return RiskCheckResult(True, "")

    def _check_position_size(self, weights: pd.Series) -> RiskCheckResult:
        max_pos = weights.abs().max()
        if max_pos > self.max_single_position:
            worst = weights.abs().idxmax()
            return RiskCheckResult(
                False,
                f"Position {worst} = {max_pos:.1%} > max {self.max_single_position:.1%}"
            )
        return RiskCheckResult(True, "")

    def _check_drawdown(self, account_value: float) -> RiskCheckResult:
        if self._peak_value <= 0:
            return RiskCheckResult(True, "")
        dd = (account_value - self._peak_value) / self._peak_value
        if dd < self.max_drawdown:
            return RiskCheckResult(
                False,
                f"Drawdown {dd:.1%} < limite {self.max_drawdown:.1%} — passage en flat"
            )
        return RiskCheckResult(True, "")

    def _update_peak(self, account_value: float) -> None:
        if account_value > self._peak_value:
            self._peak_value = account_value
