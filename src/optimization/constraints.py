# src/optimization/constraints.py
"""
Contraintes de portefeuille — standard institutionnel.

POURQUOI DES CONTRAINTES ?
  Sans contraintes, un optimiseur Markowitz produit des poids extrêmes :
  +200% sur un actif, -150% sur un autre. C'est mathématiquement optimal
  mais impossible en pratique (manque de liquidité, appels de marge, etc.)

  Les contraintes reproduisent les limites réelles d'un fonds :
    - Limite de position : pas plus de X% sur un seul actif
    - Levier max : sum(|w|) ≤ 2 = max 200% de capital engagé
    - Neutralité marché : sum(w) ≈ 0 (pas de biais directionnel)
    - Turnover : limite les changements de positions → coûts de transaction
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class PortfolioConstraints:
    """
    Contraintes appliquées à chaque step d'optimisation du portefeuille.

    Paramètres
    ----------
    max_position : float
        Poids absolu maximum par actif.
        0.15 = max 15% long ou 15% short sur un actif individuel.
        Citadel utilise typiquement 5-10% pour les gros fonds.

    min_position : float
        Poids minimum par actif (négatif = short autorisé).
        -max_position par défaut (portefeuille symétrique long/short).

    max_leverage : float
        Levier total = sum(|w_i|).
        1.0 = 100% engagé (pas de levier) → conservateur
        2.0 = 200% = standard des fonds long/short equity

    market_neutral : bool
        Si True : sum(w_i) = 0 → pas d'exposition beta nette.
        Le portefeuille ne parie pas sur la direction du marché.

    max_turnover : float, optional
        Changement maximum de poids total entre deux rebalancements.
        None = pas de contrainte de turnover.
        0.20 = max 20% des poids peuvent changer → limite les frais.

    long_only : bool
        Si True : w_i ≥ 0 pour tous les actifs (pas de shorts).
        Utile pour des portefeuilles grand public ou sur compte retail.
    """
    max_position:   float           = 0.15
    min_position:   Optional[float] = None     # défaut : -max_position
    max_leverage:   float           = 1.0
    market_neutral: bool            = True
    max_turnover:   Optional[float] = None
    long_only:      bool            = False

    def __post_init__(self):
        if self.long_only:
            self.min_position = 0.0
        elif self.min_position is None:
            self.min_position = -self.max_position

        if self.max_position <= 0:
            raise ValueError("max_position doit être > 0")
        if self.max_leverage <= 0:
            raise ValueError("max_leverage doit être > 0")

    @classmethod
    def long_short_standard(cls) -> "PortfolioConstraints":
        """
        Profil standard d'un fonds long/short equity institutionnel.
        Levier 2x, max 15% par actif, market-neutral.
        """
        return cls(max_position=0.15, max_leverage=2.0, market_neutral=True)

    @classmethod
    def conservative(cls) -> "PortfolioConstraints":
        """
        Profil conservateur : pas de levier, max 20% par actif.
        Adapté au trading sur compte demo IBKR.
        """
        return cls(max_position=0.20, max_leverage=1.0, market_neutral=True)

    @classmethod
    def long_only_etf(cls) -> "PortfolioConstraints":
        """
        Portefeuille long only sur ETFs, adapté aux contraintes retail.
        """
        return cls(max_position=0.30, max_leverage=1.0, market_neutral=False, long_only=True)
