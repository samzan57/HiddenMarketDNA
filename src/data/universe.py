# src/data/universe.py
"""
Définition de l'univers d'investissement : 40+ actifs couvrant toutes les
classes d'actifs principales (actions, obligations, matières premières, etc.)

Pourquoi centraliser ici ?
- Un seul endroit à modifier pour ajouter/retirer un actif
- Les métadonnées (secteur, classe d'actif) servent plus tard pour
  l'attribution de performance et les contraintes d'optimisation
"""

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Asset:
    """Métadonnées d'un actif financier."""
    ticker: str
    name: str
    asset_class: str  # equity | bond | commodity | volatility | real_estate
    sector: str       # sous-catégorie (ex: Technology, Long Duration, Gold...)
    region: str       # US | International | Global


# =============================================================================
# UNIVERS COMPLET — 40+ actifs
# =============================================================================
UNIVERSE: Dict[str, Asset] = {

    # -------------------------------------------------------------------------
    # ETFs Sectoriels US (SPDR Select Sector) — les 11 secteurs du S&P 500
    # -------------------------------------------------------------------------
    "XLK":  Asset("XLK",  "Technology Select SPDR",           "equity", "Technology",            "US"),
    "XLF":  Asset("XLF",  "Financial Select SPDR",            "equity", "Financials",             "US"),
    "XLE":  Asset("XLE",  "Energy Select SPDR",               "equity", "Energy",                 "US"),
    "XLV":  Asset("XLV",  "Health Care Select SPDR",          "equity", "Health Care",            "US"),
    "XLI":  Asset("XLI",  "Industrial Select SPDR",           "equity", "Industrials",            "US"),
    "XLY":  Asset("XLY",  "Consumer Discretionary SPDR",      "equity", "Consumer Discretionary", "US"),
    "XLP":  Asset("XLP",  "Consumer Staples Select SPDR",     "equity", "Consumer Staples",       "US"),
    "XLU":  Asset("XLU",  "Utilities Select SPDR",            "equity", "Utilities",              "US"),
    "XLRE": Asset("XLRE", "Real Estate Select SPDR",          "equity", "Real Estate",            "US"),
    "XLB":  Asset("XLB",  "Materials Select SPDR",            "equity", "Materials",              "US"),
    "XLC":  Asset("XLC",  "Communication Services SPDR",      "equity", "Communication",          "US"),

    # -------------------------------------------------------------------------
    # Indices larges US — benchmarks de référence
    # -------------------------------------------------------------------------
    "SPY":  Asset("SPY",  "SPDR S&P 500 ETF",                 "equity", "Large Cap Blend",        "US"),
    "QQQ":  Asset("QQQ",  "Invesco Nasdaq-100 ETF",           "equity", "Large Cap Growth",       "US"),
    "IWM":  Asset("IWM",  "iShares Russell 2000 ETF",         "equity", "Small Cap",              "US"),
    "MDY":  Asset("MDY",  "SPDR S&P MidCap 400 ETF",         "equity", "Mid Cap",                "US"),
    "DIA":  Asset("DIA",  "SPDR Dow Jones ETF",               "equity", "Large Cap Value",        "US"),

    # -------------------------------------------------------------------------
    # Actions internationales — diversification géographique
    # -------------------------------------------------------------------------
    "EFA":  Asset("EFA",  "iShares MSCI EAFE ETF",            "equity", "International Dev",      "International"),
    "EEM":  Asset("EEM",  "iShares MSCI Emerging Markets",    "equity", "Emerging Markets",       "International"),
    "VEA":  Asset("VEA",  "Vanguard FTSE Developed Markets",  "equity", "International Dev",      "International"),
    "FXI":  Asset("FXI",  "iShares China Large-Cap ETF",      "equity", "China",                  "International"),
    "EWJ":  Asset("EWJ",  "iShares MSCI Japan ETF",           "equity", "Japan",                  "International"),
    "EWZ":  Asset("EWZ",  "iShares MSCI Brazil ETF",          "equity", "Brazil",                 "International"),

    # -------------------------------------------------------------------------
    # Obligations US — le "risk-off" du portefeuille
    # Corrélation négative avec les actions en période de stress = couverture naturelle
    # -------------------------------------------------------------------------
    "TLT":  Asset("TLT",  "iShares 20+ Year Treasury Bond",   "bond",   "Long Duration",          "US"),
    "IEF":  Asset("IEF",  "iShares 7-10 Year Treasury Bond",  "bond",   "Intermediate Duration",  "US"),
    "SHY":  Asset("SHY",  "iShares 1-3 Year Treasury Bond",   "bond",   "Short Duration",         "US"),
    "LQD":  Asset("LQD",  "iShares IG Corporate Bond ETF",    "bond",   "Investment Grade Corp",  "US"),
    "HYG":  Asset("HYG",  "iShares High Yield Corp Bond",     "bond",   "High Yield",             "US"),
    "TIP":  Asset("TIP",  "iShares TIPS Bond ETF",            "bond",   "Inflation Protected",    "US"),
    "AGG":  Asset("AGG",  "iShares Core US Aggregate Bond",   "bond",   "Aggregate Bond",         "US"),

    # -------------------------------------------------------------------------
    # Matières premières — diversification + hedge inflation
    # -------------------------------------------------------------------------
    "GLD":  Asset("GLD",  "SPDR Gold Shares",                 "commodity", "Gold",                "Global"),
    "SLV":  Asset("SLV",  "iShares Silver Trust",             "commodity", "Silver",              "Global"),
    "USO":  Asset("USO",  "United States Oil Fund",           "commodity", "Oil",                 "Global"),
    "DBA":  Asset("DBA",  "Invesco DB Agriculture Fund",      "commodity", "Agriculture",         "Global"),
    "PDBC": Asset("PDBC", "Invesco Optimum Yield Commodity",  "commodity", "Broad Commodity",     "Global"),

    # -------------------------------------------------------------------------
    # Immobilier coté (REITs) — revenu + inflation hedge
    # -------------------------------------------------------------------------
    "VNQ":  Asset("VNQ",  "Vanguard Real Estate ETF",         "real_estate", "US REITs",          "US"),
    "REM":  Asset("REM",  "iShares Mortgage Real Estate",     "real_estate", "Mortgage REITs",    "US"),

    # -------------------------------------------------------------------------
    # Volatilité — instrument de couverture en cas de choc de marché
    # VIXY monte fortement quand le VIX monte → protection en crise
    # -------------------------------------------------------------------------
    "VIXY": Asset("VIXY", "ProShares VIX Short-Term Futures", "volatility", "VIX ST",             "US"),
}


# =============================================================================
# SOUS-UNIVERS PRÉDÉFINIS — utilisés par les différentes stratégies
# =============================================================================

# Les 11 secteurs SPDR (base de la stratégie ACP actuelle)
SECTOR_ETFS: List[str] = [
    "XLK", "XLF", "XLE", "XLV", "XLI",
    "XLY", "XLP", "XLU", "XLRE", "XLB", "XLC",
]

# Indices larges US uniquement
BROAD_US_EQUITY: List[str] = ["SPY", "QQQ", "IWM", "MDY", "DIA"]

# Univers multi-asset (actions + oblig + commo + reits) — stratégie principale
MULTI_ASSET: List[str] = (
    SECTOR_ETFS
    + ["SPY", "QQQ", "IWM"]           # indices larges
    + ["EFA", "EEM"]                   # international
    + ["TLT", "IEF", "HYG", "TIP"]    # obligations
    + ["GLD", "USO"]                   # matières premières
    + ["VNQ"]                          # immobilier
)

# Tout l'univers
ALL_TICKERS: List[str] = list(UNIVERSE.keys())


# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

def get_tickers_by_class(asset_class: str) -> List[str]:
    """
    Retourne tous les tickers d'une classe d'actifs.
    asset_class ∈ {'equity', 'bond', 'commodity', 'real_estate', 'volatility'}
    """
    return [t for t, a in UNIVERSE.items() if a.asset_class == asset_class]


def get_tickers_by_region(region: str) -> List[str]:
    """Retourne tous les tickers d'une région géographique."""
    return [t for t, a in UNIVERSE.items() if a.region == region]


def get_asset_metadata(ticker: str) -> Asset:
    """Retourne les métadonnées d'un actif. Lève KeyError si inconnu."""
    if ticker not in UNIVERSE:
        raise KeyError(f"Ticker '{ticker}' absent de l'univers. Ajouter dans universe.py.")
    return UNIVERSE[ticker]


def describe_universe() -> None:
    """Affiche un résumé de l'univers par classe d'actifs (utile en debug)."""
    from collections import Counter
    counts = Counter(a.asset_class for a in UNIVERSE.values())
    print(f"Univers total : {len(UNIVERSE)} actifs")
    for cls, n in sorted(counts.items()):
        print(f"  {cls:<15} : {n} actifs")
