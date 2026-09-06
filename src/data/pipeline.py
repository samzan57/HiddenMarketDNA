# src/data/pipeline.py
"""
Orchestrateur du pipeline de données — point d'entrée unique.

Principe : tout le code qui a besoin de données de marché passe par ici.
On ne télécharge jamais directement depuis un autre module.

Flux :
    DataPipeline.run()
        └─ MarketDataFetcher.fetch()    → prix bruts (depuis cache ou Yahoo)
        └─ MarketDataCleaner.clean()    → prix nettoyés + rapport qualité
        └─ compute_log_returns()        → rendements log (optionnel)
"""

import logging
from typing import List, Optional, Tuple

import pandas as pd

from src.data.cleaner import DataQualityReport, MarketDataCleaner
from src.data.fetcher import MarketDataFetcher
from src.data.universe import MULTI_ASSET, SECTOR_ETFS
from src.returns import compute_log_returns

logger = logging.getLogger(__name__)


class DataPipeline:
    """
    Pipeline complet : téléchargement → nettoyage → DataFrame prêt pour backtest.

    C'est L'UNIQUE point d'entrée pour obtenir des données dans tout le projet.
    Avantages :
    - Centralisation : un seul endroit à modifier si on change de source de données
    - Cache automatique : les données sont stockées localement
    - Qualité garantie : le cleaner s'assure de la cohérence des données

    Paramètres
    ----------
    cache_dir : str
        Répertoire du cache local (créé automatiquement si inexistant).
    cache_expiry_hours : int
        Durée de vie du cache. 24h = données "intraday fresh".
        Mettre 0 pour désactiver le cache (force_refresh à chaque fois).
    min_history_ratio : float
        Transmis au cleaner : fraction minimale d'historique valide par actif.
    """

    def __init__(
        self,
        cache_dir: str = "data/cache",
        cache_expiry_hours: int = 24,
        min_history_ratio: float = 0.90,
    ):
        self.fetcher = MarketDataFetcher(
            cache_dir=cache_dir,
            cache_expiry_hours=cache_expiry_hours,
        )
        self.cleaner = MarketDataCleaner(
            min_history_ratio=min_history_ratio,
        )
        # Stocké après chaque run pour permettre l'audit
        self._last_report: Optional[DataQualityReport] = None

    @property
    def last_quality_report(self) -> Optional[DataQualityReport]:
        """Rapport qualité du dernier run — utile pour le debug."""
        return self._last_report

    # ------------------------------------------------------------------
    # Interface principale
    # ------------------------------------------------------------------

    def run(
        self,
        tickers: Optional[List[str]] = None,
        start: str = "2010-01-01",
        end: Optional[str] = None,
        force_refresh: bool = False,
        universe: str = "multi_asset",
    ) -> pd.DataFrame:
        """
        Exécute le pipeline et retourne les PRIX ajustés nettoyés.

        Pour obtenir les rendements, appeler ensuite compute_log_returns().
        Ou utiliser directement run_returns() ci-dessous.

        Paramètres
        ----------
        tickers : list[str], optional
            Liste manuelle de tickers. Si None, utilise l'univers 'universe'.
        start : str
            Date de début "YYYY-MM-DD". Plus on remonte, plus les données
            sont riches pour le backtest.
        end : str, optional
            Date de fin (défaut : aujourd'hui).
        force_refresh : bool
            True = ignore le cache, re-télécharge depuis Yahoo Finance.
        universe : str
            Univers prédéfini si tickers=None.
            'multi_asset' (défaut) | 'sector_etfs' | 'all'

        Retourne
        --------
        pd.DataFrame
            index = DatetimeIndex (jours de bourse)
            colonnes = tickers disponibles et valides
        """
        # Sélection de l'univers si pas de tickers manuels
        if tickers is None:
            tickers = self._resolve_universe(universe)

        logger.info(
            f"[Pipeline] Démarrage — {len(tickers)} actifs "
            f"| {start} → {end or 'today'} "
            f"| univers='{universe}'"
        )

        # --- Étape 1 : Fetch (cache ou Yahoo Finance) ---
        raw_prices = self.fetcher.fetch(
            tickers=tickers,
            start=start,
            end=end,
            force_refresh=force_refresh,
        )

        # --- Étape 2 : Nettoyage ---
        clean_prices, report = self.cleaner.clean(raw_prices)
        self._last_report = report

        # Avertissement si des tickers demandés ont été perdus
        available = set(clean_prices.columns)
        requested = set(tickers)
        lost = requested - available
        if lost:
            logger.warning(
                f"[Pipeline] {len(lost)} actif(s) perdus après nettoyage : {lost}"
            )

        logger.info(
            f"[Pipeline] Terminé — {clean_prices.shape[1]} actifs × "
            f"{clean_prices.shape[0]} jours"
        )

        return clean_prices

    def run_returns(
        self,
        tickers: Optional[List[str]] = None,
        start: str = "2010-01-01",
        end: Optional[str] = None,
        force_refresh: bool = False,
        universe: str = "multi_asset",
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Raccourci : retourne (prix, rendements_log) en une seule ligne.

        Exemple d'utilisation :
            pipeline = DataPipeline()
            prices, returns = pipeline.run_returns(start="2015-01-01")

        Retourne
        --------
        (prices, log_returns) : deux DataFrames alignés
        """
        prices = self.run(
            tickers=tickers,
            start=start,
            end=end,
            force_refresh=force_refresh,
            universe=universe,
        )
        log_returns = compute_log_returns(prices)
        return prices, log_returns

    # ------------------------------------------------------------------
    # Utilitaires
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_universe(universe: str) -> List[str]:
        """Mappe le nom d'univers vers la liste de tickers correspondante."""
        mapping = {
            "multi_asset":  MULTI_ASSET,
            "sector_etfs":  SECTOR_ETFS,
        }
        if universe not in mapping:
            raise ValueError(
                f"Univers '{universe}' inconnu. "
                f"Choisir parmi : {list(mapping.keys())}"
            )
        return mapping[universe]

    def print_report(self) -> None:
        """Affiche le dernier rapport qualité. Appeler après run()."""
        if self._last_report is None:
            print("Aucun run effectué pour l'instant.")
        else:
            print(self._last_report.summary())
