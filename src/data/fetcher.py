# src/data/fetcher.py
"""
Téléchargement de données de marché via Yahoo Finance (yfinance) avec :

1. CACHE LOCAL (Parquet) — évite de re-télécharger à chaque run.
   Parquet est un format colonne ultra-rapide, standard en data engineering.

2. RETRY + BACKOFF EXPONENTIEL — robustesse réseau.
   Si Yahoo Finance est lent ou timeout → on réessaie automatiquement.

3. LOGGING STRUCTURÉ — chaque étape est tracée pour le debug.
"""

import hashlib
import logging
import os
import time
from datetime import datetime
from typing import List, Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class MarketDataFetcher:
    """
    Télécharge les prix de clôture ajustés depuis Yahoo Finance.

    Le cache est stocké en .parquet dans cache_dir, nommé par un hash MD5
    des paramètres (tickers + dates) pour éviter les collisions.

    Paramètres
    ----------
    cache_dir : str
        Répertoire de stockage du cache.
    cache_expiry_hours : int
        Au-delà de cette durée, le cache est considéré périmé et retéléchargé.
    max_retries : int
        Nombre max de tentatives en cas d'erreur réseau.
    """

    def __init__(
        self,
        cache_dir: str = "data/cache",
        cache_expiry_hours: int = 24,
        max_retries: int = 3,
    ):
        self.cache_dir = cache_dir
        self.cache_expiry_hours = cache_expiry_hours
        self.max_retries = max_retries
        os.makedirs(cache_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Gestion du cache
    # ------------------------------------------------------------------

    def _cache_path(self, tickers: List[str], start: str, end: str) -> str:
        """
        Chemin unique du fichier cache pour ces paramètres.
        On hache les paramètres en MD5 → nom de fichier court et stable.
        """
        key = "_".join(sorted(tickers)) + f"_{start}_{end}"
        h = hashlib.md5(key.encode()).hexdigest()[:12]
        return os.path.join(self.cache_dir, f"prices_{h}.parquet")

    def _is_cache_valid(self, path: str) -> bool:
        """
        Retourne True si le fichier cache existe et a moins de
        cache_expiry_hours heures.
        """
        if not os.path.exists(path):
            return False
        age_hours = (time.time() - os.path.getmtime(path)) / 3600
        return age_hours < self.cache_expiry_hours

    # ------------------------------------------------------------------
    # Téléchargement avec retry
    # ------------------------------------------------------------------

    def _download_with_retry(
        self, tickers: List[str], start: str, end: str
    ) -> pd.DataFrame:
        """
        Télécharge via yfinance avec stratégie de retry exponentielle.

        Backoff : 1s → 2s → 4s entre les tentatives.
        Si toutes les tentatives échouent, lève RuntimeError.
        """
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(
                    f"[Fetch] {len(tickers)} actifs | {start} → {end} "
                    f"(tentative {attempt}/{self.max_retries})"
                )

                # auto_adjust=True : les prix 'Close' sont déjà ajustés
                # pour les dividendes et les splits — essentiel pour le
                # calcul des rendements historiques.
                raw = yf.download(
                    tickers=tickers,
                    start=start,
                    end=end,
                    auto_adjust=True,
                    progress=False,
                    threads=True,
                )

                if raw.empty:
                    raise ValueError("yfinance a retourné un DataFrame vide.")

                # yfinance retourne des colonnes multi-niveaux quand il y a
                # plusieurs tickers : ('Close', 'SPY'), ('Close', 'QQQ'), etc.
                # → On extrait uniquement le niveau 'Close'.
                if isinstance(raw.columns, pd.MultiIndex):
                    prices = raw["Close"].copy()
                else:
                    # Cas d'un seul ticker : colonnes simples
                    prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

                # Supprimer les colonnes entièrement vides
                prices = prices.dropna(axis=1, how="all")

                logger.info(
                    f"[Fetch] OK — {prices.shape[1]} actifs récupérés, "
                    f"{prices.shape[0]} jours"
                )
                return prices

            except Exception as exc:
                wait = 2 ** (attempt - 1)  # 1s, 2s, 4s
                logger.warning(
                    f"[Fetch] Tentative {attempt} échouée : {exc}. "
                    f"Attente {wait}s avant retry..."
                )
                if attempt == self.max_retries:
                    raise RuntimeError(
                        f"Impossible de télécharger les données après "
                        f"{self.max_retries} tentatives."
                    ) from exc
                time.sleep(wait)

    # ------------------------------------------------------------------
    # Interface publique
    # ------------------------------------------------------------------

    def fetch(
        self,
        tickers: List[str],
        start: str,
        end: Optional[str] = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Point d'entrée principal : retourne un DataFrame de prix ajustés.

        Logique :
          1. Calcule le chemin de cache
          2. Si cache valide et force_refresh=False → lecture directe (rapide)
          3. Sinon → téléchargement + écriture en cache

        Paramètres
        ----------
        tickers : list[str]
            Ex: ["SPY", "TLT", "GLD"]
        start : str
            Date de début au format "YYYY-MM-DD"
        end : str, optional
            Date de fin (défaut : aujourd'hui)
        force_refresh : bool
            True = ignore le cache et retélécharge depuis Yahoo Finance

        Retourne
        --------
        pd.DataFrame
            index = DatetimeIndex (jours de bourse)
            colonnes = tickers disponibles (certains peuvent manquer si Yahoo
            ne les a pas)
        """
        if end is None:
            end = datetime.today().strftime("%Y-%m-%d")

        cache_path = self._cache_path(tickers, start, end)

        if not force_refresh and self._is_cache_valid(cache_path):
            logger.info(f"[Cache] Lecture depuis : {cache_path}")
            return pd.read_parquet(cache_path)

        # Téléchargement
        prices = self._download_with_retry(tickers, start, end)

        # Persistance en Parquet (format colonne compressé, ~10x plus rapide
        # que CSV en lecture, et ~5x plus petit)
        prices.to_parquet(cache_path)
        logger.info(f"[Cache] Sauvegardé : {cache_path}")

        return prices

    def invalidate_cache(self) -> int:
        """
        Supprime tous les fichiers cache.
        Utile si on veut forcer un refresh complet de toutes les données.
        Retourne le nombre de fichiers supprimés.
        """
        count = 0
        for fname in os.listdir(self.cache_dir):
            if fname.endswith(".parquet"):
                os.remove(os.path.join(self.cache_dir, fname))
                count += 1
        logger.info(f"[Cache] {count} fichier(s) supprimé(s).")
        return count
