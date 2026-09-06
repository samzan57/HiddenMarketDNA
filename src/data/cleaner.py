# src/data/cleaner.py
"""
Nettoyage institutionnel des données de marché brutes.

Problèmes réels que ce module traite :
  - Yahoo Finance retourne parfois des prix aberrants (ex: -1, 0, ou un spike
    de +50% sur un ETF liquide → erreur de données, pas un vrai mouvement)
  - Certains actifs ont des jours manquants (fériés locaux, suspension de cotation)
  - Les données de différents actifs ne sont pas forcément alignées sur les
    mêmes dates

Pipeline de nettoyage (dans l'ordre) :
  1. Suppression des actifs avec trop peu d'historique
  2. Interpolation des petits gaps (≤ N jours consécutifs)
  3. Détection et correction des outliers (z-score sur rendements)
  4. Suppression des lignes restantes avec NaN
  5. Normalisation en float64
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# Rapport de qualité — produit par le cleaner après nettoyage
# =============================================================================

@dataclass
class DataQualityReport:
    """
    Résumé des opérations de nettoyage effectuées.
    Permet d'auditer la qualité des données avant de lancer un backtest.
    """
    tickers_dropped: List[str] = field(default_factory=list)
    outliers_fixed: Dict[str, int] = field(default_factory=dict)
    gaps_interpolated: Dict[str, int] = field(default_factory=dict)
    final_shape: Tuple[int, int] = (0, 0)
    date_range: Tuple[str, str] = ("", "")

    def summary(self) -> str:
        """Retourne un résumé lisible du rapport."""
        total_outliers = sum(self.outliers_fixed.values())
        total_gaps = sum(self.gaps_interpolated.values())
        lines = [
            "=" * 50,
            "   RAPPORT QUALITÉ DES DONNÉES",
            "=" * 50,
            f"  Période     : {self.date_range[0]}  →  {self.date_range[1]}",
            f"  Shape finale : {self.final_shape[0]} jours × {self.final_shape[1]} actifs",
            f"  Actifs supprimés   : {self.tickers_dropped or 'aucun'}",
            f"  Outliers corrigés  : {total_outliers}",
            f"  Gaps interpolés    : {total_gaps} jours",
            "=" * 50,
        ]
        return "\n".join(lines)


# =============================================================================
# Cleaner principal
# =============================================================================

class MarketDataCleaner:
    """
    Nettoie un DataFrame de prix bruts pour le rendre utilisable en backtest.

    Paramètres
    ----------
    min_history_ratio : float
        Fraction minimale de données non-nulles requises par actif.
        Ex: 0.90 → on garde un actif seulement s'il a ≥ 90% de ses jours.
        Les ETFs récents (XLRE, XLC) ont moins d'historique que les anciens.

    outlier_std_threshold : float
        Seuil en nombre d'écarts-types pour détecter un outlier sur
        les rendements journaliers.
        5σ correspond à une probabilité de ~3×10⁻⁷ sous loi normale :
        si on observe ça, c'est très probablement une erreur de données.

    max_consecutive_nans : int
        Nombre max de jours consécutifs manquants qu'on accepte d'interpoler.
        3 jours = ok (long week-end, férié)
        20 jours = non (suspension de cotation → on préfère supprimer)
    """

    def __init__(
        self,
        min_history_ratio: float = 0.90,
        outlier_std_threshold: float = 5.0,
        max_consecutive_nans: int = 3,
    ):
        self.min_history_ratio = min_history_ratio
        self.outlier_std_threshold = outlier_std_threshold
        self.max_consecutive_nans = max_consecutive_nans

    # ------------------------------------------------------------------
    # Pipeline principal
    # ------------------------------------------------------------------

    def clean(
        self, prices: pd.DataFrame
    ) -> Tuple[pd.DataFrame, DataQualityReport]:
        """
        Exécute le pipeline de nettoyage complet.

        Retourne
        --------
        (DataFrame nettoyé, DataQualityReport)
        """
        report = DataQualityReport()
        df = prices.copy()

        # S'assurer que l'index est bien en DatetimeIndex trié
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        # --- Étape 1 : Supprimer les actifs avec historique insuffisant ---
        df, report = self._drop_insufficient_history(df, report)

        # --- Étape 2 : Interpoler les petits gaps ---
        df, report = self._interpolate_small_gaps(df, report)

        # --- Étape 3 : Corriger les outliers ---
        df, report = self._fix_outliers(df, report)

        # --- Étape 4 : Supprimer les lignes avec NaN restants ---
        n_before = len(df)
        df = df.dropna(how="any")
        n_dropped = n_before - len(df)
        if n_dropped > 0:
            logger.info(f"[Cleaner] {n_dropped} ligne(s) supprimée(s) (NaN résiduels).")

        # --- Étape 5 : Forcer float64 ---
        df = df.astype("float64")

        # Rapport final
        report.final_shape = df.shape
        if len(df) > 0:
            report.date_range = (
                df.index[0].strftime("%Y-%m-%d"),
                df.index[-1].strftime("%Y-%m-%d"),
            )

        logger.info(report.summary())
        return df, report

    # ------------------------------------------------------------------
    # Étapes internes
    # ------------------------------------------------------------------

    def _drop_insufficient_history(
        self, df: pd.DataFrame, report: DataQualityReport
    ) -> Tuple[pd.DataFrame, DataQualityReport]:
        """
        Supprime les colonnes dont le ratio de données valides est trop bas.

        Cas typique : un ETF lancé en 2015 dans un univers qui démarre en 2010
        → 5 ans de NaN au début → ratio ~50% → supprimé si threshold=0.90.
        Solution : soit baisser min_history_ratio, soit réduire la date de début.
        """
        total_rows = len(df)
        if total_rows == 0:
            return df, report

        valid_ratio = df.notna().mean()  # fraction de valeurs non-NaN par colonne
        to_drop = valid_ratio[valid_ratio < self.min_history_ratio].index.tolist()

        if to_drop:
            logger.warning(
                f"[Cleaner] Actifs supprimés (historique < "
                f"{self.min_history_ratio:.0%}) : {to_drop}"
            )
            df = df.drop(columns=to_drop)
            report.tickers_dropped.extend(to_drop)

        return df, report

    def _interpolate_small_gaps(
        self, df: pd.DataFrame, report: DataQualityReport
    ) -> Tuple[pd.DataFrame, DataQualityReport]:
        """
        Interpole linéairement les petits trous dans les séries de prix.

        Pourquoi l'interpolation linéaire ?
        Sur des données quotidiennes, un jour manquant entre deux prix
        connus est mieux estimé par la moyenne que par un forward-fill
        (qui introduirait un biais directionnel).

        Le paramètre `limit` évite d'interpoler de longs trous qui
        correspondraient à une vraie absence de cotation.
        """
        for col in df.columns:
            n_missing_before = df[col].isna().sum()
            if n_missing_before == 0:
                continue

            df[col] = df[col].interpolate(
                method="linear",
                limit=self.max_consecutive_nans,
                limit_direction="forward",
            )

            n_fixed = int(n_missing_before - df[col].isna().sum())
            if n_fixed > 0:
                report.gaps_interpolated[col] = n_fixed
                logger.debug(f"[Cleaner] {col} : {n_fixed} gap(s) interpolé(s).")

        return df, report

    def _fix_outliers(
        self, df: pd.DataFrame, report: DataQualityReport
    ) -> Tuple[pd.DataFrame, DataQualityReport]:
        """
        Détecte les erreurs de données via le z-score des rendements journaliers.

        Stratégie de correction : remplacement par la moyenne des prix
        voisins (j-1 et j+1) — interpolation ponctuelle.

        Exemple réel : Yahoo Finance a parfois des erreurs de données où
        un prix est multiplié ou divisé par 10 pour un jour. Cela crée
        un rendement de ±900% → clairement un outlier (z > 100σ).

        On ne supprime PAS le jour (car d'autres actifs sont valides ce
        jour-là), on corrige juste le prix aberrant.
        """
        if len(df) < 10:
            return df, report

        # Calcul des rendements log pour la détection (sans modifier df)
        log_rets = np.log(df / df.shift(1)).dropna()

        for col in df.columns:
            series = log_rets[col].dropna()
            if len(series) < 10:
                continue

            mean = series.mean()
            std = series.std()

            if std == 0:
                continue

            # Masque des rendements statistiquement impossibles
            z_scores = np.abs((series - mean) / std)
            outlier_dates = series[z_scores > self.outlier_std_threshold].index

            if len(outlier_dates) == 0:
                continue

            n_fixed = 0
            for date in outlier_dates:
                loc = df.index.get_loc(date)
                # On ne corrige que si on a un prix avant et après
                if 0 < loc < len(df) - 1:
                    prev_price = df.iloc[loc - 1][col]
                    next_price = df.iloc[loc + 1][col]
                    if pd.notna(prev_price) and pd.notna(next_price):
                        df.at[date, col] = (prev_price + next_price) / 2
                        n_fixed += 1
                        logger.debug(
                            f"[Cleaner] Outlier corrigé — {col} @ {date.date()} "
                            f"(z={z_scores[date]:.1f}σ)"
                        )

            if n_fixed > 0:
                report.outliers_fixed[col] = n_fixed

        return df, report
