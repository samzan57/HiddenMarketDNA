# src/regimes/garch_vol.py
"""
Modèle GARCH(1,1) pour la prévision de la volatilité conditionnelle.

POURQUOI GARCH ?
  Les rendements financiers présentent des "clusters de volatilité" :
  les périodes agitées tendent à rester agitées, et les périodes calmes
  aussi. Ce phénomène est appelé hétéroscédasticité conditionnelle.

  Le modèle naïf (vol rolling Sprint 1/2) : σ_t = std(r_{t-20:t})
    → Regarde en arrière, retardé.

  GARCH(1,1) : σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}
    → Modélise σ_t comme fonction de la vol passée ET du choc récent.
    → Prédit σ_{t+1} AVANT qu'on l'observe.

  Paramètres économiques :
    ω (omega) : niveau de vol de long terme (constant)
    α (alpha) : réactivité aux nouveaux chocs (si α élevé → vol monte vite)
    β (beta)  : persistance de la volatilité (si β élevé → vol redescend lentement)
    α + β < 1 → stationnarité (garantit que la vol revient à la moyenne)

  Exemple : pour le S&P 500, typiquement α≈0.09, β≈0.90 → haute persistance.

USAGE DANS CE SYSTÈME :
  Le ratio forecast_vol / long_run_vol mesure si la vol actuelle est
  au-dessus ou en dessous de sa moyenne historique :
    - Ratio > 1.5 → stress (vol 50% au-dessus de la normale)
    - Ratio ≈ 1.0 → régime normal
    - Ratio < 0.7 → marché très calme (opportunité ?)
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Import optionnel : fallback si arch non installé
try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False
    logger.warning(
        "[GARCH] arch non installé. Fallback sur vol rolling. "
        "Installer avec : pip install arch"
    )


@dataclass
class GARCHResult:
    """Résultat complet d'une estimation GARCH."""
    forecast_vol:   float   # volatilité annualisée prévue pour t+1
    long_run_vol:   float   # volatilité inconditionnelle (long terme)
    vol_ratio:      float   # forecast_vol / long_run_vol
    alpha:          float   # persistance chocs courts
    beta:           float   # persistance vol longue
    converged:      bool    # True si GARCH a convergé normalement


class GARCHVolatility:
    """
    Estime GARCH(1,1) sur le portefeuille équipondéré et prédit σ_{t+1}.

    Paramètres
    ----------
    min_obs : int
        Nombre minimum d'observations pour tenter l'estimation GARCH.
        En dessous, on retourne directement la vol rolling.
    """

    def __init__(self, min_obs: int = 100):
        self.min_obs = min_obs

    def fit_forecast(self, returns: pd.DataFrame) -> GARCHResult:
        """
        Fit GARCH(1,1) sur les rendements et prédit la vol du jour suivant.

        On utilise le portefeuille équipondéré comme série de référence
        (représente le "marché" de notre univers).

        Paramètres
        ----------
        returns : pd.DataFrame
            Fenêtre de rendements in-sample (ex: 252 jours × 9 actifs).

        Retourne
        --------
        GARCHResult avec le ratio vol forecast / vol long-terme.
        """
        # Portefeuille équipondéré = proxy du "marché" de notre univers
        ew = returns.mean(axis=1)

        if not ARCH_AVAILABLE or len(ew) < self.min_obs:
            return self._fallback(ew)

        try:
            # On multiplie par 100 pour passer en % → meilleure stabilité numérique
            # arch l'exige souvent pour que les paramètres ω ne soient pas ≈ 0
            ew_pct = ew * 100

            model = arch_model(
                ew_pct,
                vol="Garch",
                p=1, q=1,
                dist="normal",
                rescale=False,
            )

            result = model.fit(
                disp="off",          # supprime l'output de convergence
                show_warning=False,
                options={"maxiter": 200},
            )

            # --- Prévision 1 pas en avant ---
            forecast = result.forecast(horizon=1, reindex=False)
            # variance prévue en (%)² → convertir en vol annualisée décimale
            forecast_var_pct2 = float(forecast.variance.iloc[-1, 0])
            forecast_vol = np.sqrt(forecast_var_pct2) / 100 * np.sqrt(252)

            # --- Vol inconditionnelle (long terme) ---
            # σ²_∞ = ω / (1 - α - β)
            omega = float(result.params.get("omega", 0))
            alpha = float(result.params.get("alpha[1]", 0))
            beta  = float(result.params.get("beta[1]", 0))

            denom = 1 - alpha - beta
            if denom > 0.01:
                long_run_var_pct2 = omega / denom
                long_run_vol = np.sqrt(long_run_var_pct2) / 100 * np.sqrt(252)
            else:
                # α + β ≈ 1 → IGARCH (pas stationnaire) → fallback sur std empirique
                long_run_vol = float(ew.std() * np.sqrt(252))

            long_run_vol = max(long_run_vol, 1e-6)  # éviter division par zéro
            vol_ratio = forecast_vol / long_run_vol

            return GARCHResult(
                forecast_vol=forecast_vol,
                long_run_vol=long_run_vol,
                vol_ratio=float(vol_ratio),
                alpha=alpha,
                beta=beta,
                converged=True,
            )

        except Exception as exc:
            logger.debug(f"[GARCH] Estimation échouée : {exc}. Fallback activé.")
            return self._fallback(ew)

    def _fallback(self, ew: pd.Series) -> GARCHResult:
        """
        Fallback robuste si GARCH échoue ou arch non disponible.
        Ratio = vol 21 jours / vol 252 jours.
        """
        vol_recent   = float(ew.iloc[-21:].std() * np.sqrt(252)) if len(ew) >= 21 else 0.15
        vol_long_run = float(ew.std() * np.sqrt(252))
        vol_long_run = max(vol_long_run, 1e-6)

        return GARCHResult(
            forecast_vol=vol_recent,
            long_run_vol=vol_long_run,
            vol_ratio=vol_recent / vol_long_run,
            alpha=0.0,
            beta=0.0,
            converged=False,
        )
