# src/regimes/regime_manager.py
"""
Orchestrateur de la détection de régimes — combine HMM + GARCH.

LOGIQUE DE COMBINAISON :

  stress_score = w_hmm × P(HMM=stress) + w_garch × f(vol_ratio)

  Où f(vol_ratio) normalise le ratio GARCH en [0,1] :
    f(r) = clip((r - 0.5) / 1.5, 0, 1)
    → vol_ratio = 0.5 → 0.0 (très calme)
    → vol_ratio = 1.0 → 0.33 (normal)
    → vol_ratio = 2.0 → 1.0 (stress fort)

  risk_scale = 1 - stress_score × (1 - min_scale)
    → stress_score = 0   → risk_scale = 1.0   (pleine exposition)
    → stress_score = 0.5 → risk_scale = 0.65  (réduction modérée)
    → stress_score = 1.0 → risk_scale = 0.3   (réduction maximale)

AVANTAGE VS. BINAIRE (Sprint 1/2) :
  Sprint 1/2 : si vol > seuil → ×0.5, sinon ×1.0 (tout ou rien)
  Sprint 3   : réduction CONTINUE proportionnelle au niveau de stress
               → Moins de frais de transaction (on ne passe pas de 100% à 50% d'un coup)
               → Plus stable, moins de faux signaux

FRÉQUENCE DE MISE À JOUR :
  Refitter HMM + GARCH à chaque step serait trop lent (~5-8 min total).
  Par défaut : on refit tous les `update_freq` jours (= 21 = mensuel).
  Entre deux refits, on utilise le dernier régime calculé.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.regimes.hmm_detector import HMMRegimeDetector
from src.regimes.garch_vol import GARCHVolatility

logger = logging.getLogger(__name__)


@dataclass
class RegimeResult:
    """
    Résultat complet de l'évaluation du régime pour une date donnée.
    Stocké à chaque pas du backtest pour le monitoring et la visualisation.
    """
    stress_score:       float   # score composite [0,1] : 0=calme, 1=stress
    risk_scale:         float   # facteur multiplicatif des poids [min_scale, 1.0]
    label:              str     # "calm" | "transition" | "stress"
    hmm_stress_prob:    float   # P(HMM=stress) ∈ [0,1]
    garch_vol_ratio:    float   # vol forecast / vol long terme


class RegimeManager:
    """
    Combine HMM et GARCH pour une détection de régime continue et robuste.

    Paramètres
    ----------
    min_scale : float
        Exposition minimale du portefeuille en régime de stress total.
        0.3 = on garde au moins 30% de l'exposition même en crise.
    hmm_weight : float
        Poids du signal HMM dans le score composite (défaut 60%).
    garch_weight : float
        Poids du signal GARCH dans le score composite (défaut 40%).
    update_freq : int
        Fréquence de refit des modèles en jours (21 = mensuel).
        Compromise entre précision et vitesse d'exécution.
    stress_threshold : float
        Seuil du stress_score pour passer en label "stress" (défaut 0.60).
    transition_threshold : float
        Seuil pour passer en label "transition" (défaut 0.35).
    """

    def __init__(
        self,
        min_scale:             float = 0.30,
        hmm_weight:            float = 0.60,
        garch_weight:          float = 0.40,
        update_freq:           int   = 21,
        stress_threshold:      float = 0.60,
        transition_threshold:  float = 0.35,
    ):
        if abs(hmm_weight + garch_weight - 1.0) > 1e-6:
            raise ValueError("hmm_weight + garch_weight doit = 1.0")

        self.min_scale            = min_scale
        self.hmm_weight           = hmm_weight
        self.garch_weight         = garch_weight
        self.update_freq          = update_freq
        self.stress_threshold     = stress_threshold
        self.transition_threshold = transition_threshold

        self._hmm   = HMMRegimeDetector(n_states=2)
        self._garch = GARCHVolatility()

        # Cache pour éviter de refitter à chaque step
        self._last_result:       Optional[RegimeResult] = None
        self._steps_since_refit: int = self.update_freq  # force refit au premier appel

    # ------------------------------------------------------------------
    # Interface principale
    # ------------------------------------------------------------------

    def evaluate(self, returns: pd.DataFrame) -> RegimeResult:
        """
        Évalue le régime actuel sur la fenêtre in-sample.

        Refit les modèles seulement tous les `update_freq` jours pour
        limiter le temps de calcul.

        Paramètres
        ----------
        returns : pd.DataFrame
            Rendements in-sample (typiquement 252 jours).

        Retourne
        --------
        RegimeResult avec stress_score, risk_scale, et label.
        """
        # Refit si nécessaire (premier appel ou fréquence atteinte)
        if self._steps_since_refit >= self.update_freq:
            self._refit(returns)
            self._steps_since_refit = 0
        else:
            self._steps_since_refit += 1

        # Utilise les modèles déjà fittés pour obtenir les signaux actuels
        result = self._compute_result(returns)
        self._last_result = result
        return result

    def force_evaluate(self, returns: pd.DataFrame) -> RegimeResult:
        """Évalue le régime en refittant inconditionnellement les modèles."""
        self._steps_since_refit = self.update_freq  # déclenche le refit dans evaluate
        return self.evaluate(returns)

    # ------------------------------------------------------------------
    # Calcul interne
    # ------------------------------------------------------------------

    def _refit(self, returns: pd.DataFrame) -> None:
        """Ré-entraîne HMM et GARCH sur la fenêtre courante."""
        try:
            self._hmm.fit(returns)
        except Exception as e:
            logger.debug(f"[RegimeManager] HMM refit échoué : {e}")

    def _compute_result(self, returns: pd.DataFrame) -> RegimeResult:
        """
        Calcule le RegimeResult à partir des modèles déjà fittés.
        """
        # --- Signal HMM ---
        hmm_prob = self._hmm.predict_stress_probability(returns)

        # --- Signal GARCH ---
        garch_res = self._garch.fit_forecast(returns)
        vol_ratio = garch_res.vol_ratio

        # Normaliser le vol_ratio en [0,1]
        # Mapping : 0.5 → 0.0 (calme), 1.0 → 0.33, 2.0 → 1.0 (stress)
        garch_signal = float(np.clip((vol_ratio - 0.5) / 1.5, 0.0, 1.0))

        # --- Score de stress composite ---
        stress_score = self.hmm_weight * hmm_prob + self.garch_weight * garch_signal

        # --- Risk scale continu ---
        # Varie entre min_scale (stress total) et 1.0 (calme total)
        risk_scale = 1.0 - stress_score * (1.0 - self.min_scale)
        risk_scale = float(np.clip(risk_scale, self.min_scale, 1.0))

        # --- Label qualitatif ---
        if stress_score >= self.stress_threshold:
            label = "stress"
        elif stress_score >= self.transition_threshold:
            label = "transition"
        else:
            label = "calm"

        return RegimeResult(
            stress_score=float(stress_score),
            risk_scale=risk_scale,
            label=label,
            hmm_stress_prob=float(hmm_prob),
            garch_vol_ratio=float(vol_ratio),
        )

    def reset(self) -> None:
        """Réinitialise le cache — utile entre deux backtests."""
        self._last_result = None
        self._steps_since_refit = self.update_freq
        self._hmm = HMMRegimeDetector(n_states=2)
        self._garch = GARCHVolatility()
