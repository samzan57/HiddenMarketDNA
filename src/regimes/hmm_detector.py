# src/regimes/hmm_detector.py
"""
Détection de régimes de marché par Hidden Markov Model (HMM).

POURQUOI UN HMM ?
  Le marché financier alterne entre des états "cachés" (latents) :
    - État 0 : Régime calme   → faible vol, tendance haussière lente
    - État 1 : Régime stress  → forte vol, mouvements brusques

  Ces états ne sont pas directement observables (d'où "hidden").
  Le HMM infère leur probabilité à partir de ce qu'on observe :
  les rendements, la volatilité, la dispersion cross-sectorielle.

AVANTAGE VS. SEUIL SIMPLE (Sprint 1/2) :
  - Le seuil PC1 est RÉACTIF  : détecte le stress après qu'il a commencé
  - Le HMM est PRÉDICTIF      : détecte la transition calme→stress
    dès que les rendements commencent à se comporter différemment,
    AVANT que la volatilité explose complètement.

ALGORITHME :
  - _NumpyGaussianHMM : implémentation NumPy pure, K états gaussiens diagonaux
  - 3 features : rendement moyen, vol réalisée, dispersion cross-sectorielle
  - Baum-Welch (EM) pour l'apprentissage des paramètres
  - Forward algorithm pour les posteriors P(état | observations)
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class _NumpyGaussianHMM:
    """
    HMM gaussien diagonal à K états, implémenté en NumPy pur.

    Pas de dépendance externe — fonctionne sur toutes les versions Python.

    Algorithmes :
      - Baum-Welch (EM) pour l'apprentissage des paramètres
      - Forward algorithm pour les posteriors (probabilités d'état)

    Paramètres
    ----------
    n_states : int     Nombre d'états cachés (typiquement 2)
    n_iter   : int     Itérations EM max
    tol      : float   Critère de convergence (log-vraisemblance)
    """

    def __init__(self, n_states: int = 2, n_iter: int = 50, tol: float = 1e-4, random_state: int = 42):
        self.n_states     = n_states
        self.n_iter       = n_iter
        self.tol          = tol
        self.random_state = random_state
        # Paramètres appris — initialisés dans fit()
        self.pi:    np.ndarray = None   # distribution initiale (K,)
        self.A:     np.ndarray = None   # matrice de transition (K, K)
        self.means: np.ndarray = None   # moyennes (K, D)
        self.vars:  np.ndarray = None   # variances (K, D) — diagonale

    # ------------------------------------------------------------------
    # Fit (Baum-Welch)
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> "_NumpyGaussianHMM":
        """
        X : (T, D) — T observations, D features.
        """
        T, D = X.shape
        K    = self.n_states
        rng  = np.random.default_rng(self.random_state)

        # --- Initialisation par k-means simple (percentiles) ---
        self.pi    = np.ones(K) / K
        self.A     = np.full((K, K), 1 / K)
        # On initialise les moyennes sur des percentiles de la 1ère feature
        pct        = np.linspace(20, 80, K)
        self.means = np.stack([
            np.percentile(X, p, axis=0) for p in pct
        ])
        # Variance globale × bruit aléatoire pour casser la symétrie
        self.vars  = np.tile(X.var(axis=0) + 1e-6, (K, 1)) * (0.5 + rng.random((K, D)))

        prev_ll = -np.inf

        for _ in range(self.n_iter):
            # E-step : forward-backward
            log_emit = self._log_emission(X)      # (T, K)
            alpha    = self._forward(log_emit)     # (T, K) log-scaled
            beta     = self._backward(log_emit)    # (T, K) log-scaled
            gamma    = self._gamma(alpha, beta)    # (T, K) posteriors
            xi       = self._xi(alpha, beta, log_emit)  # (T-1, K, K)

            ll = self._log_likelihood(alpha)
            if abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll

            # M-step : mise à jour des paramètres
            self.pi    = gamma[0] / gamma[0].sum()
            self.A     = xi.sum(0) / xi.sum(0).sum(1, keepdims=True)
            g_sum      = gamma.sum(0)              # (K,)
            self.means = (gamma.T @ X) / g_sum[:, None]
            diff       = X[None, :, :] - self.means[:, None, :]  # (K, T, D)
            self.vars  = ((gamma.T[:, :, None] * diff**2).sum(1) / g_sum[:, None]) + 1e-6

        return self

    # ------------------------------------------------------------------
    # Prédiction des posteriors
    # ------------------------------------------------------------------

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Retourne les posteriors (T, K) = P(état_k | observations)."""
        log_emit = self._log_emission(X)
        alpha    = self._forward(log_emit)
        beta     = self._backward(log_emit)
        return self._gamma(alpha, beta)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _log_emission(self, X: np.ndarray) -> np.ndarray:
        """Log-probabilité d'émission P(x_t | état_k) — gaussienne diagonale."""
        T, D = X.shape
        K    = self.n_states
        out  = np.zeros((T, K))
        for k in range(K):
            diff     = X - self.means[k]                          # (T, D)
            log_norm = -0.5 * (np.log(2 * np.pi * self.vars[k])  # normalisation
                               + diff**2 / self.vars[k]).sum(1)
            out[:, k] = log_norm
        return out  # (T, K)

    def _forward(self, log_emit: np.ndarray) -> np.ndarray:
        """Forward algorithm en log-espace (évite l'underflow numérique)."""
        T, K  = log_emit.shape
        alpha = np.full((T, K), -np.inf)
        alpha[0] = np.log(self.pi + 1e-300) + log_emit[0]
        log_A    = np.log(self.A + 1e-300)

        # alpha[t-1, :, None] + log_A → (K, K), logsumexp sur axis=0 → (K,)
        for t in range(1, T):
            alpha[t] = self._logsumexp(alpha[t-1, :, None] + log_A, axis=0) + log_emit[t]
        return alpha

    def _backward(self, log_emit: np.ndarray) -> np.ndarray:
        """Backward algorithm en log-espace."""
        T, K = log_emit.shape
        beta  = np.zeros((T, K))   # beta[T-1] = log(1) = 0
        log_A = np.log(self.A + 1e-300)

        # log_A + log_emit[t+1] + beta[t+1] → (K, K), logsumexp sur axis=1 → (K,)
        for t in range(T - 2, -1, -1):
            beta[t] = self._logsumexp(log_A + log_emit[t+1] + beta[t+1], axis=1)
        return beta

    def _gamma(self, alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
        """Posteriors P(z_t=k | x_{1:T}) normalisés."""
        log_g = alpha + beta
        log_g -= self._logsumexp(log_g, axis=1, keepdims=True)
        return np.exp(log_g)

    def _xi(self, alpha, beta, log_emit) -> np.ndarray:
        """Joint posteriors P(z_t=j, z_{t+1}=k | x) — (T-1, K, K)."""
        T, K  = log_emit.shape
        log_A = np.log(self.A + 1e-300)
        # Fully vectorised: no Python loops over t, j, k
        xi = (alpha[:-1, :, None]       # (T-1, K, 1)
              + log_A[None, :, :]       # (1,   K, K)
              + log_emit[1:, None, :]   # (T-1, 1, K)
              + beta[1:, None, :])      # (T-1, 1, K)
        log_norm = self._logsumexp(xi.reshape(T - 1, -1), axis=1)  # (T-1,)
        xi -= log_norm[:, None, None]
        return np.exp(xi)

    def _log_likelihood(self, alpha: np.ndarray) -> float:
        return float(self._logsumexp(alpha[-1]))

    @staticmethod
    def _logsumexp(a, axis=None, keepdims=False):
        """logsumexp numériquement stable."""
        a_max = np.max(a, axis=axis, keepdims=True)
        out   = np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=keepdims))
        if not keepdims:
            a_max = np.squeeze(a_max, axis=axis) if axis is not None else a_max.ravel()[0]
        return out + a_max


class HMMRegimeDetector:
    """
    Détecte le régime de marché (calme / stress) via un HMM gaussien à 2 états.

    Paramètres
    ----------
    n_states : int
        Nombre d'états cachés. 2 = calme/stress. 3 = calme/transition/crise.
    n_iter : int
        Itérations max de l'algorithme EM (Baum-Welch) pour l'apprentissage.
    random_state : int
        Seed pour la reproductibilité.
    """

    def __init__(
        self,
        n_states: int = 2,
        n_iter: int = 50,
        random_state: int = 42,
    ):
        self.n_states     = n_states
        self.n_iter       = n_iter
        self.random_state = random_state
        self._model:       _NumpyGaussianHMM = None
        self._stress_state: int = None   # index de l'état "stress" après fit

    # ------------------------------------------------------------------
    # Construction des features
    # ------------------------------------------------------------------

    def _build_features(self, returns: pd.DataFrame) -> np.ndarray:
        """
        Construit la matrice de features pour le HMM.

        3 features choisies pour leur capacité discriminante :
          1. Rendement moyen glissant (20j)  : négatif en période de stress
          2. Volatilité réalisée (20j, ann.)  : élevée en stress
          3. Dispersion cross-sectorielle     : élevée en stress (rotation forte)

        Toutes les features sont normalisées (z-score) pour éviter
        que la volatilité (grande échelle) domine les autres.
        """
        # Portefeuille équipondéré sur l'univers
        ew_returns = returns.mean(axis=1)

        # Feature 1 : rendement moyen glissant 20 jours
        rolling_ret = ew_returns.rolling(20, min_periods=5).mean().fillna(0.0)

        # Feature 2 : volatilité réalisée annualisée sur 20 jours
        rolling_vol = (
            ew_returns.rolling(20, min_periods=5)
            .std()
            .fillna(ew_returns.std())
            * np.sqrt(252)
        )

        # Feature 3 : dispersion cross-sectorielle (écart-type des rendements
        # entre actifs, par jour) — capte la rotation sectorielle extrême
        cross_disp = returns.std(axis=1).fillna(returns.std(axis=1).median())

        features = np.column_stack([
            rolling_ret.values,
            rolling_vol.values,
            cross_disp.values,
        ])

        # Z-score global pour normaliser les échelles
        features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)

        return features

    # ------------------------------------------------------------------
    # Fit + prédiction
    # ------------------------------------------------------------------

    def fit(self, returns: pd.DataFrame) -> "HMMRegimeDetector":
        """
        Entraîne le HMM sur la fenêtre de rendements in-sample.

        L'état "stress" est identifié automatiquement comme l'état dont
        la volatilité réalisée moyenne est la plus élevée.
        """
        features = self._build_features(returns)

        model = _NumpyGaussianHMM(n_states=self.n_states, n_iter=self.n_iter, random_state=self.random_state)

        try:
            model.fit(features)
            self._model = model
            # État stress = celui dont la vol réalisée moyenne (feature 1) est la plus haute
            self._stress_state = int(np.argmax(model.means[:, 1]))
        except Exception as exc:
            logger.warning(f"[HMM] Échec de l'entraînement : {exc}. Fallback activé.")
            self._model = None

        return self

    def predict_stress_probability(self, returns: pd.DataFrame) -> float:
        """
        Retourne P(état=stress) au dernier pas de temps de la fenêtre.

        C'est le signal clé : plus cette probabilité est élevée,
        plus le régime ressemble à du stress → on réduit l'exposition.

        Retourne
        --------
        float in [0, 1]
            0.0 = certainement calme
            1.0 = certainement en stress
        """
        if self._model is None:
            return self._fallback_stress_prob(returns)

        features = self._build_features(returns)

        try:
            # posteriors : (T, n_states) → P(état_k | observations) à chaque t
            posteriors = self._model.predict_proba(features)
            return float(posteriors[-1, self._stress_state])
        except Exception:
            return self._fallback_stress_prob(returns)

    def _fallback_stress_prob(self, returns: pd.DataFrame) -> float:
        """Fallback si HMM non convergé : ratio vol récente / vol longue."""
        ew = returns.mean(axis=1)
        vol_recent = ew.iloc[-21:].std() * np.sqrt(252)
        vol_long   = ew.std() * np.sqrt(252)
        if vol_long < 1e-8:
            return 0.5
        ratio = vol_recent / vol_long
        # Mapper ratio → probabilité : ratio=1 → 0.5, ratio=2 → 1.0, ratio=0.5 → 0.0
        return float(np.clip((ratio - 0.5) / 1.5, 0.0, 1.0))
