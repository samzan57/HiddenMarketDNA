# dashboard.py
"""
HiddenMarketDNA — Dashboard Streamlit
Lancer : streamlit run dashboard.py
"""

import os
import re
import glob
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# =============================================================================
# CONFIG
# =============================================================================

st.set_page_config(
    page_title="HiddenMarketDNA",
    page_icon="📈",
    layout="wide",
)

LOG_DIR      = Path(__file__).parent / "logs"
IBKR_PORT    = int(os.getenv("IBKR_PORT", "0")) or None  # voir .env.example
INITIAL_NAV  = 1_000_000.0   # valeur de départ du paper account


# =============================================================================
# PARSING DES LOGS
# =============================================================================

def parse_logs() -> pd.DataFrame:
    """Extrait les cycles d'exécution depuis les fichiers de log."""
    records = []
    for log_file in sorted(LOG_DIR.glob("live_trader_*.log")):
        content = log_file.read_text(encoding="utf-8", errors="ignore")

        # Chaque cycle commence à "Demarrage :"
        cycles = re.split(r"Demarrage\s*:", content)
        for cycle in cycles[1:]:
            date_match = re.search(r"(\d{2}/\d{2}/\d{4})", cycle)
            if not date_match:
                continue
            try:
                date = datetime.strptime(date_match.group(1), "%d/%m/%Y")
            except ValueError:
                continue

            nav_match   = re.search(r"Compte\s*:\s*\$([\d,]+)", cycle)
            nav         = float(nav_match.group(1).replace(",", "")) if nav_match else None

            regime_match = re.search(r"R.{0,2}gime\s*:\s*(\w+)", cycle)
            regime       = regime_match.group(1) if regime_match else "inconnu"

            scale_match = re.search(r"risk_scale=([\d.]+)", cycle)
            risk_scale  = float(scale_match.group(1)) if scale_match else None

            if "Risk check" in cycle and "ECHOUE" in cycle:
                result = "Bloqué"
            elif "Aucun ordre" in cycle:
                result = "Aucun ordre"
            elif "Tous les ordres" in cycle:
                result = "Ordres exécutés"
            elif "DRY-RUN" in cycle:
                result = "Dry-run"
            else:
                result = "Inconnu"

            orders = re.findall(r"(BUY |SELL)\s+(\d+)\s+(\w+)", cycle)
            records.append({
                "date": date, "nav": nav, "regime": regime,
                "risk_scale": risk_scale, "result": result, "orders": orders,
            })

    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records).drop_duplicates(subset=["date"]).sort_values("date")


# =============================================================================
# CONNEXION IBKR (optionnel)
# =============================================================================

@st.cache_data(ttl=60)
def fetch_ibkr_data():
    """Récupère NAV et positions depuis IBKR (si TWS ouvert)."""
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from src.execution.ibkr_client import IBKRClient
        with IBKRClient(port=IBKR_PORT) as ib:
            nav       = ib.get_net_liquidation()
            positions = ib.get_positions()
        return nav, positions
    except Exception as e:
        return None, None


# =============================================================================
# SIGNAL ACTUEL
# =============================================================================

@st.cache_data(ttl=300)
def fetch_signal():
    """Calcule le signal factoriel sur les données récentes."""
    try:
        import sys
        import numpy as np
        sys.path.insert(0, str(Path(__file__).parent))
        from src.data.pipeline import DataPipeline
        from src.pca_engine import PCAEngine
        from src.factors.momentum import compute_multi_horizon_momentum
        from src.factors.composite import CompositeSignal, FactorWeights
        from src.regimes.regime_manager import RegimeManager

        pipeline = DataPipeline(cache_dir="data/cache", cache_expiry_hours=1)
        _, returns = pipeline.run_returns(start="2010-01-01", end=None, universe="sector_etfs")

        in_sample = returns.iloc[-252:]
        pca       = PCAEngine(n_components=3)
        pca.fit(in_sample)
        pc2 = pca.get_eigen_portfolios()["PC2"]

        mom    = compute_multi_horizon_momentum(in_sample)
        signal = CompositeSignal(
            weights=FactorWeights(pca=0.60, momentum=0.40, quality=0.00),
            normalize=True,
        ).compute_with_breakdown(
            pca_loadings=pc2,
            momentum_scores=mom,
            quality_scores=pd.Series(0.0, index=in_sample.columns),
        )["weights"]

        regime_mgr    = RegimeManager(min_scale=0.30, update_freq=21)
        regime_result = regime_mgr.force_evaluate(in_sample)

        return signal * regime_result.risk_scale, regime_result
    except Exception as e:
        return None, None


# =============================================================================
# LAYOUT
# =============================================================================

st.title("📈 HiddenMarketDNA — Dashboard")
st.caption("PCA + Momentum + HMM/GARCH | Paper trading IBKR")

# --- Bouton refresh ---
if st.button("🔄 Rafraîchir"):
    st.cache_data.clear()
    st.rerun()

st.divider()

# ---- LIGNE 1 : métriques live ----
col1, col2, col3, col4 = st.columns(4)

nav_live, positions_live = fetch_ibkr_data()

with col1:
    if nav_live:
        pnl = nav_live - INITIAL_NAV
        st.metric("NAV (live)", f"${nav_live:,.0f}", delta=f"${pnl:+,.0f}")
    else:
        st.metric("NAV (live)", "TWS fermé", delta=None)

with col2:
    if positions_live is not None and not positions_live.empty:
        n_long  = (positions_live > 0).sum()
        n_short = (positions_live < 0).sum()
        st.metric("Positions", f"{len(positions_live)} actifs", delta=f"{n_long}L / {n_short}S")
    else:
        st.metric("Positions", "—")

signal, regime_result = fetch_signal()

with col3:
    if regime_result:
        color = {"bull": "🟢", "bear": "🔴", "stress": "🟡"}.get(regime_result.label, "⚪")
        st.metric("Régime", f"{color} {regime_result.label}")
    else:
        st.metric("Régime", "—")

with col4:
    if regime_result:
        st.metric("Risk Scale", f"{regime_result.risk_scale:.0%}")
    else:
        st.metric("Risk Scale", "—")

st.divider()

# ---- LIGNE 2 : signal + historique ----
col_signal, col_history = st.columns([1, 2])

with col_signal:
    st.subheader("Signal actuel")
    if signal is not None:
        df_sig = signal.sort_values().reset_index()
        df_sig.columns = ["ETF", "Poids"]
        df_sig["Sens"] = df_sig["Poids"].apply(lambda x: "LONG" if x > 0 else "SHORT")
        colors = df_sig["Poids"].apply(lambda x: "#2ecc71" if x > 0 else "#e74c3c")

        fig = go.Figure(go.Bar(
            x=df_sig["Poids"],
            y=df_sig["ETF"],
            orientation="h",
            marker_color=colors,
            text=df_sig["Poids"].apply(lambda x: f"{x:+.3f}"),
            textposition="auto",
        ))
        fig.update_layout(
            height=350,
            margin=dict(l=10, r=40, t=10, b=10),
            xaxis_title="Poids",
            plot_bgcolor="#0e1117",
            paper_bgcolor="#0e1117",
            font_color="white",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Signal indisponible (données non chargées)")

with col_history:
    st.subheader("Historique des exécutions")
    logs_df = parse_logs()

    if logs_df.empty:
        st.info("Aucun log disponible — le trader n'a pas encore tourné.")
    else:
        # Courbe NAV
        nav_df = logs_df[logs_df["nav"].notna()].copy()
        if not nav_df.empty:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=nav_df["date"],
                y=nav_df["nav"],
                mode="lines+markers",
                line=dict(color="#3498db", width=2),
                marker=dict(size=6),
                name="NAV",
            ))
            fig2.add_hline(
                y=INITIAL_NAV,
                line_dash="dash",
                line_color="gray",
                annotation_text="Capital initial",
            )
            fig2.update_layout(
                height=280,
                margin=dict(l=10, r=10, t=10, b=10),
                yaxis_title="NAV ($)",
                plot_bgcolor="#0e1117",
                paper_bgcolor="#0e1117",
                font_color="white",
            )
            st.plotly_chart(fig2, use_container_width=True)

        # Tableau des cycles
        display_df = logs_df[["date", "nav", "regime", "risk_scale", "result"]].copy()
        display_df["date"]       = display_df["date"].dt.strftime("%Y-%m-%d")
        display_df["nav"]        = display_df["nav"].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "—")
        display_df["risk_scale"] = display_df["risk_scale"].apply(lambda x: f"{x:.0%}" if pd.notna(x) else "—")
        display_df.columns       = ["Date", "NAV", "Régime", "Risk Scale", "Résultat"]
        st.dataframe(display_df, use_container_width=True, hide_index=True)

st.divider()

# ---- LIGNE 3 : positions live ----
st.subheader("Positions actuelles (IBKR live)")
if positions_live is not None and not positions_live.empty:
    pos_df = positions_live.reset_index()
    pos_df.columns = ["ETF", "Quantité"]
    pos_df["Sens"] = pos_df["Quantité"].apply(lambda x: "🟢 LONG" if x > 0 else "🔴 SHORT")
    pos_df = pos_df.sort_values("Quantité")
    st.dataframe(pos_df, use_container_width=True, hide_index=True)
else:
    st.info("TWS fermé ou aucune position — ouvre TWS pour voir les positions en direct.")

st.divider()
st.caption("Mis à jour automatiquement · Signal recalculé toutes les 5 min · NAV toutes les 60s")
