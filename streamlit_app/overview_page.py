"""Page « Vue d'ensemble » — synthèse façon Power BI (KPI + analyses clés)."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

import data as D

_PLOT = dict(template="plotly_white", margin=dict(l=10, r=10, t=40, b=10))


def _kpi_card(label, value, sub=""):
    st.markdown(
        f"""
        <div class="card" style="text-align:center; padding:16px 10px;">
            <div style="font-size:0.78rem; color:var(--faint); text-transform:uppercase; letter-spacing:0.3px; min-height:2.4em; display:flex; align-items:center; justify-content:center; line-height:1.2;">{label}</div>
            <div style="font-size:1.4rem; font-weight:700; color:var(--accent); margin-top:4px; white-space:nowrap;">{value}</div>
            <div style="font-size:0.76rem; color:var(--faint); margin-top:2px;">{sub}&nbsp;</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(ttl=60)
def _load(start, end):
    return D.load_observations(start, end)


def _period_filter(obs_all):
    """Fenêtre glissante relative à la dernière donnée disponible (7 j par défaut)."""
    max_ts = obs_all.index.max()
    with st.sidebar:
        st.markdown("### Période")
        choice = st.radio("Fenêtre d'affichage", ["7 derniers jours", "30 derniers jours"],
                          index=0, key="ov_period")
    days = 7 if choice.startswith("7") else 30
    start = max_ts - pd.Timedelta(days=days)
    end = max_ts + pd.Timedelta(hours=1)
    return start, end


def overview():
    st.markdown("<h1 class='main-title'>Vue d'ensemble</h1>", unsafe_allow_html=True)

    obs_all = _load(None, None)
    if obs_all.empty:
        st.info("Aucune donnée en base. Lancez l'ingestion (DAG ingest_hourly).")
        return

    start, end = _period_filter(obs_all)
    obs = _load(start, end)
    if obs.empty:
        st.info("Aucune donnée sur la période sélectionnée.")
        return

    prod_cols = [c for c in D.PRODUCTION_SOURCES if c in obs.columns]
    has_conso = "consommation_totale" in obs.columns
    prod_total = obs[prod_cols].sum(axis=1) if prod_cols else None

    # ------------------------------------------------------------------ KPI
    avg_conso = obs["consommation_totale"].mean() if has_conso else np.nan
    peak_conso = obs["consommation_totale"].max() if has_conso else np.nan
    peak_hour = obs["consommation_totale"].idxmax() if has_conso else None
    avg_temp = obs["temp"].mean() if "temp" in obs.columns else np.nan
    avg_prod = prod_total.mean() if prod_total is not None else np.nan

    renew_share = np.nan
    if prod_cols:
        renew_cols = [c for c in prod_cols if c != "NUCLEAR"]
        denom = obs[prod_cols].sum().sum()
        renew_share = 100 * obs[renew_cols].sum().sum() / denom if denom else np.nan

    deficit_gw = np.nan
    pct_deficit = np.nan
    if prod_total is not None and has_conso:
        deficit = prod_total - obs["consommation_totale"]
        deficit_gw = deficit.mean() / 1000.0
        pct_deficit = 100 * (deficit < 0).mean()

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        _kpi_card("Consommation moy.", f"{avg_conso:,.0f} MW".replace(",", " ") if pd.notna(avg_conso) else "—")
    with c2:
        _kpi_card("Pic de conso.", f"{peak_conso:,.0f} MW".replace(",", " ") if pd.notna(peak_conso) else "—",
                  sub=peak_hour.strftime("%d/%m %Hh") if peak_hour is not None else "")
    with c3:
        _kpi_card("Production suivie moy.", f"{avg_prod:,.0f} MW".replace(",", " ") if pd.notna(avg_prod) else "—")
    with c4:
        _kpi_card("Part renouvelable", f"{renew_share:.0f} %" if pd.notna(renew_share) else "—",
                  sub="hors nucléaire")
    with c5:
        _kpi_card("Écart moyen (partiel)", f"{deficit_gw:+.2f} GW" if pd.notna(deficit_gw) else "—",
                  sub="4 sources − conso")
    with c6:
        _kpi_card("Heures sous consommation", f"{pct_deficit:.0f} %" if pd.notna(pct_deficit) else "—")

    st.markdown("<br>", unsafe_allow_html=True)

    # ----------------------------------------------- Conso vs prod + mix
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Consommation vs production suivie")
        # Réindexation horaire continue : les heures manquantes deviennent NaN,
        # ce qui casse la ligne au lieu de relier les blocs par une droite fictive.
        obs_line = obs.reindex(pd.date_range(obs.index.min(), obs.index.max(), freq="h"))
        prod_line = obs_line[prod_cols].sum(axis=1, min_count=1) if prod_cols else None
        fig = go.Figure()
        if has_conso:
            fig.add_trace(go.Scatter(x=obs_line.index, y=obs_line["consommation_totale"], name="Consommation",
                                     line=dict(color="#e74c3c", width=2), connectgaps=False))
        if prod_line is not None:
            fig.add_trace(go.Scatter(x=obs_line.index, y=prod_line, name="Production suivie (4 sources)",
                                     line=dict(color="#27ae60", width=2), fill="tozeroy",
                                     fillcolor="rgba(39,174,96,0.08)", connectgaps=False))
        fig.update_layout(height=360, yaxis_title="MW",
                          legend=dict(orientation="h", y=1.02), **_PLOT)
        st.plotly_chart(fig, width="stretch")
    with right:
        st.subheader("Mix de production")
        if prod_cols:
            means = obs[prod_cols].mean()
            figp = go.Figure(go.Pie(labels=means.index, values=means.values, hole=0.55,
                                    marker=dict(colors=[D.ENERGY_COLORS.get(s, "#888") for s in means.index]),
                                    textinfo="percent"))
            figp.update_layout(height=360, legend=dict(orientation="h", y=-0.1), **_PLOT)
            st.plotly_chart(figp, width="stretch")

    st.caption(
        "Périmètre de production suivi : solaire, biomasse, éolien terrestre et "
        "nucléaire. L'hydraulique, le thermique fossile, les échanges et les "
        "pertes réseau ne sont pas disponibles ; l'écart affiché n'est donc pas "
        "le solde électrique national."
    )

    # ------------------------------------------------------ Écart journalier
    if prod_total is not None and has_conso:
        st.subheader("Écart production suivie − consommation (journalier)")
        daily = (prod_total - obs["consommation_totale"]).resample("D").mean() / 1000.0
        colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in daily]
        figd = go.Figure(go.Bar(x=daily.index, y=daily.values, marker_color=colors,
                                marker_line=dict(color="rgba(0,0,0,0.15)", width=1)))
        figd.add_hline(y=0, line_dash="dash", line_color="gray",
                       annotation_text="Équilibre", annotation_position="bottom right")
        figd.update_layout(height=300, yaxis_title="Écart partiel (GW)", **_PLOT)
        st.plotly_chart(figd, width="stretch")

    # ------------------------------------------- Production mensuelle empilée
    if prod_cols:
        st.subheader("Production mensuelle par source")
        monthly = obs[prod_cols].copy()
        monthly["mois"] = monthly.index.to_period("M").astype(str)
        agg = monthly.groupby("mois")[prod_cols].sum().reset_index()
        figm = go.Figure()
        for src in prod_cols:
            figm.add_trace(go.Bar(x=agg["mois"], y=agg[src], name=src,
                                  marker_color=D.ENERGY_COLORS.get(src, "#888")))
        figm.update_layout(barmode="stack", height=330, yaxis_title="Production (MWh)",
                           xaxis_title="Mois", legend=dict(orientation="h", y=1.02),
                           xaxis=dict(tickangle=-45), **_PLOT)
        st.plotly_chart(figm, width="stretch")

    # ---------------------------------------------------------- Distributions
    st.subheader("Distributions des variables principales")
    main_vars = [v for v in (["consommation_totale"] + prod_cols + ["temp"]) if v in obs.columns]
    if main_vars:
        n = len(main_vars)
        cols = min(3, n)
        rows = (n + cols - 1) // cols
        figh = make_subplots(rows=rows, cols=cols, subplot_titles=[v.replace("_", " ").title() for v in main_vars],
                             vertical_spacing=0.18, horizontal_spacing=0.08)
        palette = ["#e74c3c", "#f1c40f", "#27ae60", "#3498db", "#9b59b6", "#f39c12"]
        for i, var in enumerate(main_vars):
            r, c = i // cols + 1, i % cols + 1
            figh.add_trace(go.Histogram(x=obs[var], nbinsx=30, marker_color=palette[i % len(palette)],
                                        opacity=0.75, showlegend=False), row=r, col=c)
        figh.update_layout(height=260 * rows, template="plotly_white",
                           margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(figh, width="stretch")
