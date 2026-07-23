"""Page « Analyse Production » — mix, profils par source et corrélations météo."""

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
def _load():
    return D.load_observations()


def _window_filter(obs_all):
    max_ts = obs_all.index.max()
    with st.sidebar:
        st.markdown("### Période")
        choice = st.radio("Fenêtre d'affichage", ["30 derniers jours", "7 derniers jours"],
                          index=0, key="prod_period")
    days = 30 if choice.startswith("30") else 7
    return obs_all[obs_all.index >= max_ts - pd.Timedelta(days=days)]


def production():
    st.markdown("<h1 class='main-title'>Analyse de la production suivie</h1>", unsafe_allow_html=True)

    obs_all = _load()
    prod_cols_all = [c for c in D.PRODUCTION_SOURCES if c in obs_all.columns]
    if obs_all.empty or not prod_cols_all:
        st.info("Aucune donnée de production en base.")
        return

    obs = _window_filter(obs_all).sort_index()
    prod_cols = [c for c in D.PRODUCTION_SOURCES if c in obs.columns]
    if obs.empty:
        st.info("Aucune donnée sur la fenêtre sélectionnée.")
        return

    prod_total = obs[prod_cols].sum(axis=1)
    totals = obs[prod_cols].sum()
    grand_total = totals.sum()

    # -------------------------------------------------------------- KPI
    nuclear_share = 100 * totals.get("NUCLEAR", 0) / grand_total if grand_total else np.nan
    renew_share = 100 * totals[[c for c in prod_cols if c != "NUCLEAR"]].sum() / grand_total if grand_total else np.nan

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        _kpi_card("Production suivie moy.", f"{prod_total.mean():,.0f} MW".replace(",", " "))
    with k2:
        _kpi_card("Pic de production", f"{prod_total.max():,.0f} MW".replace(",", " "),
                  sub=prod_total.idxmax().strftime("%d/%m %Hh"))
    with k3:
        _kpi_card("Part nucléaire", f"{nuclear_share:.0f} %" if pd.notna(nuclear_share) else "—")
    with k4:
        _kpi_card("Part renouvelable", f"{renew_share:.0f} %" if pd.notna(renew_share) else "—",
                  sub="hors nucléaire")
    with k5:
        top_src = totals.idxmax() if grand_total else "—"
        _kpi_card("Source #1", top_src, sub=f"{100 * totals.max() / grand_total:.0f} %" if grand_total else "")

    st.markdown("<br>", unsafe_allow_html=True)
    st.caption(
        "Périmètre : solaire, biomasse, éolien terrestre et nucléaire. "
        "Les autres filières et les échanges réseau ne figurent pas dans la source chargée."
    )

    # ---------------------------------------------- Évolution empilée
    st.subheader("Évolution de la production par source")
    fige = go.Figure()
    for src in prod_cols:
        fige.add_trace(go.Scatter(x=obs.index, y=obs[src], name=src, mode="lines",
                                  stackgroup="one", line=dict(width=0.5, color=D.ENERGY_COLORS.get(src, "#888")),
                                  fillcolor=D.ENERGY_COLORS.get(src, "#888")))
    fige.update_layout(height=360, yaxis_title="MW", legend=dict(orientation="h", y=1.02), **_PLOT)
    st.plotly_chart(fige, width="stretch")

    # --------------------------- Répartition (donut) + moyenne par source
    r1, r2 = st.columns(2)
    with r1:
        st.subheader("Répartition moyenne")
        means = obs[prod_cols].mean()
        figp = go.Figure(go.Pie(labels=means.index, values=means.values, hole=0.55, textinfo="percent",
                                marker=dict(colors=[D.ENERGY_COLORS.get(s, "#888") for s in means.index])))
        figp.update_layout(height=340, legend=dict(orientation="h", y=-0.1), **_PLOT)
        st.plotly_chart(figp, width="stretch")
    with r2:
        st.subheader("Production moyenne par source")
        means_sorted = obs[prod_cols].mean().sort_values()
        figb = go.Figure(go.Bar(x=means_sorted.values, y=means_sorted.index, orientation="h",
                                marker_color=[D.ENERGY_COLORS.get(s, "#888") for s in means_sorted.index]))
        figb.update_layout(height=340, xaxis_title="MW", **_PLOT)
        st.plotly_chart(figb, width="stretch")

    # ------------------------------------- Profil journalier moyen par source
    st.subheader("Profil journalier moyen (par heure)")
    prof = obs[prod_cols].copy()
    prof["heure"] = obs.index.hour
    hourly = prof.groupby("heure")[prod_cols].mean()
    figh = go.Figure()
    for src in prod_cols:
        figh.add_trace(go.Scatter(x=hourly.index, y=hourly[src], name=src, mode="lines",
                                  line=dict(color=D.ENERGY_COLORS.get(src, "#888"), width=2.5)))
    figh.update_layout(height=340, xaxis_title="Heure", yaxis_title="MW",
                       xaxis=dict(dtick=2), legend=dict(orientation="h", y=1.02), **_PLOT)
    st.plotly_chart(figh, width="stretch")
    st.caption("Le solaire culmine en milieu de journée, le nucléaire assure la base, l'éolien est variable.")

    # ------------------------------------- Corrélation température par source
    if "temp" in obs.columns:
        st.subheader("Corrélation température — production, par source")
        n = len(prod_cols)
        cols = min(2, n)
        rows = (n + cols - 1) // cols
        figc = make_subplots(rows=rows, cols=cols,
                             subplot_titles=[f"{s} (r={obs['temp'].corr(obs[s]):.2f})" for s in prod_cols],
                             vertical_spacing=0.16, horizontal_spacing=0.1)
        for i, src in enumerate(prod_cols):
            r, c = i // cols + 1, i % cols + 1
            figc.add_trace(go.Scatter(x=obs["temp"], y=obs[src], mode="markers",
                                      marker=dict(color=D.ENERGY_COLORS.get(src, "#888"), size=4, opacity=0.5),
                                      showlegend=False), row=r, col=c)
            figc.update_xaxes(title_text="°C", row=r, col=c)
            figc.update_yaxes(title_text="MW", row=r, col=c)
        figc.update_layout(height=300 * rows, template="plotly_white",
                           margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(figc, width="stretch")
