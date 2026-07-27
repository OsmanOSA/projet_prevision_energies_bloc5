"""Page « Analyse Production » — profils, saisonnalités et corrélation météo."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import data as D
from kpi_card import kpi_card, windowed_delta
from session_config import SessionConfig, create_radio_widget

_PLOT = dict(template="plotly_white", margin=dict(l=10, r=10, t=40, b=10))
_P = "#27ae60"  # couleur production
_DAYS_FR = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]
_MONTHS_FR = ["Jan", "Fév", "Mar", "Avr", "Mai", "Jun", "Jul", "Aoû", "Sep", "Oct", "Nov", "Déc"]


@st.cache_data(ttl=60)
def _load():
    return D.load_observations()


def _window_filter(obs_all):
    max_ts = obs_all.index.max()
    with st.sidebar:
        st.markdown("### Période")
        choice = create_radio_widget("Fenêtre d'affichage", SessionConfig.WINDOWS_DATA_DISPLAY,
                                     session_key="period_window", widget_key="prod_period")
    days = 30 if choice.startswith("30") else 7
    return obs_all[obs_all.index >= max_ts - pd.Timedelta(days=days)], days


def production():
    st.markdown("<h1 class='main-title'>Analyse de la production suivie</h1>", unsafe_allow_html=True)

    obs_all = _load()
    if obs_all.empty or "production_total" not in obs_all.columns:
        st.info("Aucune donnée de production en base.")
        return

    obs, days = _window_filter(obs_all)
    obs = obs.sort_index()
    prod = obs["production_total"].dropna()
    if prod.empty:
        st.info("Aucune donnée sur la fenêtre sélectionnée.")
        return

    # -------------------------------------------------------------- KPI
    has_temp = "temp" in obs.columns
    slope = corr = np.nan
    if has_temp:
        joint = obs[["temp", "production_total"]].dropna()
        if len(joint) >= 3:
            slope = float(np.polyfit(joint["temp"], joint["production_total"], 1)[0])
            corr = float(joint["temp"].corr(joint["production_total"]))

    peak_ts = prod.idxmax()
    avg_prod, avg_delta = windowed_delta(obs_all, "production_total", days)
    _, peak_delta = windowed_delta(obs_all, "production_total", days, agg="max")
    _, base_delta = windowed_delta(obs_all, "production_total", days, agg="min")

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        kpi_card("Production moy.", f"{avg_prod / 1000:,.2f} GW".replace(",", " "),
                 delta_pct=avg_delta)
    with k2:
        kpi_card("Pic", f"{prod.max() / 1000:,.2f} GW".replace(",", " "), delta_pct=peak_delta,
                 sub=peak_ts.strftime("%d/%m %Hh"))
    with k3:
        kpi_card("Base (min)", f"{prod.min() / 1000:,.2f} GW".replace(",", " "), delta_pct=base_delta)
    with k4:
        kpi_card("Sensibilité temp.", f"{slope / 1000:+.2f} GW/°C" if pd.notna(slope) else "—")
    with k5:
        kpi_card("Corrélation temp.", f"r = {corr:.2f}" if pd.notna(corr) else "—")

    st.markdown("<br>", unsafe_allow_html=True)
    st.caption(
        "Production suivie = somme solaire + biomasse + éolien terrestre + nucléaire. "
        "Les autres filières et les échanges réseau ne figurent pas dans la source chargée."
    )

    # ------------------------------------------------- Évolution temporelle
    st.subheader("Évolution temporelle")
    line = obs["production_total"].reindex(pd.date_range(obs.index.min(), obs.index.max(), freq="h"))
    fig = go.Figure(go.Scatter(x=line.index, y=line.values / 1000, line=dict(color=_P, width=2), connectgaps=False))
    fig.update_layout(height=320, yaxis_title="GW", **_PLOT)
    st.plotly_chart(fig, width="stretch")

    # ------------------------------- Distribution + courbe de charge classée
    d1, d2 = st.columns(2)
    with d1:
        st.subheader("Distribution")
        figd = go.Figure(go.Histogram(x=prod / 1000, nbinsx=40, marker_color=_P, opacity=0.75))
        figd.update_layout(height=320, xaxis_title="Production (GW)", yaxis_title="Fréquence", **_PLOT)
        st.plotly_chart(figd, width="stretch")
    with d2:
        st.subheader("Courbe de charge classée")
        sorted_vals = np.sort(prod.values)[::-1] / 1000
        pct = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals) * 100
        figl = go.Figure(go.Scatter(x=pct, y=sorted_vals, fill="tozeroy",
                                    line=dict(color=_P, width=2), fillcolor="rgba(39,174,96,0.12)"))
        figl.update_layout(height=320, xaxis_title="% du temps", yaxis_title="GW", **_PLOT)
        st.plotly_chart(figl, width="stretch")
        st.caption("Puissance dépassée pendant X % du temps (base à droite, pointe à gauche).")

    # ----------------------------------------------- Saisonnalité horaire
    st.subheader("Saisonnalité journalière (par heure)")
    figh = go.Figure(go.Box(x=obs.index.hour, y=obs["production_total"] / 1000,
                            marker_color="#3498db", line_color="#2c3e50", boxpoints="outliers"))
    figh.update_layout(height=340, xaxis_title="Heure", yaxis_title="GW",
                       xaxis=dict(dtick=1), **_PLOT)
    st.plotly_chart(figh, width="stretch")

    # --------------------------- Saisonnalités hebdomadaire + mensuelle
    s1, s2 = st.columns(2)
    with s1:
        st.subheader("Par jour de la semaine")
        wd = obs.index.dayofweek.map(lambda i: _DAYS_FR[i])
        figw = go.Figure(go.Box(x=wd, y=obs["production_total"] / 1000,
                                marker_color=_P, line_color="#2c3e50", boxpoints="outliers"))
        figw.update_layout(height=340, yaxis_title="GW", **_PLOT)
        figw.update_xaxes(categoryorder="array", categoryarray=_DAYS_FR)
        st.plotly_chart(figw, width="stretch")
    with s2:
        st.subheader("Par mois")
        mo = obs.index.month.map(lambda i: _MONTHS_FR[i - 1])
        figm = go.Figure(go.Box(x=mo, y=obs["production_total"] / 1000,
                                marker_color=_P, line_color="#2c3e50", boxpoints="outliers"))
        figm.update_layout(height=340, yaxis_title="GW", **_PLOT)
        figm.update_xaxes(categoryorder="array", categoryarray=_MONTHS_FR)
        st.plotly_chart(figm, width="stretch")

    # ------------------------------------------ Heatmap heure × jour
    st.subheader("Profil moyen — heure × jour de la semaine")
    tmp = obs[["production_total"]].copy()
    tmp["heure"] = obs.index.hour
    tmp["jour"] = obs.index.dayofweek
    pivot = tmp.pivot_table(index="heure", columns="jour", values="production_total", aggfunc="mean")
    pivot = pivot.reindex(columns=[c for c in range(7) if c in pivot.columns])
    fighm = go.Figure(go.Heatmap(
        z=pivot.values / 1000, x=[_DAYS_FR[c] for c in pivot.columns], y=pivot.index,
        colorscale="Greens", colorbar=dict(title="GW")))
    fighm.update_layout(height=420, xaxis_title="Jour", yaxis_title="Heure", **_PLOT)
    st.plotly_chart(fighm, width="stretch")

    # ------------------------------------------ Corrélation température
    if has_temp and pd.notna(slope):
        st.subheader(f"Corrélation température — production (r = {corr:.2f})")
        joint = obs[["temp", "production_total"]].dropna()
        xs = np.linspace(joint["temp"].min(), joint["temp"].max(), 50)
        ys = slope * xs + float(np.polyfit(joint["temp"], joint["production_total"], 1)[1])
        figc = go.Figure()
        figc.add_trace(go.Scatter(x=joint["temp"], y=joint["production_total"] / 1000, mode="markers",
                                  marker=dict(color="#f39c12", size=5, opacity=0.5), name="Observations"))
        figc.add_trace(go.Scatter(x=xs, y=ys / 1000, mode="lines", line=dict(color="#2c3e50", width=2, dash="dash"),
                                  name=f"Tendance ({slope / 1000:+.2f} GW/°C)"))
        figc.update_layout(height=360, xaxis_title="Température (°C)", yaxis_title="Production (GW)",
                           legend=dict(orientation="h", y=1.02), **_PLOT)
        st.plotly_chart(figc, width="stretch")
