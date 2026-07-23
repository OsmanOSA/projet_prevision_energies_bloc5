"""Page « Analyse Consommation » — profils, saisonnalités et thermosensibilité."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import data as D

_PLOT = dict(template="plotly_white", margin=dict(l=10, r=10, t=40, b=10))
_C = "#e74c3c"  # couleur consommation
_DAYS_FR = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]
_MONTHS_FR = ["Jan", "Fév", "Mar", "Avr", "Mai", "Jun", "Jul", "Aoû", "Sep", "Oct", "Nov", "Déc"]


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
        choice = st.radio("Fenêtre d'affichage",
                          ["30 derniers jours", "7 derniers jours"],
                          index=0, key="conso_period")
    days = 30 if choice.startswith("30") else 7
    return obs_all[obs_all.index >= max_ts - pd.Timedelta(days=days)]


def consumption():
    st.markdown("<h1 class='main-title'>Analyse de la consommation</h1>", unsafe_allow_html=True)

    obs_all = _load()
    if obs_all.empty or "consommation_totale" not in obs_all.columns:
        st.info("Aucune donnée de consommation en base.")
        return

    obs = _window_filter(obs_all).sort_index()
    conso = obs["consommation_totale"].dropna()
    if conso.empty:
        st.info("Aucune donnée sur la fenêtre sélectionnée.")
        return

    # -------------------------------------------------------------- KPI
    has_temp = "temp" in obs.columns
    slope = corr = np.nan
    if has_temp:
        joint = obs[["temp", "consommation_totale"]].dropna()
        if len(joint) >= 3:
            slope = float(np.polyfit(joint["temp"], joint["consommation_totale"], 1)[0])
            corr = float(joint["temp"].corr(joint["consommation_totale"]))

    peak_ts = conso.idxmax()
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        _kpi_card("Consommation moy.", f"{conso.mean():,.0f} MW".replace(",", " "))
    with k2:
        _kpi_card("Pic", f"{conso.max():,.0f} MW".replace(",", " "), sub=peak_ts.strftime("%d/%m %Hh"))
    with k3:
        _kpi_card("Base (min)", f"{conso.min():,.0f} MW".replace(",", " "))
    with k4:
        _kpi_card("Thermosensibilité", f"{slope:+.0f} MW/°C" if pd.notna(slope) else "—")
    with k5:
        _kpi_card("Corrélation temp.", f"r = {corr:.2f}" if pd.notna(corr) else "—")

    st.markdown("<br>", unsafe_allow_html=True)

    # ------------------------------------------------- Évolution temporelle
    st.subheader("Évolution temporelle")
    line = obs["consommation_totale"].reindex(pd.date_range(obs.index.min(), obs.index.max(), freq="h"))
    fig = go.Figure(go.Scatter(x=line.index, y=line.values, line=dict(color=_C, width=2), connectgaps=False))
    fig.update_layout(height=320, yaxis_title="MW", **_PLOT)
    st.plotly_chart(fig, width="stretch")

    # ------------------------------- Distribution + courbe de charge classée
    d1, d2 = st.columns(2)
    with d1:
        st.subheader("Distribution")
        figd = go.Figure(go.Histogram(x=conso, nbinsx=40, marker_color=_C, opacity=0.75))
        figd.update_layout(height=320, xaxis_title="Consommation (MW)", yaxis_title="Fréquence", **_PLOT)
        st.plotly_chart(figd, width="stretch")
    with d2:
        st.subheader("Courbe de charge classée")
        sorted_vals = np.sort(conso.values)[::-1]
        pct = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals) * 100
        figl = go.Figure(go.Scatter(x=pct, y=sorted_vals, fill="tozeroy",
                                    line=dict(color=_C, width=2), fillcolor="rgba(231,76,60,0.12)"))
        figl.update_layout(height=320, xaxis_title="% du temps", yaxis_title="MW", **_PLOT)
        st.plotly_chart(figl, width="stretch")
        st.caption("Puissance dépassée pendant X % du temps (base à droite, pointe à gauche).")

    # ----------------------------------------------- Saisonnalité horaire
    st.subheader("Saisonnalité journalière (par heure)")
    figh = go.Figure(go.Box(x=obs.index.hour, y=obs["consommation_totale"],
                            marker_color="#3498db", line_color="#2c3e50", boxpoints="outliers"))
    figh.update_layout(height=340, xaxis_title="Heure", yaxis_title="MW",
                       xaxis=dict(dtick=1), **_PLOT)
    st.plotly_chart(figh, width="stretch")

    # --------------------------- Saisonnalités hebdomadaire + mensuelle
    s1, s2 = st.columns(2)
    with s1:
        st.subheader("Par jour de la semaine")
        wd = obs.index.dayofweek.map(lambda i: _DAYS_FR[i])
        figw = go.Figure(go.Box(x=wd, y=obs["consommation_totale"],
                                marker_color=_C, line_color="#2c3e50", boxpoints="outliers"))
        figw.update_layout(height=340, yaxis_title="MW", **_PLOT)
        figw.update_xaxes(categoryorder="array", categoryarray=_DAYS_FR)
        st.plotly_chart(figw, width="stretch")
    with s2:
        st.subheader("Par mois")
        mo = obs.index.month.map(lambda i: _MONTHS_FR[i - 1])
        figm = go.Figure(go.Box(x=mo, y=obs["consommation_totale"],
                                marker_color=_C, line_color="#2c3e50", boxpoints="outliers"))
        figm.update_layout(height=340, yaxis_title="MW", **_PLOT)
        figm.update_xaxes(categoryorder="array", categoryarray=_MONTHS_FR)
        st.plotly_chart(figm, width="stretch")

    # ------------------------------------------ Heatmap heure × jour
    st.subheader("Profil moyen — heure × jour de la semaine")
    tmp = obs[["consommation_totale"]].copy()
    tmp["heure"] = obs.index.hour
    tmp["jour"] = obs.index.dayofweek
    pivot = tmp.pivot_table(index="heure", columns="jour", values="consommation_totale", aggfunc="mean")
    pivot = pivot.reindex(columns=[c for c in range(7) if c in pivot.columns])
    fighm = go.Figure(go.Heatmap(
        z=pivot.values, x=[_DAYS_FR[c] for c in pivot.columns], y=pivot.index,
        colorscale="YlOrRd", colorbar=dict(title="MW")))
    fighm.update_layout(height=420, xaxis_title="Jour", yaxis_title="Heure", **_PLOT)
    st.plotly_chart(fighm, width="stretch")

    # ------------------------------------------ Corrélation température
    if has_temp and pd.notna(slope):
        st.subheader(f"Corrélation température — consommation (r = {corr:.2f})")
        joint = obs[["temp", "consommation_totale"]].dropna()
        xs = np.linspace(joint["temp"].min(), joint["temp"].max(), 50)
        ys = slope * xs + float(np.polyfit(joint["temp"], joint["consommation_totale"], 1)[1])
        figc = go.Figure()
        figc.add_trace(go.Scatter(x=joint["temp"], y=joint["consommation_totale"], mode="markers",
                                  marker=dict(color="#f39c12", size=5, opacity=0.5), name="Observations"))
        figc.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color="#2c3e50", width=2, dash="dash"),
                                  name=f"Tendance ({slope:+.0f} MW/°C)"))
        figc.update_layout(height=360, xaxis_title="Température (°C)", yaxis_title="Consommation (MW)",
                           legend=dict(orientation="h", y=1.02), **_PLOT)
        st.plotly_chart(figc, width="stretch")
