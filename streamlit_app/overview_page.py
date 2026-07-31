"""Page « Vue d'ensemble » — synthèse façon Power BI (KPI + analyses clés)."""

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

import data as D
from kpi_card import kpi_card, windowed_delta
from session_config import SessionConfig, create_radio_widget

_SOURCE_LABELS = {
    "SOLAR": "Solaire",
    "BIOMASS": "Biomasse",
    "WIND_ONSHORE": "Éolien terrestre",
    "NUCLEAR": "Nucléaire",
}

_PLOT = dict(template="plotly_white", margin=dict(l=10, r=10, t=40, b=10))


@st.cache_data(ttl=60)
def _load(start, end):
    return D.load_observations(start, end)


def _period_filter(obs_all):
    """Fenêtre glissante relative à la dernière donnée disponible (7 j par défaut)."""
    max_ts = obs_all.index.max()
    with st.sidebar:
        st.markdown("### Période")
        choice = create_radio_widget("Fenêtre d'affichage", SessionConfig.WINDOWS_DATA_DISPLAY,
                                     session_key="period_window", widget_key="ov_period")
    days = 7 if choice.startswith("7") else 30
    start = max_ts - pd.Timedelta(days=days)
    end = max_ts + pd.Timedelta(hours=1)
    return start, end, days


def overview():
    st.markdown("<h1 class='main-title'>Vue d'ensemble</h1>", unsafe_allow_html=True)

    obs_all = _load(None, None)
    if obs_all.empty:
        st.info("Aucune donnée en base. Lancez l'ingestion (DAG ingest_hourly).")
        return

    start, end, days = _period_filter(obs_all)
    obs = _load(start, end)
    if obs.empty:
        st.info("Aucune donnée sur la période sélectionnée.")
        return

    has_prod = "production_total" in obs.columns
    has_conso = "consommation_totale" in obs.columns
    prod_total = obs["production_total"] if has_prod else None

    # ------------------------------------------------------------------ KPI
    peak_conso = obs["consommation_totale"].max() if has_conso else np.nan
    peak_hour = obs["consommation_totale"].idxmax() if has_conso else None
    peak_prod = prod_total.max() if has_prod else np.nan
    peak_prod_hour = prod_total.idxmax() if has_prod else None

    deficit_mw = np.nan
    if has_prod and has_conso:
        deficit_mw = (prod_total - obs["consommation_totale"]).mean()

    avg_conso, conso_delta = windowed_delta(obs_all, "consommation_totale", days)
    avg_prod, prod_delta = windowed_delta(obs_all, "production_total", days)
    _, peak_conso_delta = windowed_delta(obs_all, "consommation_totale", days, agg="max")
    _, peak_prod_delta = windowed_delta(obs_all, "production_total", days, agg="max")

    # Fenêtre de comparaison nommée explicitement dans les infobulles : le
    # badge compare aux `days` jours qui précèdent la fenêtre affichée.
    periode = f"les {days} jours précédents"

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        kpi_card("Consommation moy.", f"{avg_conso/1000:,.2f} GW" if pd.notna(avg_conso) else "—",
                 delta_pct=conso_delta, higher_is_better=False, period_label=periode)
    with c2:
        kpi_card("Pic de conso.", f"{peak_conso/1000:,.2f} GW" if pd.notna(peak_conso) else "—",
                 delta_pct=peak_conso_delta, higher_is_better=False, period_label=periode,
                 sub=peak_hour.strftime("%d/%m %Hh") if peak_hour is not None else "")
    with c3:
        kpi_card("Production suivie moy.", f"{avg_prod/1000:,.2f} GW" if pd.notna(avg_prod) else "—",
                 delta_pct=prod_delta, period_label=periode)
    with c4:
        kpi_card("Pic de production", f"{peak_prod/1000:,.2f} GW" if pd.notna(peak_prod) else "—",
                 delta_pct=peak_prod_delta, period_label=periode,
                 sub=peak_prod_hour.strftime("%d/%m %Hh") if peak_prod_hour is not None else "")
    with c5:
        kpi_card("Écart moyen (partiel)",
                 f"{deficit_mw/1000:+,.2f} GW" if pd.notna(deficit_mw) else "—")

    st.markdown("<br>", unsafe_allow_html=True)

    # ----------------------------------------------------- Conso vs production
    col_line, col_bar = st.columns(2)
    with col_line:
        st.subheader(
            "Consommation vs production suivie",
            help=(
                "Production suivie = somme solaire + biomasse + éolien terrestre + "
                "nucléaire. L'hydraulique, le thermique fossile, les échanges et les "
                "pertes réseau ne sont pas disponibles ; l'écart affiché n'est donc pas "
                "le solde électrique national."
            ),
        )
        # Réindexation horaire continue : les heures manquantes deviennent NaN,
        # ce qui casse la ligne au lieu de relier les blocs par une droite fictive.
        obs_line = obs.reindex(pd.date_range(obs.index.min(), obs.index.max(), freq="h"))
        fig = go.Figure()
        if has_conso:
            fig.add_trace(go.Scatter(x=obs_line.index, y=obs_line["consommation_totale"] / 1000, name="Consommation",
                                     line=dict(color="#e74c3c", width=2), connectgaps=False))
        if has_prod:
            fig.add_trace(go.Scatter(x=obs_line.index, y=obs_line["production_total"] / 1000, name="Production suivie",
                                     line=dict(color="#27ae60", width=2), connectgaps=False))
        fig.update_layout(height=360, yaxis=dict(title="GW", range=[0, 70], dtick=20),
                          legend=dict(orientation="h", y=1.02), **_PLOT)
        st.plotly_chart(fig, width="stretch")

    # ------------------------------------------------------ Écart journalier
    if has_prod and has_conso:
        with col_bar:
            st.subheader("Evolution du déficit (PROD - CONSO)")
            daily = (prod_total - obs["consommation_totale"]).resample("D").mean() / 1000.0
            colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in daily]
            figd = go.Figure(go.Bar(x=daily.index, y=daily.values, marker_color=colors,
                                    marker_line=dict(color="rgba(0,0,0,0.15)", width=1)))
            figd.add_hline(y=0, line_dash="dash", line_color="gray",
                           annotation_text="Équilibre", annotation_position="bottom right")
            figd.update_layout(height=360, yaxis_title="Écart partiel (GW)", **_PLOT)
            st.plotly_chart(figd, width="stretch")

    # ------------------------------------------------- Répartition production
    if all(c in obs.columns for c in D.PRODUCTION_SOURCES):
        st.subheader("Répartition de la production suivie par source")
        source_means = {s: float(obs[s].mean()) / 1000.0 for s in D.PRODUCTION_SOURCES}  # GW, cohérent avec le reste de la page
        pie_data = [
            {
                "name": _SOURCE_LABELS[source],
                "value": round(value, 2),
                "itemStyle": {"color": D.ENERGY_COLORS.get(source, "#999999")},
            }
            for source, value in source_means.items()
        ]

        options = {
            "tooltip": {"trigger": "item", "formatter": "{b}: {c} GW ({d}%)"},
            "legend": {"top": "0%", "left": "center"},
            "series": [{
                "name": "Source",
                "type": "pie",
                "radius": ["37%", "63%"],
                "center": ["50%", "58%"],
                "itemStyle": {
                    "borderColor": "#FFFFFF",
                    "borderWidth": 5,
                    "borderRadius": 11,
                },
                "label": {"formatter": "{b}\n{d}%"},
                "data": pie_data,
            }],
        }

        echarts_html = f"""
        <div id="echarts_prod_sources" style="width:100%;height:420px;"></div>
        <script src="https://cdn.jsdelivr.net/npm/echarts@5.4.3/dist/echarts.min.js"></script>
        <script>
            var chartDom = document.getElementById('echarts_prod_sources');
            var myChart = echarts.init(chartDom);
            var option = {json.dumps(options)};
            myChart.setOption(option);
            window.addEventListener('resize', myChart.resize);
        </script>
        """
        st.iframe(echarts_html, height=430)

    # ---------------------------------------------------------- Distributions
    st.subheader("Distributions des variables principales")
    main_vars = [v for v in ["consommation_totale", "production_total", "temp"] if v in obs.columns]
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
