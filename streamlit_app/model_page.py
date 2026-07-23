"""Page « Performance modèle » — qualité prévu vs réalisé (forecast_metrics)."""

import pandas as pd
import plotly.graph_objects as go
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
def _metrics():
    return D.load_metrics()


def model_performance():
    st.markdown("<h1 class='main-title'>Performance du modèle</h1>", unsafe_allow_html=True)

    m = _metrics()
    if m.empty:
        st.info("Aucune métrique. Lancez le DAG evaluate_daily (ou `python -m scripts.evaluate`).")
        return

    cov = m["coverage"].dropna().mean()
    conso24 = m[(m["variable"] == "consommation_totale") & (m["horizon_h"] == 24)]["mae"]
    versions = m["model_version"].dropna().astype(str).unique()
    version_label = versions[0] if len(versions) == 1 else "mixte / non renseignée"
    if "period_start" in m and m["period_start"].notna().any():
        start = pd.to_datetime(m["period_start"]).min()
        end = pd.to_datetime(m["period_end"]).max()
        st.caption(
            f"Modèle : {version_label} · période évaluée : "
            f"{start:%d/%m/%Y} – {end:%d/%m/%Y}."
        )
    else:
        st.caption(f"Modèle : {version_label}.")

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        _kpi_card("Couverture IC moy.", f"{cov:.0f} %" if pd.notna(cov) else "—", sub="cible ~95 %")
    with k2:
        _kpi_card("Points évalués", f"{int(m['n_points'].sum())}")
    with k3:
        _kpi_card("MAE conso (H+24)", f"{conso24.iloc[0]:,.0f} MW".replace(",", " ") if not conso24.empty else "—")
    with k4:
        _kpi_card("Variables suivies", str(m["variable"].nunique()))

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("MAE moyen des variables électriques")
        power_metrics = m[m["variable"] != "temp"]
        by_var = power_metrics.groupby("variable")["mae"].mean().sort_values()
        figb = go.Figure(go.Bar(
            x=by_var.values, y=by_var.index, orientation="h",
            marker_color="#16a2b8",
        ))
        figb.update_layout(height=320, xaxis_title="MAE (MW)", **_PLOT)
        st.plotly_chart(figb, width="stretch")
        temp_mae = m.loc[m["variable"] == "temp", "mae"].mean()
        if pd.notna(temp_mae):
            st.caption(f"Température : MAE moyenne {temp_mae:.2f} °C (échelle séparée).")
    with c2:
        st.subheader("Couverture IC par variable")
        by_cov = m.groupby("variable")["coverage"].mean().sort_values()
        figc = go.Figure(go.Bar(x=by_cov.values, y=by_cov.index, orientation="h", marker_color="#27ae60"))
        figc.add_vline(x=95, line_dash="dash", line_color="gray", annotation_text="95 %")
        figc.update_layout(height=320, xaxis_title="Couverture (%)", **_PLOT)
        st.plotly_chart(figc, width="stretch")

    # MAE par horizon pour une variable
    st.subheader("Erreur par horizon")
    var = st.selectbox("Variable", sorted(m["variable"].unique()),
                       index=sorted(m["variable"].unique()).index("consommation_totale")
                       if "consommation_totale" in m["variable"].values else 0)
    gv = m[m["variable"] == var].sort_values("horizon_h")
    figh = go.Figure()
    figh.add_trace(go.Bar(x=gv["horizon_h"], y=gv["mae"], name="MAE", marker_color="#16a2b8"))
    unit = "°C" if var == "temp" else "MW"
    figh.update_layout(
        height=300, xaxis_title="Horizon (h)", yaxis_title=f"MAE ({unit})", **_PLOT
    )
    st.plotly_chart(figh, width="stretch")

    st.subheader("Détail par variable × horizon")
    st.dataframe(
        m.rename(columns={"variable": "Variable", "horizon_h": "Horizon", "mae": "MAE", "rmse": "RMSE",
                          "mape": "MAPE %", "bias": "Biais", "coverage": "Couv IC %", "n_points": "N"})
        .round(1),
        width="stretch", hide_index=True,
    )
