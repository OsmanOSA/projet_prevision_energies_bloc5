"""Page « Pipelines » — santé des données et exécutions des DAGs."""

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


@st.cache_data(ttl=30)
def _kpis():
    return D.load_kpis()


@st.cache_data(ttl=30)
def _runs():
    return D.load_runs(50)


def pipelines():
    st.markdown("<h1 class='main-title'>Pipelines &amp; santé des données</h1>", unsafe_allow_html=True)

    kpis = _kpis()
    runs = _runs()

    fresh = kpis.get("freshness_h")
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        _kpi_card("Fraîcheur des données", f"{fresh:.1f} h" if fresh is not None else "—",
                  sub="cible < 2 h")
    with k2:
        _kpi_card("Observations", f"{kpis['n_obs']:,}".replace(",", " "))
    with k3:
        _kpi_card("Prévisions", f"{kpis['n_forecasts']:,}".replace(",", " "))
    with k4:
        _kpi_card("Runs OK (24 h)", str(kpis["runs_ok_24h"]))

    st.caption(
        "Modèle actif : "
        + (str(kpis.get("model_version")) if kpis.get("model_version") else "non renseigné")
    )
    st.markdown("<br>", unsafe_allow_html=True)

    if runs.empty:
        st.info("Aucune exécution de pipeline enregistrée.")
        return

    runs["run_ts"] = pd.to_datetime(runs["run_ts"])

    c1, c2 = st.columns([2, 3])
    with c1:
        st.subheader("Exécutions par DAG")
        by_dag = runs.groupby(["dag_id", "status"]).size().reset_index(name="n")
        fig = go.Figure()
        for status, color in [("success", "#27ae60"), ("failed", "#e74c3c")]:
            sub = by_dag[by_dag["status"] == status]
            if not sub.empty:
                fig.add_trace(go.Bar(x=sub["dag_id"], y=sub["n"], name=status, marker_color=color))
        fig.update_layout(barmode="stack", height=320, yaxis_title="Exécutions",
                          legend=dict(orientation="h", y=1.02), **_PLOT)
        st.plotly_chart(fig, width="stretch")
    with c2:
        st.subheader("Statut du dernier run par DAG")
        last = runs.sort_values("run_ts").groupby("dag_id").tail(1)[["dag_id", "status", "run_ts", "rows"]]
        for _, r in last.iterrows():
            icon = "✅" if r["status"] == "success" else "❌"
            st.markdown(
                f"<div class='card' style='padding:12px 16px; margin:8px 0;'>"
                f"{icon} <strong>{r['dag_id']}</strong> — {r['status']} "
                f"<span style='color:var(--faint);'>· {r['run_ts'].strftime('%d/%m %H:%M')} · {int(r['rows']) if pd.notna(r['rows']) else '—'} lignes</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.subheader("Dernières exécutions")
    st.dataframe(
        runs.rename(columns={"run_ts": "Heure", "dag_id": "DAG", "status": "Statut",
                             "rows": "Lignes", "duration_s": "Durée (s)", "message": "Détail"}),
        width="stretch", hide_index=True,
    )
