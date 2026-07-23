"""Page « Prévisions » — prévision multi-horizon avec intervalles conformes."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import data as D

_PLOT = dict(template="plotly_white", margin=dict(l=10, r=10, t=40, b=10))

# Libellé -> horizon. None = mode « Toutes » (prévisions quotidiennes stockées).
_ECHEANCES = {
    "H+1 (glissant, continu)": 1,
    "H+6 (glissant)": 6,
    "H+12 (glissant)": 12,
    "Toutes (autorégressif, aplati)": None,
}
_LABELS = {
    "temp": "Température (°C)", "SOLAR": "Solaire (MW)", "BIOMASS": "Biomasse (MW)",
    "WIND_ONSHORE": "Éolien (MW)", "NUCLEAR": "Nucléaire (MW)",
    "consommation_totale": "Consommation (MW)",
}


def _fmt(value, signed=False):
    """Formate une erreur en s'adaptant à l'ordre de grandeur.

    Les MW se lisent en entiers (~1000), mais la température se compte en
    dixièmes de degré : arrondir à l'entier afficherait « 0 ».
    """
    if value is None or pd.isna(value):
        return "—"
    decimals = 2 if abs(value) < 100 else 0
    text = f"{value:+,.{decimals}f}" if signed else f"{value:,.{decimals}f}"
    return text.replace(",", " ")


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
def _latest_forecast():
    return D.load_latest_forecast()


@st.cache_data(ttl=60)
def _observations():
    return D.load_observations()


@st.cache_data(ttl=60)
def _forecast_vs_actual(variable, horizon, days):
    return D.load_forecast_vs_actual(variable, horizon, days)


@st.cache_data(ttl=60)
def _horizons():
    return D.available_horizons()


@st.cache_data(ttl=300)
def _singlestep_backtest(variable, days):
    return D.load_singlestep_backtest(variable, days)


@st.cache_data(ttl=300)
def _rolling_backtest(variable, horizon, days):
    return D.load_rolling_backtest(variable, horizon, days)


def forecast():
    st.markdown("<h1 class='main-title'>Prévisions énergétiques</h1>", unsafe_allow_html=True)

    fc = _latest_forecast()
    if fc.empty:
        st.info("Aucune prévision en base. Lancez le DAG forecast_daily (ou `python -m scripts.forecast`).")
        return

    for column in ("run_ts", "origin_ts", "target_ts"):
        fc[column] = pd.to_datetime(fc[column])
    origin = fc["origin_ts"].iloc[0]
    generated_at = fc["run_ts"].iloc[0]
    model_version = fc["model_version"].iloc[0] or "non renseignée"
    horizon = int(fc["horizon_h"].max())

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        _kpi_card("Origine de la prévision", origin.strftime("%d/%m %Hh"))
    with k2:
        _kpi_card("Horizon", f"{horizon} h")
    with k3:
        _kpi_card("Variables prévues", str(fc["variable"].nunique()))
    with k4:
        _kpi_card("Version du modèle", str(model_version))

    st.caption(
        f"Générée le {generated_at:%d/%m/%Y à %H:%M} · "
        f"dernière observation utilisée : {origin:%d/%m/%Y à %H:%M}."
    )
    st.markdown("<br>", unsafe_allow_html=True)

    var = st.selectbox("Variable", D.FEATURES,
                       index=D.FEATURES.index("consommation_totale"),
                       format_func=lambda v: _LABELS.get(v, v))

    g = fc[fc["variable"] == var].sort_values("target_ts")
    obs = _observations()
    hist = obs[var].tail(72) if var in obs.columns else None

    fig = go.Figure()
    if hist is not None and not hist.empty:
        fig.add_trace(go.Scatter(x=hist.index, y=hist.values, name="Historique",
                                 line=dict(color="#7f8c8d", width=2)))
    if g["y_lower"].notna().any() and g["y_upper"].notna().any():
        fig.add_trace(go.Scatter(
            x=list(g["target_ts"]) + list(g["target_ts"])[::-1],
            y=list(g["y_upper"]) + list(g["y_lower"])[::-1],
            fill="toself", fillcolor="rgba(38,188,207,0.18)",
            line=dict(color="rgba(0,0,0,0)"), name="IC ~95 %", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=g["target_ts"], y=g["y_pred"], name="Prévision",
                             line=dict(color="#16a2b8", width=2.5, dash="dash"), mode="lines+markers"))

    # Réalisé arrivé depuis (ingestion horaire) superposé sur l'horizon prévu :
    # permet de comparer à l'œil, heure par heure, prévu vs réel.
    realise = None
    if var in obs.columns:
        realise = obs[var].reindex(g["target_ts"])
        if realise.notna().any():
            fig.add_trace(go.Scatter(x=realise.index, y=realise.values, name="Réalisé",
                                     line=dict(color="#e74c3c", width=2.5),
                                     mode="lines+markers", marker=dict(size=5)))

    fig.update_layout(height=420, yaxis_title=_LABELS.get(var, var),
                      legend=dict(orientation="h", y=1.02), **_PLOT)
    st.plotly_chart(fig, width="stretch")

    if realise is not None and realise.notna().any():
        aligned = pd.DataFrame({"pred": g.set_index("target_ts")["y_pred"], "reel": realise}).dropna()
        if not aligned.empty:
            ecart = (aligned["pred"] - aligned["reel"])
            e1, e2, e3 = st.columns(3)
            with e1:
                _kpi_card("Heures déjà réalisées", f"{len(aligned)} / {len(g)}")
            with e2:
                _kpi_card("Écart absolu moyen", _fmt(ecart.abs().mean()))
            with e3:
                _kpi_card("Biais (prévu − réel)", _fmt(ecart.mean(), signed=True))
    else:
        st.caption("Le réalisé s'affichera au fur et à mesure de l'ingestion horaire.")

    st.caption("Bande = intervalle conforme dynamique (~95 %), il s'élargit avec l'horizon.")

    # Tableau détaillé
    with st.expander("Détail des valeurs prévues"):
        table = g[["target_ts", "horizon_h", "y_pred", "y_lower", "y_upper"]].copy()
        table.columns = ["Horodatage", "Horizon (h)", "Prévision", "Borne basse", "Borne haute"]
        numeric_columns = ["Prévision", "Borne basse", "Borne haute"]
        table[numeric_columns] = table[numeric_columns].round(1)
        st.dataframe(table, width="stretch", hide_index=True)

    _backtesting_section(var)


def _backtesting_section(var):
    """Backtesting : courbe prévue vs courbe réelle sur les derniers jours."""
    st.markdown("---")
    st.markdown("<h2>Backtesting — prévu vs réalisé</h2>", unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1])
    with c1:
        periode = st.radio("Période", ["7 derniers jours", "30 derniers jours"],
                           index=0, horizontal=True, key="bt_days")
    days = 7 if periode.startswith("7") else 30

    with c2:
        choix = st.selectbox(
            "Échéance", list(_ECHEANCES), index=0, key="bt_horizon",
            help="H+1 : le modèle dans son régime natif (1 pas sur observations "
                 "réelles) → aucune accumulation d'erreur. H+6 / H+12 : le modèle "
                 "tourne toutes les 6 / 12 h et prévoit d'autant, en autorégressif. "
                 "« Toutes » : le mode quotidien historisé (origines à 24 h).",
        )

    # Comparaison exacte : un test par préfixe confondrait « H+12 » avec « H+1 ».
    horizon = _ECHEANCES[choix]
    mode_singlestep = horizon == 1
    mode_stocke = horizon is None

    try:
        if mode_singlestep:
            bt, origines = _singlestep_backtest(var, days), []
        elif mode_stocke:
            bt = _forecast_vs_actual(var, None, days)
            if bt.empty:
                st.info("Aucune prévision historisée sur cette période. "
                        "Lancez `python -m scripts.backfill_forecasts 30 24`.")
                return
            bt = bt.dropna(subset=["y_true"]).sort_values("target_ts")
            origines = sorted(pd.to_datetime(bt["origin_ts"]).unique()) if "origin_ts" in bt else []
        else:
            bt, origines = _rolling_backtest(var, horizon, days)
    except Exception as exc:
        st.error(f"Calcul du backtesting impossible : {exc}")
        return

    if bt.empty:
        st.info("Pas assez d'observations sur la période.")
        return

    bt = bt.dropna(subset=["y_true"]).sort_values("target_ts")
    if bt.empty:
        st.info("Les prévisions de cette période n'ont pas encore de réalisé associé.")
        return

    err = bt["y_pred"] - bt["y_true"]
    mae = err.abs().mean()
    rmse = float(((err ** 2).mean()) ** 0.5)
    biais = err.mean()
    couverture = None
    interval_mask = bt["y_lower"].notna() & bt["y_upper"].notna()
    if interval_mask.any():
        dedans = (
            (bt.loc[interval_mask, "y_true"] >= bt.loc[interval_mask, "y_lower"])
            & (bt.loc[interval_mask, "y_true"] <= bt.loc[interval_mask, "y_upper"])
        )
        couverture = 100 * dedans.mean()

    nb_origines = bt["target_ts"].dt.normalize().nunique() if horizon is not None else None
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        _kpi_card("MAE", _fmt(mae), sub=f"{len(bt)} points")
    with k2:
        _kpi_card("RMSE", _fmt(rmse))
    with k3:
        _kpi_card("Biais", _fmt(biais, signed=True), sub="prévu − réel")
    with k4:
        _kpi_card("Couverture IC", f"{couverture:.0f} %" if couverture is not None else "—")

    if mode_singlestep:
        st.caption(
            "**H+1 glissant** : à chaque heure, la valeur suivante est prédite à partir des "
            "observations **réelles** — aucune autorégression, donc aucune accumulation d'erreur. "
            "C'est la performance intrinsèque du modèle (calculée à la volée, non persistée)."
        )
    elif horizon is not None:
        st.caption(
            f"**H+{horizon} glissant** : le modèle tourne toutes les {horizon} h et prévoit "
            f"{horizon} h en autorégressif ({len(origines)} origines). Entre deux coutures "
            "l'erreur s'accumule, puis de nouvelles observations la remettent à zéro."
        )

    fig = go.Figure()
    if bt["y_lower"].notna().any():
        fig.add_trace(go.Scatter(
            x=list(bt["target_ts"]) + list(bt["target_ts"])[::-1],
            y=list(bt["y_upper"]) + list(bt["y_lower"])[::-1],
            fill="toself", fillcolor="rgba(38,188,207,0.12)",
            line=dict(color="rgba(0,0,0,0)"), name="IC ~95 %", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=bt["target_ts"], y=bt["y_true"], name="Réalisé",
                             line=dict(color="#e74c3c", width=2)))
    fig.add_trace(go.Scatter(x=bt["target_ts"], y=bt["y_pred"], name="Prévu",
                             line=dict(color="#16a2b8", width=2, dash="dash")))

    # Chaque origine démarre une nouvelle prévision autorégressive : on marque
    # la couture pour que l'aplatissement reste lisible.
    if origines:
        debut, fin = bt["target_ts"].min(), bt["target_ts"].max()
        for origine in origines:
            if debut <= origine <= fin:
                fig.add_vline(x=origine, line_width=1, line_dash="dot",
                              line_color="rgba(120,140,160,0.55)")

    fig.update_layout(height=400, yaxis_title=_LABELS.get(var, var),
                      legend=dict(orientation="h", y=1.02), **_PLOT)
    st.plotly_chart(fig, width="stretch")
    if origines:
        st.caption("Traits verticaux = début de chaque prévision (nouvelle origine). "
                   "Entre deux traits, l'erreur s'accumule par autorégression.")

    # Distribution des erreurs : révèle un biais systématique éventuel
    fige = go.Figure(go.Histogram(x=err, nbinsx=40, marker_color="#16a2b8", opacity=0.75))
    fige.add_vline(x=0, line_dash="dash", line_color="gray")
    fige.update_layout(height=260, xaxis_title="Erreur (prévu − réel)",
                       yaxis_title="Fréquence", **_PLOT)
    st.plotly_chart(fige, width="stretch")
