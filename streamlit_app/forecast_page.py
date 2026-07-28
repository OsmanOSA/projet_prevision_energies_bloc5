"""Page « Prévisions » — prévision multi-horizon avec intervalles conformes."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import data as D
from kpi_card import kpi_card
from session_config import SessionConfig, create_radio_widget

_PLOT = dict(template="plotly_white", margin=dict(l=10, r=10, t=40, b=10))

# Libellé -> horizon. None = mode « Toutes » (prévisions quotidiennes stockées).
# Chaque horizon est un modèle direct dédié (pas d'autorégressif) : plus de
# distinction « glissant » vs « autorégressif » avec cette architecture.
_ECHEANCES = {
    "H+1": 1,
    "H+6": 6,
    "H+12": 12,
    "H+24": 24,
    "Toutes (historisé)": None,
}
_LABELS = {
    "production_total": "Production (GW)",
    "consommation_totale": "Consommation (GW)",
    "SOLAR": "Solaire (GW)",
    "BIOMASS": "Biomasse (GW)",
    "WIND_ONSHORE": "Éolien terrestre (GW)",
    "NUCLEAR": "Nucléaire (GW)",
}
# temp est la seule variable non énergétique de D.FEATURES (°C, jamais GW).
_POWER_SCALE = 1000.0  # MW -> GW


def _var_scale(var: str) -> tuple[float, str]:
    """(diviseur, unité) selon la variable : GW pour les grandeurs
    énergétiques, °C (inchangé) pour la température."""
    if var == "temp":
        return 1.0, "°C"
    return _POWER_SCALE, "GW"
_SEVERITY_COLORS = {"green": "#27ae60", "orange": "#f39c12", "red": "#e74c3c"}


def _fmt(value, signed=False):
    """Formate une erreur en s'adaptant à l'ordre de grandeur.

    Les GW se lisent avec 2 décimales (valeurs typiquement < 100), la
    température en dixièmes de degré : arrondir à l'entier afficherait « 0 ».
    """
    if value is None or pd.isna(value):
        return "—"
    decimals = 2 if abs(value) < 100 else 0
    text = f"{value:+,.{decimals}f}" if signed else f"{value:,.{decimals}f}"
    return text.replace(",", " ")


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
def _backtest(variable, horizon, days):
    return D.load_backtest(variable, horizon, days)


@st.cache_data(ttl=300)
def _rte_forecast(start, end):
    return D.load_rte_consumption_forecast(start, end)


@st.cache_data(ttl=60)
def _forecast_for_day(variable, start, end):
    return D.load_forecast_for_day(variable, start, end)


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

    st.caption(
        f"Générée le {generated_at:%d/%m/%Y à %H:%M} · "
        f"dernière observation utilisée : {origin:%d/%m/%Y à %H:%M}."
    )
    st.markdown("<br>", unsafe_allow_html=True)

    obs = _observations()

    var = st.selectbox("Variable", D.FEATURES,
                       index=D.FEATURES.index("consommation_totale"),
                       format_func=lambda v: _LABELS.get(v, v))
    scale, unit = _var_scale(var)

    # Alignée sur la journée civile (00h-23h) RÉELLE ("aujourd'hui"), pas sur
    # le jour de l'origine : juste après minuit, la dernière heure observée
    # complète porte quasi toujours la date de la veille (l'ingestion horaire
    # ingère l'heure qui vient de s'écouler), donc ancrer sur `origin.date()`
    # affiche systématiquement le jour civil d'hier. On va chercher, heure
    # par heure, la prévision issue de l'origine la plus récente disponible
    # (cf. `load_forecast_for_day`) sur le jour civil courant -- exactement
    # comme RTE (J-1) qui porte lui aussi sur la journée entière.
    day_start = pd.Timestamp.now(tz=D.DISPLAY_TZ).tz_localize(None).normalize()
    day_end = day_start + pd.Timedelta(hours=23)

    g = _forecast_for_day(var, day_start, day_end)
    if g.empty:
        # Repli : aucune prévision pour cette journée -> le batch le plus
        # récent tel quel, plutôt qu'un graphique vide.
        g = fc[fc["variable"] == var].sort_values("target_ts")

    hist = obs[var].tail(72) if var in obs.columns else None

    realise = obs[var].reindex(g["target_ts"]) if var in obs.columns else None
    aligned = None
    if realise is not None and realise.notna().any():
        aligned = pd.DataFrame({"pred": g.set_index("target_ts")["y_pred"], "reel": realise}).dropna()
        if aligned.empty:
            aligned = None

    # Repère de crédibilité RTE (J-1), seulement pour consommation_totale
    # (cf. scripts/fetch_rte_forecast.py) — récupéré une seule fois, réutilisé
    # à la fois pour la carte « Précision vs RTE » ci-dessous et pour la
    # courbe superposée sur le graphique plus bas.
    rte = pd.DataFrame()
    if var == "consommation_totale" and not g.empty:
        rte = _rte_forecast(g["target_ts"].min(), g["target_ts"].max())

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        kpi_card("Origine de la prévision", origin.strftime("%d/%m %Hh"))
    if aligned is not None:
        ecart = aligned["pred"] - aligned["reel"]
        mae = ecart.abs().mean()
        # WAPE : erreur totale rapportée au volume réel total (mieux que
        # rapporter à une valeur fixe arbitraire, ou que le MAPE si la
        # variable peut s'approcher de 0 sur certaines heures).
        wape = 100 * mae / aligned["reel"].abs().mean()
        with k2:
            kpi_card("Écart absolu moyen", f"{_fmt(mae / scale)} {unit}",
                     sub=f"Sur {len(aligned)} / {len(g)} heures réalisées")
        with k3:
            kpi_card("WAPE partielle", f"{_fmt(wape)} %",
                     sub=f"Sur {len(aligned)} / {len(g)} heures réalisées")
        with k4:
            biais = ecart.mean()
            sens = "Sous-estimation moyenne" if biais < 0 else "Surestimation moyenne"
            kpi_card("Biais moyen (prévu − réel)", f"{_fmt(biais / scale, signed=True)} {unit}", sub=sens)
        with k5:
            if var != "consommation_totale":
                kpi_card("GAIN DE MAE VS RTE J-1", "—", sub="Réservé à la consommation")
            else:
                skill_rte, n_rte = None, 0
                if not rte.empty:
                    # Comparaison stricte sur l'intersection des heures : mêmes
                    # target_ts pour les deux MAE, sinon le score serait faussé
                    # par un échantillon différent entre modèle et RTE.
                    rte_aligned = aligned.merge(
                        rte.rename(columns={"y_pred": "y_pred_rte"}).set_index("target_ts"),
                        left_index=True, right_index=True, how="inner",
                    )
                    if not rte_aligned.empty:
                        mae_modele_rte = (rte_aligned["pred"] - rte_aligned["reel"]).abs().mean()
                        mae_rte = (rte_aligned["y_pred_rte"] - rte_aligned["reel"]).abs().mean()
                        if mae_rte:
                            skill_rte = 100 * (1 - mae_modele_rte / mae_rte)
                            n_rte = len(rte_aligned)
                if skill_rte is not None:
                    # Seuil d'égalité : en dessous de 2 points d'écart, la
                    # différence tient plus au bruit qu'à un vrai avantage
                    # (surtout avec aussi peu d'heures communes). Le badge
                    # ↑/↓ (delta_pct, même code que les autres pages) porte
                    # le chiffre ; le mot ne fait que trancher qui gagne.
                    if abs(skill_rte) < 2:
                        verdict = "Égalité"
                    elif skill_rte > 0:
                        verdict = "Nous"
                    else:
                        verdict = "RTE"
                    kpi_card("GAIN DE MAE VS RTE (J-1)", verdict, delta_pct=skill_rte,
                             sub=f"Sur {n_rte} heures communes")
                else:
                    kpi_card("GAIN DE MAE VS RTE (J-1)", "—", sub="Pas de recouvrement RTE")
    else:
        for i, col in enumerate((k2, k3, k4)):
            with col:
                kpi_card("En attente du réalisé", "—", key=f"en-attente-{i}")
        with k5:
            kpi_card("En attente du réalisé", "—", key="en-attente-rte")

    fig = go.Figure()
    if hist is not None and not hist.empty:
        fig.add_trace(go.Scatter(x=hist.index, y=hist.values / scale, name="Historique",
                                 line=dict(color="#7f8c8d", width=2)))
    if g["y_lower"].notna().any() and g["y_upper"].notna().any():
        fig.add_trace(go.Scatter(
            x=list(g["target_ts"]) + list(g["target_ts"])[::-1],
            y=list(g["y_upper"] / scale) + list(g["y_lower"] / scale)[::-1],
            fill="toself", fillcolor="rgba(38,188,207,0.18)",
            line=dict(color="rgba(0,0,0,0)"), name="IC ~95 %", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=g["target_ts"], y=g["y_pred"] / scale, name="Prévision",
                             line=dict(color="#16a2b8", width=2.5, dash="dash"), mode="lines+markers"))

    # Repère de crédibilité : prévision officielle RTE J-1 (déjà chargée
    # ci-dessus pour la carte « Précision vs RTE »), comparaison par
    # target_ts, indépendante de notre propre notion d'origine/horizon.
    # RTE (consommation uniquement) est toujours en puissance -> /1000 fixe.
    if not rte.empty:
        fig.add_trace(go.Scatter(x=rte["target_ts"], y=rte["y_pred"] / _POWER_SCALE, name="Prévision RTE (J-1)",
                                 line=dict(color="#8e44ad", width=2, dash="dot"), mode="lines+markers",
                                 marker=dict(size=4)))

    # Réalisé arrivé depuis (ingestion horaire) superposé sur l'horizon prévu :
    # permet de comparer à l'œil, heure par heure, prévu vs réel.
    if realise is not None and realise.notna().any():
        fig.add_trace(go.Scatter(x=realise.index, y=realise.values / scale, name="Réalisé",
                                 line=dict(color="#e74c3c", width=2.5),
                                 mode="lines+markers", marker=dict(size=5)))

    fig.update_layout(height=420, yaxis_title=_LABELS.get(var, var),
                      legend=dict(orientation="h", y=1.02), **_PLOT)
    st.plotly_chart(fig, width="stretch")

    if aligned is None:
        st.caption("Le réalisé s'affichera au fur et à mesure de l'ingestion horaire.")
    st.caption("Bande = intervalle conforme dynamique (~95 %), il s'élargit avec l'horizon.")
    if var == "consommation_totale":
        st.caption(
            "Prévision RTE (J-1) : prévision officielle publiée par RTE la veille pour le "
            "jour affiché — repère externe indépendant, jamais utilisé pour entraîner notre modèle."
        )
        st.caption(
            "GAIN DE MAE VS RTE (J-1) = 100 × (1 − MAE modèle / MAE RTE), calculée sur les heures "
            "déjà réalisées communes aux deux prévisions. Positif = notre modèle fait moins "
            "d'erreur que RTE sur ces heures ; négatif = l'inverse."
        )

    # Tableau détaillé
    with st.expander("Détail des valeurs prévues"):
        table = g[["target_ts", "horizon_h", "y_pred", "y_lower", "y_upper"]].copy()
        numeric_columns = ["y_pred", "y_lower", "y_upper"]
        table[numeric_columns] = (table[numeric_columns] / scale).round(2)
        table.columns = ["Horodatage", "Horizon (h)", f"Prévision ({unit})",
                         f"Borne basse ({unit})", f"Borne haute ({unit})"]
        st.dataframe(table, width="stretch", hide_index=True)

    _deficit_section(fc, obs)
    _backtesting_section(var)


def _deficit_section(fc, obs):
    """Évolution du déficit (production − consommation) : historique réalisé
    prolongé par la prévision directe à 24h, avec un code couleur de
    sévérité par rapport à l'équilibre (0) et à un seuil critique
    data-driven (10e percentile du déficit historique) — pas de valeur
    arbitraire codée en dur.
    """
    if "production_total" not in obs.columns or "consommation_totale" not in obs.columns:
        return

    deficit_hist_full = ((obs["production_total"] - obs["consommation_totale"]) / _POWER_SCALE).dropna()
    if deficit_hist_full.empty:
        return

    prod_fc = fc[fc["variable"] == "production_total"].set_index("target_ts")
    conso_fc = fc[fc["variable"] == "consommation_totale"].set_index("target_ts")
    common = prod_fc.index.intersection(conso_fc.index)
    if common.empty:
        return

    st.markdown("---")
    st.markdown("<h2>Évolution du déficit</h2>", unsafe_allow_html=True)

    critical_threshold = float(deficit_hist_full.quantile(0.10))
    deficit_hist = deficit_hist_full.tail(7 * 24)
    deficit_pred = ((prod_fc.loc[common, "y_pred"] - conso_fc.loc[common, "y_pred"]) / _POWER_SCALE).sort_index()

    # Bande d'incertitude approximative : bornes de prod/conso combinées en
    # supposant leurs erreurs indépendantes (deux modèles entraînés
    # séparément) -> var(A-B) = var(A) + var(B), d'où la combinaison des
    # bornes plutôt qu'une simple soustraction des IC.
    has_bounds = prod_fc["y_lower"].notna().any() and conso_fc["y_lower"].notna().any()
    if has_bounds:
        deficit_lower = ((prod_fc.loc[common, "y_lower"] - conso_fc.loc[common, "y_upper"]) / _POWER_SCALE).sort_index()
        deficit_upper = ((prod_fc.loc[common, "y_upper"] - conso_fc.loc[common, "y_lower"]) / _POWER_SCALE).sort_index()

    def _severity(value):
        if value >= 0:
            return "green"
        if value >= critical_threshold:
            return "orange"
        return "red"

    colors = [_SEVERITY_COLORS[_severity(v)] for v in deficit_pred.values]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=deficit_hist.index, y=deficit_hist.values, name="Réalisé",
                             line=dict(color="#95a5a6", width=2)))
    if has_bounds:
        fig.add_trace(go.Scatter(
            x=list(deficit_pred.index) + list(deficit_pred.index)[::-1],
            y=list(deficit_upper.values) + list(deficit_lower.values)[::-1],
            fill="toself", fillcolor="rgba(38,188,207,0.15)",
            line=dict(color="rgba(0,0,0,0)"), name="IC ~95 % (approx.)", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=deficit_pred.index, y=deficit_pred.values, name="Prévu",
                             mode="lines+markers",
                             line=dict(color="rgba(120,140,160,0.6)", width=1, dash="dot"),
                             marker=dict(color=colors, size=9, line=dict(color="white", width=1))))

    fig.add_hline(y=0, line_dash="dash", line_color="gray",
                  annotation_text="Équilibre", annotation_position="top left")
    fig.add_hline(y=critical_threshold, line_dash="dot", line_color="#e74c3c",
                  annotation_text="Seuil critique (10ᵉ percentile historique)",
                  annotation_position="bottom left")
    if len(deficit_hist):
        fig.add_vline(x=deficit_hist.index[-1], line_dash="dash", line_color="rgba(0,0,0,0.4)")

    fig.update_layout(height=380, yaxis_title="Déficit = production − consommation (GW)",
                      legend=dict(orientation="h", y=1.02), **_PLOT)
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Trait vertical = passage de l'historique réalisé à la prévision (24h). "
        "Couleur des points prévus : vert = équilibre ou surplus, orange = déficit "
        "modéré, rouge = déficit sévère (parmi les 10 % pires historiquement). "
        "Bande IC = approximation (bornes de production/consommation combinées en "
        "supposant leurs erreurs indépendantes)."
    )


def _backtesting_section(var):
    """Backtesting : courbe prévue vs courbe réelle sur les derniers jours."""
    scale, unit = _var_scale(var)
    st.markdown("---")
    st.markdown("<h2>Backtesting — prévu vs réalisé</h2>", unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### Période")
        periode = create_radio_widget("Fenêtre d'affichage", SessionConfig.WINDOWS_DATA_DISPLAY,
                                      session_key="period_window", widget_key="bt_days")
    days = 7 if periode.startswith("7") else 30

    choix = st.selectbox(
        "Échéance", list(_ECHEANCES), index=0, key="bt_horizon",
        help="Chaque échéance est prédite directement depuis son origine par "
             "un modèle dédié (pas d'autorégression, pas d'accumulation "
             "d'erreur d'un horizon à l'autre). « Toutes » : le mode "
             "quotidien historisé (origines à 24 h, persisté en base).",
    )

    horizon = _ECHEANCES[choix]
    mode_stocke = horizon is None

    try:
        if mode_stocke:
            bt = _forecast_vs_actual(var, None, days)
            if bt.empty:
                st.info("Aucune prévision historisée sur cette période. "
                        "Lancez `python -m scripts.backfill_forecasts 30 24`.")
                return
            bt = bt.dropna(subset=["y_true"]).sort_values("target_ts")
            origines = sorted(pd.to_datetime(bt["origin_ts"]).unique()) if "origin_ts" in bt else []
        else:
            bt = _backtest(var, horizon, days)
            origines = []
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
    wape = 100 * mae / bt["y_true"].abs().mean()
    couverture = None
    interval_mask = bt["y_lower"].notna() & bt["y_upper"].notna()
    if interval_mask.any():
        dedans = (
            (bt.loc[interval_mask, "y_true"] >= bt.loc[interval_mask, "y_lower"])
            & (bt.loc[interval_mask, "y_true"] <= bt.loc[interval_mask, "y_upper"])
        )
        couverture = 100 * dedans.mean()

    sens = "Sous-estimation moyenne" if biais < 0 else "Surestimation moyenne"
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        kpi_card("MAE", f"{_fmt(mae / scale)} {unit}", sub=f"{len(bt)} points")
    with k2:
        kpi_card("RMSE", f"{_fmt(rmse / scale)} {unit}")
    with k3:
        kpi_card("WAPE", f"{_fmt(wape)} %", sub="vs valeur réelle moyenne")
    with k4:
        kpi_card("Biais (prévu − réel)", f"{_fmt(biais / scale, signed=True)} {unit}", sub=sens)
    with k5:
        kpi_card("Couverture IC", f"{couverture:.0f} %" if couverture is not None else "—")

    if horizon is not None:
        st.caption(
            f"**H+{horizon} direct** : chaque point est prédit depuis son origine (H-{horizon}) "
            "par le modèle dédié à cette échéance — aucune autorégression, donc aucune "
            "accumulation d'erreur d'un horizon à l'autre. Calculé à la volée, non persisté."
        )

    fig = go.Figure()
    if bt["y_lower"].notna().any():
        fig.add_trace(go.Scatter(
            x=list(bt["target_ts"]) + list(bt["target_ts"])[::-1],
            y=list(bt["y_upper"] / scale) + list(bt["y_lower"] / scale)[::-1],
            fill="toself", fillcolor="rgba(38,188,207,0.12)",
            line=dict(color="rgba(0,0,0,0)"), name="IC ~95 %", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=bt["target_ts"], y=bt["y_true"] / scale, name="Réalisé",
                             line=dict(color="#e74c3c", width=2)))
    fig.add_trace(go.Scatter(x=bt["target_ts"], y=bt["y_pred"] / scale, name="Prévu",
                             line=dict(color="#16a2b8", width=2, dash="dash")))

    # Chaque origine correspond à un batch de prévision historisé distinct
    # (une par jour, via scripts.backfill_forecasts) : on marque le début de
    # chacun pour repérer les recalages successifs sur le graphique aplati.
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
        st.caption("Traits verticaux = début de chaque prévision historisée (nouvelle origine).")

    # Distribution des erreurs : révèle un biais systématique éventuel
    fige = go.Figure(go.Histogram(x=err / scale, nbinsx=40, marker_color="#16a2b8", opacity=0.75))
    fige.add_vline(x=0, line_dash="dash", line_color="gray")
    fige.update_layout(height=260, xaxis_title=f"Erreur (prévu − réel) ({unit})",
                       yaxis_title="Fréquence", **_PLOT)
    st.plotly_chart(fige, width="stretch")
