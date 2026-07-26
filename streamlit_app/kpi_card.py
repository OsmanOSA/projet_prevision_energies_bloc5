"""Carte KPI partagée : valeur + delta vs période précédente.

Un seul composant pour toutes les pages (évite les 6 copies de `_kpi_card`
qui ne différaient que par du copier-coller). Le delta est optionnel : une
page qui n'a pas de série temporelle sous-jacente (version du modèle, statut
d'un run…) peut appeler `kpi_card(label, value)` seule.
"""

import re

import pandas as pd
import streamlit as st

_GOOD = "#27ae60"
_BAD = "#e74c3c"


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def _slug(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-") or "kpi"


def windowed_delta(obs_all: pd.DataFrame, column: str, days: int, agg: str = "mean"):
    """Valeur agrégée sur la fenêtre courante + delta % vs la fenêtre précédente
    de même durée. Retourne (valeur, delta_pct) ; delta_pct vaut None si
    l'historique est insuffisant pour comparer.
    """
    if obs_all is None or obs_all.empty or column not in obs_all.columns:
        return None, None

    series = obs_all[column].dropna()
    if series.empty:
        return None, None

    max_ts = series.index.max()
    current = series[series.index > max_ts - pd.Timedelta(days=days)]
    previous = series[
        (series.index <= max_ts - pd.Timedelta(days=days))
        & (series.index > max_ts - pd.Timedelta(days=2 * days))
    ]
    if current.empty:
        return None, None

    value = getattr(current, agg)()
    delta_pct = None
    if not previous.empty:
        prev_value = getattr(previous, agg)()
        if pd.notna(prev_value) and prev_value != 0:
            delta_pct = 100 * (value - prev_value) / abs(prev_value)

    return value, delta_pct


def kpi_card(label: str, value: str, delta_pct: float | None = None, sub: str = "",
            higher_is_better: bool = True, key: str | None = None):
    """Carte KPI. `delta_pct` omis -> rendu simple (label/valeur/sub).

    `higher_is_better` fixe le sens de la couleur : True (par défaut, ex.
    production) -> hausse verte ; False (ex. consommation, où moins consommer
    est préférable) -> hausse rouge.

    `key` : à fournir explicitement si plusieurs cartes de la même page
    partagent le même `label` (la clé du conteneur Streamlit, sinon dérivée
    du label, doit rester unique par page).
    """
    with st.container(border=True, key=f"kpi-{key or _slug(label)}"):
        st.markdown(
            f"""
            <div style="font-size:0.78rem; color:var(--faint); text-transform:uppercase;
                        letter-spacing:0.3px; min-height:2.2em; display:flex; align-items:center;">
                {label}
            </div>
            """,
            unsafe_allow_html=True,
        )

        if delta_pct is None:
            st.markdown(
                f"""
                <div style="font-size:1.4rem; font-weight:700; color:var(--accent); margin-top:2px; white-space:nowrap;">{value}</div>
                <div style="font-size:0.76rem; color:var(--faint); margin-top:2px;">{sub}&nbsp;</div>
                """,
                unsafe_allow_html=True,
            )
            return

        is_improving = delta_pct >= 0 if higher_is_better else delta_pct <= 0
        color = _GOOD if is_improving else _BAD
        arrow = "↑" if delta_pct >= 0 else "↓"
        st.markdown(
            f"""
            <div style="display:flex; align-items:baseline; gap:8px; margin-top:2px;">
                <span style="font-size:1.4rem; font-weight:700; color:var(--ink); white-space:nowrap;">{value}</span>
                <span style="font-size:0.76rem; font-weight:700; color:{color}; background:{_hex_to_rgba(color, 0.12)};
                            border-radius:999px; padding:2px 8px; white-space:nowrap;">{arrow} {abs(delta_pct):.1f}%</span>
            </div>
            <div style="font-size:0.72rem; color:var(--faint); margin-top:2px;">{sub}&nbsp;</div>
            """,
            unsafe_allow_html=True,
        )
