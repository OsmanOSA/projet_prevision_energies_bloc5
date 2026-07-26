"""Configuration centralisée des variables de session Streamlit.

L'app est une "multipage" à un seul script long-vivant (cf. app_main.py) :
st.session_state persiste déjà nativement d'une page à l'autre pour un
widget donné, tant qu'il garde la même `key`. Le problème réel ici n'était
pas la persistance en soi, mais le fait que chaque page (Vue d'ensemble,
Analyse Consommation, Analyse Production) définissait sa PROPRE clé
(ov_period/conso_period/prod_period) : changer la période sur une page
n'avait donc aucun effet sur les autres. SessionConfig centralise une seule
clé partagée pour que le choix reste cohérent partout.
"""

import streamlit as st


class SessionConfig:
    """Valeurs de session partagées entre les pages."""

    DEFAULT_VALUES = {
        "period_window": "7 derniers jours",
    }

    WINDOWS_DATA_DISPLAY = ["7 derniers jours", "30 derniers jours"]

    @classmethod
    def initialize_all(cls):
        """Initialise les variables de session absentes (à appeler une fois, tôt)."""
        for key, default_value in cls.DEFAULT_VALUES.items():
            if key not in st.session_state:
                st.session_state[key] = default_value

    @classmethod
    def get(cls, key, default=None):
        """Récupère une valeur de session_state, avec repli sur la valeur par défaut."""
        if key in st.session_state:
            return st.session_state[key]
        return default if default is not None else cls.DEFAULT_VALUES.get(key)

    @classmethod
    def set(cls, key, value):
        st.session_state[key] = value

    @classmethod
    def reset_all(cls):
        """Remet toutes les variables à leur valeur par défaut."""
        for key, default_value in cls.DEFAULT_VALUES.items():
            st.session_state[key] = default_value

    @classmethod
    def get_radio_index(cls, key, options):
        """Index courant pour un widget radio, 0 si valeur absente/inconnue."""
        try:
            return options.index(cls.get(key))
        except (ValueError, TypeError):
            return 0


def create_radio_widget(label, options, session_key, widget_key, **kwargs):
    """Radio Streamlit relié à une clé de session partagée entre les pages."""
    current_index = SessionConfig.get_radio_index(session_key, options)
    choice = st.radio(label, options=options, index=current_index, key=widget_key, **kwargs)
    SessionConfig.set(session_key, choice)
    return choice
