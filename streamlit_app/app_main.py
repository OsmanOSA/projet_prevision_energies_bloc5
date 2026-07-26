import sys
import logging
import streamlit as st
import warnings

from pathlib import Path
from home_page import home
from overview_page import overview
from consumption_page import consumption
from production_page import production
from forecast_page import forecast
from model_page import model_performance
from session_config import SessionConfig

warnings.simplefilter(action="ignore", category=FutureWarning)


#---------------------------------------------------------------------------
# Configuration de la page  streamlit
#---------------------------------------------------------------------------

st.set_page_config(
    page_title="EnergIA — Pilotage énergétique",
    layout="wide",
    initial_sidebar_state="auto")



def resource_path(relative_path):
    """Compatible .py normal et .exe PyInstaller."""
    # Mode PyInstaller
    if hasattr(sys, '_MEIPASS'):
        return Path(sys._MEIPASS) / relative_path
    
    # Mode normal → reprendre ton comportement EXACT
    CURRENT = Path(__file__).resolve()
    ROOT = CURRENT.parent.parent

    return ROOT / relative_path

FILE_PATH = resource_path(".streamlit/styles.css")

#---------------------------------------------------------------------------
# Chargez le css personnalisé pour le style de l'application
#---------------------------------------------------------------------------

def load_custom_css():
    try:
      
      with open(FILE_PATH) as f:
          
          st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    except FileNotFoundError:
        st.error("Le fichier style.css est introuvable. Veuillez vérifier le chemin d'accès.")

load_custom_css()


class MultipageApp:

    def __init__(self):

        self.apps = []

    def add_page(self, title, func):

        self.apps.append({"title": title, "function": func})

    def run(self):
        """ Affiche la page sélectionnée dans la barre latérale et ajoute un pied de page avec navigation"""

        SessionConfig.initialize_all()

        # Initialisation de l'index de page dans la session state si nécessaire
        if "page_index" not in st.session_state:
            st.session_state.page_index = 0

        # Barre latérale pour la navigation entre les pages
        with st.sidebar:
            st.markdown("## Menu de navigation")
            st.markdown("---")

            # Sélection de page via radio buttons
            selected_page = st.radio(
                "Sélectionnez une page :",
                range(len(self.apps)),
                format_func=lambda i: self.apps[i]["title"],
                index=st.session_state.page_index)


            # Si l'utilisateur sélectionne une page différente via le sidebar
            if selected_page != st.session_state.page_index:
                st.session_state.page_index = selected_page
                st.rerun()


            st.markdown("---")

        # Obtenir l'application sélectionnée basée sur l'index de page
        selected_app = self.apps[st.session_state.page_index]

        # Les pannes de base/modèle deviennent un état utilisateur explicite,
        # sans exposer une stack trace ni des informations de connexion.
        try:
            selected_app["function"]()
        except Exception:
            logging.exception("Échec du rendu de la page %s", selected_app["title"])
            st.error("Cette page ne peut pas charger ses données pour le moment.")
            st.info(
                "Vérifiez que PostgreSQL est démarré, que le schéma est initialisé "
                "et que les artefacts du modèle sont disponibles."
            )


        # Création d'un espace vide pour pousser le footer vers le bas
        st.markdown("<div style='flex-grow: 1;'></div>", unsafe_allow_html=True)

        # Pied de page avec navigation
        footer_container = st.container()

        with footer_container:
            # HTML pour le pied de page
            current_page = self.apps[st.session_state.page_index]["title"]
            footer_html = f"""
            <div class="footer">
                <div class="page-indicator">
                    Page {st.session_state.page_index + 1} sur {len(self.apps)} : <strong>{current_page}</strong>
                </div>
                <div class="footer-content">
            """
            st.markdown(footer_html, unsafe_allow_html=True)

            # Boutons de navigation en pied de page
            prev_button, next_button = st.columns(spec=[0.85, 0.15])

            with prev_button:
                st.markdown("<div class='prev-button'>", unsafe_allow_html=True)
                if st.button("Page précédente", key="prev_page_footer", disabled=st.session_state.page_index <= 0):
                    st.session_state.page_index -= 1
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)

            with next_button:
                st.markdown("<div class='next-button'>", unsafe_allow_html=True)
                if st.button("Page suivante", key="next_page_footer", disabled=st.session_state.page_index >= len(self.apps) - 1):
                    st.session_state.page_index += 1
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)

            # Fermer le div du footer
            st.markdown("</div></div>", unsafe_allow_html=True)

app = MultipageApp()
app.add_page("Accueil", home)
app.add_page("Vue d'ensemble", overview)
app.add_page("Analyse Consommation", consumption)
app.add_page("Analyse Production", production)
app.add_page("Prévisions", forecast)
app.add_page("Performance modèle", model_performance)
app.run()
