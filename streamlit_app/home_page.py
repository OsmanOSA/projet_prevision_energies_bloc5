import streamlit as st


def home():
    """Page d'accueil : présentation de l'application et guide de navigation."""

    # --- Titre + sous-titre (centrés sur cette page uniquement) ---
    st.markdown(
        """
        <style>
        .home-hero .main-title { text-align: center !important; }
        .home-hero .main-title::after { left: 50% !important; transform: translateX(-50%); }
        .home-hero .subheader { text-align: center !important; }
        </style>
        <div class="home-hero">
            <h1 class='main-title'> Prévision et Pilotage Energétique (PPE)</h1>
            <p class='subheader'>Prévision et analyse du système électrique français</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Carte d'introduction ---
    st.markdown(
        """
        <div class="card">
            <p><em>Un outil dédié à la prévision des différentes sources d'énergie
            et de la consommation électrique.</em></p>
            <p><strong>Développé par :</strong> Osman SAID ALI</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Deux colonnes : à propos + mode d'emploi ---
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <div class="card">
                <h4 style="color: var(--main-title-color);">À propos de l'application</h4>
                <p>L'objectif est de prédire et d'analyser les différentes sources
                d'énergie et la consommation sur le réseau électrique. Vous pouvez
                ainsi estimer les puissances par source et la consommation à
                différents horizons de prévision.</p>
                <p>L'interface aide à anticiper les déséquilibres offre/demande et
                à éclairer l'analyse exploratoire, avec un suivi continu
                de la qualité des prévisions.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="card">
                <h4 style="color: var(--main-title-color);">Comment utiliser l'application</h4>
                <ol>
                    <li>Choisissez une page dans le <strong>menu latéral</strong>
                    (ou via les boutons <em>Précédent / Suivant</em>).</li>
                    <li>Configurez une <strong>période</strong> dans le menu latéral.</li>
                    <li>Explorez les <strong>analyses</strong> et les <strong>prévisions</strong>,
                    puis suivez la <strong>qualité du modèle</strong>.</li>
                    <li>Surveillez les <strong>pipelines</strong> et la fraîcheur des données.</li>
                </ol>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --- Détail des pages ---
    st.markdown(
        """
        <div class="card">
            <h4 style="color: var(--main-title-color);">Les pages en un coup d'œil</h4>
            <ul>
                <li><strong>Accueil :</strong> vous êtes ici — présentation et prise en main.</li>
                <li><strong>Vue d'ensemble :</strong> KPI clés, consommation vs production, déficit et mix énergétique.</li>
                <li><strong>Analyse Consommation :</strong> évolution, distribution et saisonnalités (jour, semaine, mois) ; corrélation à la température.</li>
                <li><strong>Analyse Production :</strong> évolution par source, répartition et corrélations météo.</li>
                <li><strong>Prévisions :</strong> prévisions multi-horizon avec intervalles conformes (~95 %).</li>
                <li><strong>Performance modèle :</strong> erreurs prévu vs réalisé (MAE, RMSE, couverture des intervalles).</li>
            </ul>
            <p style="margin-top: 0.5rem; font-size: 0.9em; color: var(--text-secondary-color, inherit);">
                Le suivi des exécutions Airflow et de la fraîcheur des données est disponible
                séparément dans Grafana (accès administrateur).
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.info(
        "Périmètre actuel : données RTE et Meteostat ; quatre filières de "
        "production seulement. Les écarts production–consommation sont partiels "
        "et ne constituent pas un signal opérationnel d'achat."
    )

    # --- Fonctionnalités principales ---
    st.markdown("<h3 class='subheader'>Fonctionnalités principales</h3>", unsafe_allow_html=True)

    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown(
            """
            <div class="card">
                <h4 style="color: var(--main-title-color);">Prévision énergétique</h4>
                <p>Modèle de Machine Learning multi-source, prévisions à plusieurs
                horizons avec intervalles de prévision conformes.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with f2:
        st.markdown(
            """
            <div class="card">
                <h4 style="color: var(--main-title-color);">Analyse &amp; décision</h4>
                <p>Production suivie, écart partiel, saisonnalités et corrélations
                météo pour documenter les tendances.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with f3:
        st.markdown(
            """
            <div class="card">
                <h4 style="color: var(--main-title-color);">Supervision continue</h4>
                <p>KPI temps réel, suivi prévu vs réalisé et exécutions des pipelines
                de collecte et de prévision.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
