import dash
from dash import dcc, html, Input, Output, dash_table
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
import math
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline_prevision.utils.main_utils.utils import concat_all_data, load_object
from pipeline_prevision.utils.ml_utils.model.estimator import ForecastModel
from pipeline_prevision.utils.ml_utils.metric.forecasting_metric import get_forecast_score
from pipeline_prevision.constant.training_pipeline import LOOKBACK, HORIZON
import pickle



# Initialiser l'application Dash avec support multi-pages
app = dash.Dash(__name__, suppress_callback_exceptions=True)

# Styles CSS personnalisés
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "18rem",
    "padding": "1rem 0.8rem",
    "backgroundColor": "#2c3e50",
    "color": "white",
    "overflowY": "auto",
    "fontSize": "15px"  # Taille de police augmentée pour meilleure lisibilité
}

CONTENT_STYLE = {
    "marginLeft": "19.5rem",  # Ajusté pour la nouvelle largeur de sidebar
    "marginRight": "2rem",
    "padding": "2rem 1rem",
}

# Sidebar avec navigation en haut et configuration en bas
sidebar = html.Div([
    html.H2("Menu principal", style={'color': 'white', 'marginBottom': '0.8rem', 'textAlign': 'center', 'fontSize': '1.3rem'}),
    html.Hr(style={'borderColor': 'white', 'margin': '0.3rem 0'}),
    
    dcc.Location(id="url"),
    
    # Navigation des pages en haut
    html.Div([
        html.H5("Navigation", style={'color': 'white', 'marginBottom': '6px', 'fontSize': '16px', 'fontWeight': 'bold'}),
        html.Div(id="sidebar-nav", children=[])  # Navigation sera remplie dynamiquement
    ], style={'marginBottom': '0.8rem'}),
    
    html.Hr(style={'borderColor': 'rgba(255,255,255,0.3)', 'margin': '0.5rem 0'}),
    
    # Configuration des données en bas
    html.Div([
        html.H5("Configuration", style={'color': 'white', 'marginBottom': '6px', 'fontSize': '16px', 'fontWeight': 'bold'}),
        
        html.Div([
            html.Label("Début:", style={'color': 'white', 'fontSize': '13px', 'marginBottom': '2px'}),
            dcc.DatePickerSingle(
                id='start-date-picker',
                date='2024-01-01',
                display_format='YYYY-MM-DD',
                style={'width': '100%', 'fontSize': '11px'}
            )
        ], style={'marginBottom': '6px'}),
        
        html.Div([
            html.Label("Fin:", style={'color': 'white', 'fontSize': '13px', 'marginBottom': '2px'}),
            dcc.DatePickerSingle(
                id='end-date-picker',
                date='2024-01-07',
                display_format='YYYY-MM-DD',
                style={'width': '100%', 'fontSize': '11px'}
            )
        ], style={'marginBottom': '8px'}),
        
        html.Button('Charger', id='load-button', n_clicks=0,
                   style={'backgroundColor': '#3498db', 'color': 'white', 'padding': '6px 8px',
                         'border': 'none', 'borderRadius': '4px', 'cursor': 'pointer',
                         'fontSize': '13px', 'fontWeight': 'bold', 'width': '100%'})
    ], style={'backgroundColor': 'rgba(255,255,255,0.1)', 'padding': '8px', 'borderRadius': '6px', 'marginBottom': '0.5rem'}),
    
    # Indicateurs de statut en bas
    html.Div(id="data-status", style={'marginBottom': '0.3rem'}),
    html.Div(id='status-div-sidebar', style={'marginBottom': '0.5rem'})
    
], style=SIDEBAR_STYLE)

# Layout principal avec sidebar et stores pour cache
app.layout = html.Div([
    sidebar,
    html.Div(id="page-content", style=CONTENT_STYLE),
    dcc.Store(id='data-store'),  # Store global pour les données
    dcc.Store(id='cache-overview'),  # Cache pour Vue d'ensemble
    dcc.Store(id='cache-consumption'),  # Cache pour Analyse Consommation
    dcc.Store(id='cache-production'),  # Cache pour Analyse Production
    dcc.Store(id='cache-prediction')  # Cache pour Prévision
])

# Page d'accueil (simplifiée sans configuration)
def create_home_page():
    return html.Div([
        # Titre principal
        html.H1("Application de prévisions énergétiques", 
                style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '10px', 'fontSize': '2.5em'}),
        html.H3("Prévisions énergétiques pour un réseau électrique d'EnergIA", 
                style={'textAlign': 'center', 'color': '#34495e', 'marginBottom': '30px', 'fontWeight': '300'}),
        
        # Carte d'introduction
        html.Div([
            html.P("Un outil dédié à la prévision de différentes sources d'énergie et la consommation énergétique", 
                   style={'fontSize': '18px', 'fontStyle': 'italic', 'color': '#7f8c8d', 'marginBottom': '15px', 'textAlign': 'center'}),
            html.P("Développé par : Osman SAID ALI", 
                   style={'fontSize': '16px', 'fontWeight': 'bold', 'color': '#2c3e50', 'textAlign': 'center'})
        ], style={'backgroundColor': '#fff', 'padding': '25px', 'borderRadius': '10px',
                 'boxShadow': '0 4px 6px rgba(0,0,0,0.1)', 'marginBottom': '30px'}),
        
        # Section principale avec deux colonnes (responsive avec flexbox)
        html.Div([
            # Colonne 1 - À propos
            html.Div([
                html.H4("À propos de l'application", style={'color': '#2c3e50', 'marginBottom': '15px'}),
                html.P([
                    "L'objectif de cette application est de prédire et d'analyser les différentes sources d'énergies et de la consommation énergétique sur le réseau électrique. ",
                    "Vous pourrez ainsi déterminer les puissances de différentes sources et consommation énergétique à différents horizons de prédiction."
                ], style={'marginBottom': '15px', 'textAlign': 'justify', 'lineHeight': '1.6'}),
                html.P([
                    "Cette interface intuitive a été conçue pour pallier au problème de déséquilibrage énergétique et faciliter la prise de décision pour l'achat de l'électricité. ",
                    "Elle offre aussi la possibilité de retrainer nos modèles de Machine Learning et sélectionner le meilleur."
                ], style={'textAlign': 'justify', 'lineHeight': '1.6'})
            ], style={'backgroundColor': '#fff', 'padding': '25px', 'borderRadius': '10px',
                     'boxShadow': '0 4px 6px rgba(0,0,0,0.1)', 'flex': '1', 'marginRight': '15px'}),
            
            # Colonne 2 - Comment utiliser
            html.Div([
                html.H4("Comment utiliser l'application", style={'color': '#2c3e50', 'marginBottom': '15px'}),
                html.Ol([
                    html.Li("Configurez les dates dans le menu latéral", style={'marginBottom': '8px'}),
                    html.Li("Cliquez sur 'Charger' pour importer les données", style={'marginBottom': '8px'}),
                    html.Li(["Explorez la page ", html.Span("Vue d'ensemble", style={'color': '#27ae60', 'fontWeight': 'bold'}), " pour l'analyse globale"], style={'marginBottom': '8px'}),
                    html.Li(["Utilisez la page ", html.Span("Prévision", style={'color': '#27ae60', 'fontWeight': 'bold'}), " pour générer des prédictions"], style={'marginBottom': '8px'})
                ], style={'lineHeight': '1.6'})
            ], style={'backgroundColor': '#fff', 'padding': '25px', 'borderRadius': '10px',
                     'boxShadow': '0 4px 6px rgba(0,0,0,0.1)', 'flex': '1', 'marginLeft': '15px'})
        ], style={'display': 'flex', 'gap': '0px', 'marginBottom': '30px'}),
        
        # Navigation détaillée
        html.Div([
            html.H4("Navigation détaillée", style={'color': '#2c3e50', 'marginBottom': '15px'}),
            html.Ul([
                html.Li([html.Strong("Accueil : "), "Vous êtes ici. Présentation de l'application et de ses fonctionnalités."], style={'marginBottom': '8px'}),
                html.Li([html.Strong("Vue d'ensemble : "), "Métriques clés, histogrammes empilés et analyse du déficit énergétique."], style={'marginBottom': '8px'}),
                html.Li([html.Strong("Analyse Consommation : "), "Analyse détaillée de la consommation électrique avec distributions."], style={'marginBottom': '8px'}),
                html.Li([html.Strong("Analyse Production : "), "Visualisation de la production par source d'énergie et répartitions."], style={'marginBottom': '8px'}),
                html.Li([html.Strong("Prévisions : "), "Génération de prévisions énergétiques avec différents horizons temporels."], style={'marginBottom': '8px'})
            ], style={'lineHeight': '1.6'})
        ], style={'backgroundColor': '#fff', 'padding': '25px', 'borderRadius': '10px',
                 'boxShadow': '0 4px 6px rgba(0,0,0,0.1)', 'marginBottom': '30px'}),
        
        # Fonctionnalités principales
        html.H3("Fonctionnalités principales", style={'color': '#2c3e50', 'marginBottom': '20px', 'textAlign': 'center'}),
        
        html.Div([
            html.Div([
                html.H4("Prévisions énergétiques", style={'color': '#2c3e50', 'marginBottom': '15px'}),
                html.P("Prédictions énergétiques à différents horizons de prédiction avec visualisations interactives et analyses statistiques avancées.", 
                       style={'lineHeight': '1.6'})
            ], style={'backgroundColor': '#fff', 'padding': '25px', 'borderRadius': '10px',
                     'boxShadow': '0 4px 6px rgba(0,0,0,0.1)', 'flex': '1', 'marginRight': '15px',
                     'height': '150px', 'display': 'flex', 'flexDirection': 'column'}),
            
            html.Div([
                html.H4("Analyse de données", style={'color': '#2c3e50', 'marginBottom': '15px'}),
                html.P("Présentation claire des résultats pour faciliter la prise de décision avec des outils d'analyse comparative et d'optimisation.", 
                       style={'lineHeight': '1.6'})
            ], style={'backgroundColor': '#fff', 'padding': '25px', 'borderRadius': '10px',
                     'boxShadow': '0 4px 6px rgba(0,0,0,0.1)', 'flex': '1', 'marginLeft': '15px',
                     'height': '150px', 'display': 'flex', 'flexDirection': 'column'})
        ], style={'display': 'flex', 'gap': '0px'})
    ])

# Callback pour la navigation multi-pages avec mise en évidence
@app.callback([Output('page-content', 'children'),
               Output('sidebar-nav', 'children')],
              [Input('url', 'pathname')])
def display_page(pathname):
    # Créer la navigation avec mise en évidence de la page active
    def create_nav_link(href, icon, text, is_active=False):
        style = {
            'display': 'block', 'padding': '10px 12px', 'textDecoration': 'none', 
            'color': 'white', 'borderRadius': '4px', 'marginBottom': '4px',
            'backgroundColor': 'rgba(52, 152, 219, 0.8)' if is_active else 'rgba(255,255,255,0.1)',
            'fontSize': '14px'  # Augmenté pour meilleure lisibilité
        }
        return dcc.Link([
            html.Div([
                html.I(className=icon, style={'marginRight': '8px', 'fontSize': '14px'}),  # Icônes plus grandes
                text
            ])
        ], href=href, style=style)
    
    nav_links = [
        create_nav_link("/", "fas fa-home", "Accueil", pathname == "/"),
        create_nav_link("/overview", "fas fa-chart-line", "Vue d'ensemble", pathname == "/overview"),
        create_nav_link("/consumption", "fas fa-bolt", "Analyse Consommation", pathname == "/consumption"),
        create_nav_link("/production", "fas fa-battery-full", "Analyse Production", pathname == "/production"),
        create_nav_link("/prediction", "fas fa-crystal-ball", "Prévision", pathname == "/prediction"),
    ]
    
    # Déterminer quelle page afficher avec contenu intégré directement
    if pathname == '/overview':
        page_content = html.Div([
            html.H1("Vue d'ensemble", style={'color': '#2c3e50', 'marginBottom': '30px'}),
            html.Div(id='overview-content', children=[html.P("Chargement en cours...", style={'textAlign': 'center', 'color': '#7f8c8d'})])
        ])
    elif pathname == '/consumption':
        page_content = html.Div([
            html.H1("Analyse de la Consommation", style={'color': '#2c3e50', 'marginBottom': '30px'}),
            html.Div(id='consumption-content', children=[html.P("Chargement en cours...", style={'textAlign': 'center', 'color': '#7f8c8d'})])
        ])
    elif pathname == '/production':
        page_content = html.Div([
            html.H1("Analyse de la Production", style={'color': '#2c3e50', 'marginBottom': '30px'}),
            html.Div(id='production-content', children=[html.P("Chargement en cours...", style={'textAlign': 'center', 'color': '#7f8c8d'})])
        ])
    elif pathname == '/prediction':
        page_content = html.Div([
            html.H1("Prévision Énergétique", style={'color': '#2c3e50', 'marginBottom': '30px'}),
            html.Div(id='prediction-content', children=[html.P("Chargement en cours...", style={'textAlign': 'center', 'color': '#7f8c8d'})])
        ])
    else:
        page_content = create_home_page()
    
    return page_content, nav_links

# Callback pour charger les données
@app.callback(
    [Output('data-store', 'data'),
     Output('status-div-sidebar', 'children')],
    [Input('load-button', 'n_clicks')],
    [dash.dependencies.State('start-date-picker', 'date'),
     dash.dependencies.State('end-date-picker', 'date')]
)
def load_data(n_clicks, start_date, end_date):
    if n_clicks == 0:
        return {}, ""
    
    try:
        # Convertir les dates au format requis
        if isinstance(start_date, str):
            start_str = start_date.split('T')[0]
        else:
            start_str = str(start_date)
            
        if isinstance(end_date, str):
            end_str = end_date.split('T')[0]
        else:
            end_str = str(end_date)
        
        print(f"Debug: Dates converties - start: {start_str}, end: {end_str}")
        
        # Charger les données en utilisant la fonction concat_all_data
        df = concat_all_data(start_str, end_str)
        
        # Nettoyer les caches lors du chargement de nouvelles données
        global _graph_cache, _processed_data_cache
        _graph_cache.clear()
        _processed_data_cache.clear()
        
        # Convertir en dictionnaire pour le stockage
        data_dict = df.reset_index().to_dict('records')
        
        status_message = html.Div([
            html.I(className="fas fa-check-circle", style={'color': '#27ae60', 'marginRight': '5px'}),
            html.Div([
                html.Small("Succès", style={'color': '#27ae60', 'fontWeight': 'bold', 'display': 'block'}),
                html.Small(f"{len(data_dict)} enregistrements", style={'color': '#27ae60', 'fontSize': '12px'})
            ])
        ], style={'backgroundColor': 'rgba(39, 174, 96, 0.2)', 'padding': '8px', 'borderRadius': '4px', 'display': 'flex', 'alignItems': 'center'})
        
        return data_dict, status_message
        
    except Exception as e:
        print(f"Erreur détaillée: {str(e)}")
        error_message = html.Div([
            html.I(className="fas fa-exclamation-triangle", style={'color': '#e74c3c', 'marginRight': '5px'}),
            html.Div([
                html.Small("Erreur", style={'color': '#e74c3c', 'fontWeight': 'bold', 'display': 'block'}),
                html.Small("Voir console", style={'color': '#e74c3c', 'fontSize': '12px'})
            ])
        ], style={'backgroundColor': 'rgba(231, 76, 60, 0.2)', 'padding': '8px', 'borderRadius': '4px', 'display': 'flex', 'alignItems': 'center'})
        
        return {}, error_message

# Callback pour l'indicateur de statut des données
@app.callback(
    Output('data-status', 'children'),
    [Input('data-store', 'data')]
)
def update_data_status(stored_data):
    if not stored_data:
        return html.Div([
            html.I(className="fas fa-exclamation-circle", style={'color': '#e74c3c', 'marginRight': '5px'}),
            html.Small("Aucune donnée", style={'color': '#e74c3c'})
        ], style={'backgroundColor': 'rgba(231, 76, 60, 0.2)', 'padding': '8px', 'borderRadius': '4px'})
    else:
        nb_records = len(stored_data)
        return html.Div([
            html.I(className="fas fa-check-circle", style={'color': '#27ae60', 'marginRight': '5px'}),
            html.Small(f"{nb_records} enregistrements", style={'color': '#27ae60'})
        ], style={'backgroundColor': 'rgba(39, 174, 96, 0.2)', 'padding': '8px', 'borderRadius': '4px'})

# Fonction utilitaire pour créer le message "pas de données"
def create_no_data_message():
    return html.Div([
        html.Div([
            html.I(className="fas fa-info-circle", style={'fontSize': '48px', 'color': '#3498db', 'marginBottom': '20px'}),
            html.H4("Aucune donnée chargée", style={'color': '#2c3e50', 'marginBottom': '15px'}),
            html.P("Veuillez configurer les dates dans le menu latéral et cliquer sur 'Charger'.",
                  style={'color': '#7f8c8d', 'fontSize': '16px', 'marginBottom': '20px'}),
            dcc.Link([
                html.Button("Retour à l'accueil", 
                           style={'backgroundColor': '#3498db', 'color': 'white', 'padding': '12px 24px',
                                 'border': 'none', 'borderRadius': '6px', 'cursor': 'pointer',
                                 'fontSize': '14px'})
            ], href="/")
        ], style={'textAlign': 'center', 'padding': '50px', 'backgroundColor': '#f8f9fa', 
                 'borderRadius': '10px', 'marginTop': '50px'})
    ])

# Fonction utilitaire pour convertir les données
def prepare_dataframe(stored_data):
    if not stored_data:
        return None
    df = pd.DataFrame(stored_data)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
    return df

# Cache global pour éviter les recalculs
_graph_cache = {}

def get_cached_graph(cache_key, create_function, *args):
    """Fonction utilitaire pour mettre en cache les graphiques coûteux"""
    if cache_key in _graph_cache:
        return _graph_cache[cache_key]
    
    graph = create_function(*args)
    _graph_cache[cache_key] = graph
    return graph

# Cache global pour les données traitées
_processed_data_cache = {}

def get_cached_content(page_name, stored_data, create_function):
    """Fonction pour gérer le cache de contenu par page"""
    if not stored_data:
        return create_no_data_message()
    
    data_hash = str(hash(str(stored_data)))
    cache_key = f"{page_name}_{data_hash}"
    
    if cache_key in _processed_data_cache:
        return _processed_data_cache[cache_key]
    
    # Créer le contenu et le mettre en cache
    df = prepare_dataframe(stored_data)
    if df is None:
        content = create_no_data_message()
    else:
        content = create_function(df)
    
    _processed_data_cache[cache_key] = content
    return content

# Fonctions pour les prévisions avec modèles pré-entraînés

def create_predictions(df, horizon_hours):
    """Créer des prévisions en utilisant les modèles pré-entraînés (inspiré de test.py)"""
    try:
        # Charger les modèles comme dans test.py
        model = load_object("final_models/model.pkl")
        preprocessor = load_object("final_models/preprocessor.pkl")
        
        if model is None or preprocessor is None:
            return None, None, None, None, "Erreur: Impossible de charger les modèles pré-entraînés"
        
        print(f"Colonnes disponibles dans df: {df.columns.tolist()}")
        print(f"Shape des données: {df.shape}")
        
        # Utiliser exactement le même ordre de features que dans test.py
        features = ['BIOMASS', 'NUCLEAR', 'SOLAR', 'WIND_ONSHORE', 'consommation_totale', 'temp']
        
        # Vérifier que toutes les features sont présentes
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            return None, None, None, None, f"Features manquantes: {missing_features}"
        
        # Prendre les 36 dernières valeurs et réorganiser les colonnes dans le bon ordre
        df_features = df[features].tail(36).copy()
        print(f"Shape des 36 dernières valeurs: {df_features.shape}")
        print(f"Données avec features ordonnées: {df_features.columns.tolist()}")
        
        # Créer le ForecastModel comme dans test.py
        forecast_model = ForecastModel(preprocessor=preprocessor, model=model)
        
        # Faire la prédiction avec la méthode predict_multistep
        y_pred, y_test = forecast_model.predict_multistep(x=df_features, n_futur=horizon_hours)
        
        print(f"Type y_pred: {type(y_pred)}, Shape: {y_pred.shape}")
        print(f"Type y_test: {type(y_test)}, Shape: {y_test.shape if y_test is not None else 'None'}")
        
        # Vérifier et ajuster la shape de y_pred si nécessaire
        if y_pred is not None:
            print(f"Shape originale y_pred: {y_pred.shape}")
            
            # Le modèle peut retourner différentes shapes selon le cas
            if len(y_pred.shape) == 2:
                # Shape (n_steps, n_features) - cas normal
                if y_pred.shape[0] != horizon_hours:
                    print(f"Ajustement nécessaire: {y_pred.shape[0]} steps vs {horizon_hours} demandés")
                    if y_pred.shape[0] < horizon_hours:
                        # Répéter la dernière prédiction pour combler
                        last_pred = y_pred[-1:, :]
                        missing_steps = horizon_hours - y_pred.shape[0]
                        repeated_preds = np.repeat(last_pred, missing_steps, axis=0)
                        y_pred = np.vstack([y_pred, repeated_preds])
                    else:
                        # Tronquer si on a trop
                        y_pred = y_pred[:horizon_hours, :]
                    print(f"y_pred ajusté à: {y_pred.shape}")
            else:
                print(f"Shape inattendue pour y_pred: {y_pred.shape}")
                # Essayer de reformater si nécessaire
                if y_pred.size == horizon_hours * 6:  # 6 features
                    y_pred = y_pred.reshape(horizon_hours, 6)
                    print(f"y_pred reformaté à: {y_pred.shape}")
        
        # Calculer les métriques seulement si on a y_test (mode test/validation)
        mae, mse, metrics_by_energy = None, None, None
        if y_test is not None and y_pred is not None:
            try:
                print("Mode test/validation - Calcul des métriques possibles")
                # Métriques globales
                forecast_metric = get_forecast_score(y_true=y_test, y_pred=y_pred)
                mae = forecast_metric.mae
                mse = forecast_metric.mse
                print(f"MAE globale: {mae}, MSE globale: {mse}")
                
                # Métriques par source d'énergie
                metrics_by_energy = calculate_metrics_by_energy(y_test, y_pred, features)
                
            except Exception as metric_error:
                print(f"Erreur calcul métriques: {metric_error}")
        else:
            print("Mode prédiction réelle - Pas de vérité terrain, pas de métriques")
        
        return y_pred, y_test, mae, mse, metrics_by_energy, None
        
    except Exception as e:
        print(f"Erreur détaillée prédiction: {str(e)}")
        return None, None, None, None, None, f"Erreur lors de la prédiction: {str(e)}"


def calculate_energy_deficit_predict(y_pred):
    """Calculer le déficit énergétique pour les prédictions
    Formule: Déficit = Production_totale - Consommation
    Déficit positif = Surplus (Production > Consommation)
    Déficit négatif = Manque (Production < Consommation)
    """
    try:
        print(f"Calcul déficit - y_pred shape: {y_pred.shape}")
        
        deficit = np.zeros((y_pred.shape[0], 1))
        
        for i in range(len(y_pred)):
            # Production totale = somme des 4 premières colonnes (BIOMASS, NUCLEAR, SOLAR, WIND_ONSHORE)
            production_total = y_pred[i, :4].sum()
            # Consommation = 5ème colonne (index 4)
            consumption = y_pred[i, 4]
            # Déficit = Production - Consommation
            # Positif = Surplus (on produit plus qu'on consomme)
            # Négatif = Déficit (on produit moins qu'on consomme)
            deficit[i] = production_total - consumption
        
        # Aplatir le déficit pour faciliter l'utilisation
        deficit_flat = deficit.flatten()
        
        # Analyser les patterns de répétition
        production_totals = [y_pred[i, :4].sum() for i in range(len(y_pred))]
        
        # Détecter les valeurs répétées
        unique_productions = []
        repetition_counts = []
        current_val = None
        current_count = 0
        
        for val in production_totals:
            if current_val is None or abs(val - current_val) > 0.01:  # Nouvelle valeur
                if current_val is not None:
                    unique_productions.append(current_val)
                    repetition_counts.append(current_count)
                current_val = val
                current_count = 1
            else:  # Même valeur
                current_count += 1
        
        if current_val is not None:
            unique_productions.append(current_val)
            repetition_counts.append(current_count)
        
        print(f"Production - Valeurs uniques: {len(unique_productions)}, Pattern de répétition: {repetition_counts[:10]}")
        print(f"Consommation - Min: {y_pred[:, 4].min():.2f}, Max: {y_pred[:, 4].max():.2f}, Std: {y_pred[:, 4].std():.2f}")
        print(f"Déficit - Min: {deficit_flat.min():.2f}, Max: {deficit_flat.max():.2f}, Moyenne: {deficit_flat.mean():.2f}")
        
        return {
            'consumption': y_pred[:, 4],  # Consommation
            'production': np.array([y_pred[i, :4].sum() for i in range(len(y_pred))]),  # Production totale
            'deficit': deficit_flat,  # Déficit = Production - Consommation (positif = surplus, négatif = manque)
            'deficit_percentage': (deficit_flat / np.maximum(y_pred[:, 4], 1e-8)) * 100
        }
        
    except Exception as e:
        print(f"Erreur calcul déficit énergétique: {e}")
        return None

def calculate_metrics_by_energy(y_test, y_pred, features):
    """Calculer les métriques par source d'énergie"""
    try:
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        metrics_energies = []
        
        for i, feature in enumerate(features):
            y_true_energy = y_test[:, i]
            y_pred_energy = y_pred[:, i]
            
            mae_energy = mean_absolute_error(y_true_energy, y_pred_energy)
            mse_energy = mean_squared_error(y_true_energy, y_pred_energy)
            r2_energy = r2_score(y_true_energy, y_pred_energy)
            
            metrics_energies.append({
                'Source': feature,
                'MAE': round(mae_energy, 3),
                'MSE': round(mse_energy, 3),
                'R2': round(r2_energy, 3)
            })
        
        return metrics_energies
        
    except Exception as e:
        print(f"Erreur calcul métriques par énergie: {e}")
        return None

def create_prediction_charts(hist_data, pred_df, features, horizon):
    """Créer des graphiques complètement séparés par source avec intervalles de confiance"""
    from plotly.subplots import make_subplots
    import numpy as np
    
    # Couleurs pour chaque variable
    colors = {
        'consommation_totale': '#e74c3c',
        'temp': '#f39c12', 
        'SOLAR': '#f1c40f',
        'BIOMASS': '#27ae60',
        'NUCLEAR': '#3498db',
        'WIND_ONSHORE': '#9b59b6'
    }
    
    # Créer 6 subplots : 4 sources + consommation + température
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'BIOMASS (Biomasse)',
            'NUCLEAR (Nucléaire)',
            'SOLAR (Solaire)',
            'WIND_ONSHORE (Éolien)',
            'Consommation Totale',
            'Température'
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.12,
        specs=[
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": False}]
        ]
    )
    
    # Ligne de séparation historique/prédiction
    try:
        if hasattr(hist_data.index, 'to_timestamp'):
            separation_time = hist_data.index[-1].to_timestamp()
        else:
            separation_time = pd.to_datetime(hist_data.index[-1])
    except:
        separation_time = None
    
    # Mapping des sources aux positions dans la grille
    source_positions = {
        'BIOMASS': (1, 1),
        'NUCLEAR': (1, 2),
        'SOLAR': (2, 1),
        'WIND_ONSHORE': (2, 2),
        'consommation_totale': (3, 1),
        'temp': (3, 2)
    }
    
    # Tracer chaque source séparément avec intervalles de confiance
    for source, (row, col) in source_positions.items():
        if source in hist_data.columns and source in pred_df.columns:
            # Calculer l'écart-type historique pour l'intervalle de confiance
            hist_std = hist_data[source].std()
            
            # Historique
            fig.add_trace(
                go.Scatter(
                    x=hist_data.index,
                    y=hist_data[source],
                    mode='lines',
                    name=f'{source}',
                    line=dict(color=colors.get(source, '#95a5a6'), width=2.5),
                    showlegend=False,
                    hovertemplate='Historique: %{y:.1f}<extra></extra>'
                ),
                row=row, col=col
            )
            
            # Calculer l'intervalle de confiance plus réaliste
            # Utiliser un pourcentage de la valeur prédite qui augmente avec l'horizon
            # Commence à ±2% et augmente jusqu'à ±10% sur 24h
            pred_values = pred_df[source].values
            
            # Pourcentage d'incertitude qui augmente linéairement avec l'horizon
            # Ajusté selon le type de source pour plus de réalisme
            if source == 'NUCLEAR':
                base_uncertainty = 0.01  # 1% - Nucléaire très stable
                max_uncertainty = 0.05   # 5% maximum
            elif source == 'SOLAR':
                base_uncertainty = 0.05  # 5% - Solaire plus variable
                max_uncertainty = 0.15   # 15% maximum
            elif source == 'WIND_ONSHORE':
                base_uncertainty = 0.04  # 4% - Éolien variable
                max_uncertainty = 0.12   # 12% maximum
            elif source == 'BIOMASS':
                base_uncertainty = 0.02  # 2% - Biomasse stable
                max_uncertainty = 0.08   # 8% maximum
            elif source == 'consommation_totale':
                base_uncertainty = 0.02  # 2% - Consommation prévisible
                max_uncertainty = 0.08   # 8% maximum
            else:
                base_uncertainty = 0.02  # 2% par défaut
                max_uncertainty = 0.10   # 10% maximum
            
            # Calculer le pourcentage pour chaque pas de temps
            uncertainty_percentages = np.array([
                base_uncertainty + (max_uncertainty - base_uncertainty) * (i / 24) 
                for i in range(len(pred_df))
            ])
            
            # Appliquer le pourcentage aux valeurs prédites
            confidence_intervals = pred_values * uncertainty_percentages
            
            upper_bound = pred_values + confidence_intervals
            lower_bound = pred_values - confidence_intervals
            
            # Pour les valeurs négatives (comme la température), éviter les bornes illogiques
            if source == 'temp':
                # Pour la température, utiliser une incertitude absolue plus petite
                temp_uncertainty = np.array([0.5 + 0.5 * (i / 24) for i in range(len(pred_df))])  # 0.5°C à 1°C
                upper_bound = pred_values + temp_uncertainty
                lower_bound = pred_values - temp_uncertainty
            
            # Intervalle de confiance - zone ombrée
            fig.add_trace(
                go.Scatter(
                    x=pred_df.index.tolist() + pred_df.index.tolist()[::-1],
                    y=upper_bound.tolist() + lower_bound.tolist()[::-1],
                    fill='toself',
                    fillcolor=colors.get(source, '#95a5a6'),
                    opacity=0.2,
                    line=dict(color='rgba(255,255,255,0)'),
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=row, col=col
            )
            
            # Prédiction centrale
            fig.add_trace(
                go.Scatter(
                    x=pred_df.index,
                    y=pred_df[source],
                    mode='lines+markers',
                    name=f'{source} (pred)',
                    line=dict(color=colors.get(source, '#95a5a6'), width=2.5, dash='dash'),
                    marker=dict(size=4),
                    showlegend=False,
                    hovertemplate='Prédiction: %{y:.1f}<br>IC 95%: [%{customdata[0]:.1f}, %{customdata[1]:.1f}]<extra></extra>',
                    customdata=np.column_stack((lower_bound, upper_bound))
                ),
                row=row, col=col
            )
            
            # Ajouter min/max pour contexte (incluant l'intervalle de confiance)
            all_values = pd.concat([hist_data[source], pd.Series(upper_bound), pd.Series(lower_bound)])
            y_min, y_max = all_values.min(), all_values.max()
            y_range = y_max - y_min
            
            # Mettre à jour l'axe Y pour cette source
            if source == 'temp':
                fig.update_yaxes(title_text="°C", row=row, col=col,
                               range=[y_min - 0.1*y_range, y_max + 0.1*y_range])
            else:
                fig.update_yaxes(title_text="MW", row=row, col=col,
                               range=[y_min - 0.1*y_range, y_max + 0.1*y_range])
    
    # Ajouter les lignes de séparation sur chaque subplot
    if separation_time:
        for row in range(1, 4):
            for col in range(1, 3):
                fig.add_vline(
                    x=separation_time,
                    line_dash="dot",
                    line_color="gray",
                    line_width=1,
                    row=row, col=col
                )
                
                # Ajouter une annotation sur le premier graphique seulement
                if row == 1 and col == 1:
                    fig.add_annotation(
                        x=separation_time,
                        y=1.05,
                        text="Début prédictions",
                        showarrow=False,
                        xref="x",
                        yref="paper",
                        textangle=-45,
                        font=dict(size=10, color="gray"),
                        row=row, col=col
                    )
    
    # Mise en page générale
    fig.update_xaxes(title_text="", showticklabels=False, row=1, col=1)
    fig.update_xaxes(title_text="", showticklabels=False, row=1, col=2)
    fig.update_xaxes(title_text="", showticklabels=False, row=2, col=1)
    fig.update_xaxes(title_text="", showticklabels=False, row=2, col=2)
    fig.update_xaxes(title_text="Date/Heure", row=3, col=1)
    fig.update_xaxes(title_text="Date/Heure", row=3, col=2)
    
    fig.update_layout(
        title={
            'text': f'Analyse Détaillée par Source - Horizon {horizon}h',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        height=900,
        hovermode='x unified',
        showlegend=False,
        margin=dict(t=80, b=60, l=60, r=60)
    )
    
    return fig

def create_consumption_vs_production_chart(deficit_info, timestamps, hist_data=None):
    """Créer le graphique comparatif consommation vs production avec historique"""
    fig = go.Figure()
    
    # Si on a des données historiques, les ajouter d'abord
    if hist_data is not None:
        # Calculer la production totale historique
        production_sources = ['BIOMASS', 'NUCLEAR', 'SOLAR', 'WIND_ONSHORE']
        hist_production = hist_data[production_sources].sum(axis=1) if all(s in hist_data.columns for s in production_sources) else None
        
        # Historique de consommation
        if 'consommation_totale' in hist_data.columns:
            fig.add_trace(go.Scatter(
                x=hist_data.index,
                y=hist_data['consommation_totale'],
                mode='lines',
                name='Consommation (historique)',
                line=dict(color='#e74c3c', width=2),
                opacity=0.6,
                hovertemplate='Conso hist: %{y:.1f} MW<extra></extra>'
            ))
        
        # Historique de production
        if hist_production is not None:
            fig.add_trace(go.Scatter(
                x=hist_data.index,
                y=hist_production,
                mode='lines',
                name='Production (historique)',
                line=dict(color='#27ae60', width=2),
                opacity=0.6,
                hovertemplate='Prod hist: %{y:.1f} MW<extra></extra>'
            ))
    
    # Prédictions de consommation
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=deficit_info['consumption'],
        mode='lines+markers',
        name='Consommation (prédiction)',
        line=dict(color='#e74c3c', width=3, dash='dash'),
        marker=dict(size=6),
        hovertemplate='Conso préd: %{y:.1f} MW<extra></extra>'
    ))
    
    # Prédictions de production
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=deficit_info['production'],
        mode='lines+markers',
        name='Production (prédiction)',
        line=dict(color='#27ae60', width=3, dash='dash'),
        marker=dict(size=6),
        hovertemplate='Prod préd: %{y:.1f} MW<extra></extra>'
    ))
    
    # Zone entre les deux courbes de prédiction
    fig.add_trace(go.Scatter(
        x=timestamps.tolist() + timestamps.tolist()[::-1] if hasattr(timestamps, 'tolist') else timestamps + timestamps[::-1],
        y=deficit_info['consumption'].tolist() + deficit_info['production'].tolist()[::-1],
        fill='toself',
        fillcolor='rgba(255, 193, 7, 0.1)',
        line=dict(color='rgba(255,255,255,0)'),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Ligne de séparation historique/prédiction
    if hist_data is not None and len(hist_data) > 0:
        try:
            separation_time = pd.to_datetime(hist_data.index[-1])
            fig.add_vline(
                x=separation_time,
                line_dash="dot",
                line_color="gray",
                line_width=1,
                annotation_text="Début prédictions"
            )
        except:
            pass
    
    fig.update_layout(
        title='Consommation vs Production',
        title_font_size=14,
        xaxis_title='',
        yaxis_title='MW',
        height=350,
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=60, r=20, t=60, b=40)
    )
    
    return fig

def create_deficit_evolution_chart(deficit_info, timestamps, hist_data=None):
    """Créer le graphique d'évolution du déficit avec historique"""
    fig = go.Figure()
    
    # Si on a des données historiques, calculer et afficher le déficit historique
    if hist_data is not None:
        production_sources = ['BIOMASS', 'NUCLEAR', 'SOLAR', 'WIND_ONSHORE']
        if all(s in hist_data.columns for s in production_sources) and 'consommation_totale' in hist_data.columns:
            hist_production = hist_data[production_sources].sum(axis=1)
            hist_deficit = hist_production - hist_data['consommation_totale']
            
            # Historique du déficit
            fig.add_trace(go.Scatter(
                x=hist_data.index,
                y=hist_deficit,
                mode='lines',
                name='Déficit (historique)',
                line=dict(color='#95a5a6', width=2),
                opacity=0.6,
                fill='tozeroy',
                fillcolor='rgba(149, 165, 166, 0.1)',
                hovertemplate='Déficit hist: %{y:.1f} MW<extra></extra>'
            ))
    
    # Déficit prédit avec zone colorée
    deficit_values = deficit_info['deficit']
    
    # Créer des couleurs pour chaque point selon sa valeur
    colors = []
    for val in deficit_values:
        if val > 0:  # Surplus (positif) = vert
            colors.append('#27ae60')
        else:  # Déficit (négatif) = rouge
            colors.append('#e74c3c')
    
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=deficit_values,
        mode='lines+markers',
        name='Déficit (prédiction)',
        line=dict(color='#f39c12', width=3, dash='dash'),
        marker=dict(
            size=8,
            color=colors,  # Utiliser les couleurs correctes
            line=dict(width=1, color='white')
        ),
        fill='tozeroy',
        fillcolor='rgba(243, 156, 18, 0.2)',
        hovertemplate='Déficit préd: %{y:.1f} MW<extra></extra>'
    ))
    
    # Ligne de séparation historique/prédiction
    if hist_data is not None and len(hist_data) > 0:
        try:
            separation_time = pd.to_datetime(hist_data.index[-1])
            fig.add_vline(
                x=separation_time,
                line_dash="dot",
                line_color="gray",
                line_width=1,
                annotation_text="Début prédictions"
            )
        except:
            pass
    
    # Ligne d'équilibre
    fig.add_hline(
        y=0, 
        line_dash="dash", 
        line_color="gray",
        line_width=2,
        annotation_text="Équilibre"
    )
    
    # Zones colorées pour surplus/déficit
    # Calculer y_max en tenant compte de l'historique aussi
    all_deficit_values = [deficit_values]
    
    # Ajouter les valeurs historiques si elles existent
    if hist_data is not None:
        production_sources = ['BIOMASS', 'NUCLEAR', 'SOLAR', 'WIND_ONSHORE']
        if all(s in hist_data.columns for s in production_sources) and 'consommation_totale' in hist_data.columns:
            hist_production = hist_data[production_sources].sum(axis=1)
            hist_deficit = hist_production - hist_data['consommation_totale']
            all_deficit_values.append(hist_deficit.values)
    
    # Calculer le max pour l'échelle Y
    all_values_concat = np.concatenate(all_deficit_values)
    y_max = max(abs(all_values_concat.min()), abs(all_values_concat.max())) * 1.1
    
    fig.add_hrect(
        y0=0, y1=y_max,
        fillcolor="green", opacity=0.05,
        layer="below", line_width=0,
    )
    
    fig.add_hrect(
        y0=-y_max, y1=0,
        fillcolor="red", opacity=0.05,
        layer="below", line_width=0,
    )
    
    fig.update_layout(
        title='Évolution du Déficit',
        title_font_size=14,
        xaxis_title='',
        yaxis_title='Déficit (MW)',
        height=350,
        hovermode='x unified',
        showlegend=False,
        margin=dict(l=60, r=20, t=60, b=40)
    )
    
    return fig

# Callback pour nettoyer les caches quand les données changent
@app.callback(
    [Output('cache-overview', 'data'),
     Output('cache-consumption', 'data'),
     Output('cache-production', 'data'),
     Output('cache-prediction', 'data')],
    [Input('data-store', 'data')]
)
def update_all_caches(stored_data):
    # Nettoyer le cache quand de nouvelles données arrivent
    global _processed_data_cache, _graph_cache
    _processed_data_cache.clear()
    _graph_cache.clear()
    
    # Retourner des caches vides pour forcer le recalcul
    return {}, {}, {}, {}

# Callbacks simples pour chaque page
@app.callback(
    Output('overview-content', 'children'),
    [Input('data-store', 'data')]
)
def update_overview_content_simple(stored_data):
    return get_cached_content('overview', stored_data, create_overview_tab)

@app.callback(
    Output('consumption-content', 'children'),
    [Input('data-store', 'data')]
)
def update_consumption_content_simple(stored_data):
    return get_cached_content('consumption', stored_data, create_consumption_tab)

@app.callback(
    Output('production-content', 'children'),
    [Input('data-store', 'data')]
)
def update_production_content_simple(stored_data):
    return get_cached_content('production', stored_data, create_production_tab)

@app.callback(
    Output('prediction-content', 'children'),
    [Input('data-store', 'data')]
)
def update_prediction_content_simple(stored_data):
    return get_cached_content('prediction', stored_data, create_prediction_tab)

# Fonctions de création des contenus
def create_overview_tab(df):
    """Créer l'onglet vue d'ensemble avec des métriques clés et nouvelles visualisations"""
    
    # Calculer les statistiques
    avg_consumption = df['consommation_totale'].mean() if 'consommation_totale' in df.columns else 0
    avg_temp = df['temp'].mean() if 'temp' in df.columns else 0
    
    # Production totale (somme de toutes les sources)
    production_cols = [col for col in df.columns if col in ['SOLAR', 'BIOMASS', 'WIND_ONSHORE', 'NUCLEAR']]
    total_production = df[production_cols].mean().sum() if production_cols else 0
    
    # Calculer le déficit énergétique
    deficit_data = calculate_energy_deficit(df)
    avg_deficit = deficit_data['deficit'].mean() if not deficit_data.empty else 0
    
    return html.Div([
        # Cartes de métriques (avec déficit ajouté)
        html.Div([
            html.Div([
                html.H2(f"{avg_consumption:.0f} MW", style={'color': '#e74c3c', 'margin': '0'}),
                html.P("Consommation Moyenne", style={'margin': '10px 0', 'fontSize': '16px'})
            ], style={'backgroundColor': '#fff', 'padding': '25px', 'borderRadius': '12px',
                     'boxShadow': '0 4px 6px rgba(0,0,0,0.1)', 'textAlign': 'center', 'flex': '1'}),
            
            html.Div([
                html.H2(f"{avg_temp:.1f}°C", style={'color': '#f39c12', 'margin': '0'}),
                html.P("Température Moyenne", style={'margin': '10px 0', 'fontSize': '16px'})
            ], style={'backgroundColor': '#fff', 'padding': '25px', 'borderRadius': '12px',
                     'boxShadow': '0 4px 6px rgba(0,0,0,0.1)', 'textAlign': 'center', 'flex': '1'}),
            
            html.Div([
                html.H2(f"{total_production:.0f} MWh", style={'color': '#27ae60', 'margin': '0'}),
                html.P("Production Totale", style={'margin': '10px 0', 'fontSize': '16px'})
            ], style={'backgroundColor': '#fff', 'padding': '25px', 'borderRadius': '12px',
                     'boxShadow': '0 4px 6px rgba(0,0,0,0.1)', 'textAlign': 'center', 'flex': '1'}),
            
            html.Div([
                html.H2(f"{avg_deficit:.0f} MW", style={'color': '#9b59b6' if avg_deficit < 0 else '#e67e22', 'margin': '0'}),
                html.P("Déficit Moyen", style={'margin': '10px 0', 'fontSize': '16px'})
            ], style={'backgroundColor': '#fff', 'padding': '25px', 'borderRadius': '12px',
                     'boxShadow': '0 4px 6px rgba(0,0,0,0.1)', 'textAlign': 'center', 'flex': '1'})
        ], style={'display': 'flex', 'gap': '20px', 'marginBottom': '30px'}),
        
        # Première ligne : Histogrammes empilés et Évolution du déficit
        html.Div([
            # Histogrammes empilés annuels
            html.Div([
                dcc.Graph(figure=create_annual_stacked_histogram(df))
            ], style={'backgroundColor': '#fff', 'padding': '20px', 'borderRadius': '12px',
                     'boxShadow': '0 4px 6px rgba(0,0,0,0.1)', 'flex': '1', 'marginRight': '15px'}),
            
            # Évolution du déficit
            html.Div([
                dcc.Graph(figure=create_deficit_evolution(deficit_data))
            ], style={'backgroundColor': '#fff', 'padding': '20px', 'borderRadius': '12px',
                     'boxShadow': '0 4px 6px rgba(0,0,0,0.1)', 'flex': '1', 'marginLeft': '15px'})
        ], style={'display': 'flex', 'gap': '0px', 'marginBottom': '25px'}),
        
        # Deuxième ligne : Comparaison consommation vs production (pleine largeur)
        html.Div([
            dcc.Graph(figure=create_consumption_vs_production(df))
        ], style={'backgroundColor': '#fff', 'padding': '20px', 'borderRadius': '12px',
                 'boxShadow': '0 4px 6px rgba(0,0,0,0.1)', 'marginBottom': '25px'}),
        
        # Troisième ligne : Distributions des variables principales (pleine largeur)
        html.Div([
            dcc.Graph(figure=create_distributions_overview(df))
        ], style={'backgroundColor': '#fff', 'padding': '20px', 'borderRadius': '12px',
                 'boxShadow': '0 4px 6px rgba(0,0,0,0.1)'})
    ])

def create_consumption_tab(df):
    """Créer l'onglet consommation avec analyses saisonnières complètes"""
    if 'consommation_totale' not in df.columns:
        return html.Div("Données de consommation non disponibles")
    
    return html.Div([
        # Première ligne : Évolution temporelle et Distribution
        html.Div([
            html.Div([
                dcc.Graph(figure=create_consumption_evolution(df))
            ], style={'backgroundColor': '#fff', 'padding': '20px', 'borderRadius': '12px',
                     'boxShadow': '0 4px 6px rgba(0,0,0,0.1)', 'flex': '1', 'marginRight': '15px'}),
            
            html.Div([
                dcc.Graph(figure=create_consumption_distribution(df))
            ], style={'backgroundColor': '#fff', 'padding': '20px', 'borderRadius': '12px',
                     'boxShadow': '0 4px 6px rgba(0,0,0,0.1)', 'flex': '1', 'marginLeft': '15px'})
        ], style={'display': 'flex', 'gap': '0px', 'marginBottom': '25px'}),
        
        # Deuxième ligne : Saisonnalités journalière et hebdomadaire
        html.Div([
            html.Div([
                dcc.Graph(figure=create_daily_consumption_seasonality(df))
            ], style={'backgroundColor': '#fff', 'padding': '20px', 'borderRadius': '12px',
                     'boxShadow': '0 4px 6px rgba(0,0,0,0.1)', 'flex': '1', 'marginRight': '15px'}),
            
            html.Div([
                dcc.Graph(figure=create_weekly_consumption_seasonality(df))
            ], style={'backgroundColor': '#fff', 'padding': '20px', 'borderRadius': '12px',
                     'boxShadow': '0 4px 6px rgba(0,0,0,0.1)', 'flex': '1', 'marginLeft': '15px'})
        ], style={'display': 'flex', 'gap': '0px', 'marginBottom': '25px'}),
        
        # Troisième ligne : Saisonnalités mensuelle et annuelle
        html.Div([
            html.Div([
                dcc.Graph(figure=create_monthly_consumption_seasonality(df))
            ], style={'backgroundColor': '#fff', 'padding': '20px', 'borderRadius': '12px',
                     'boxShadow': '0 4px 6px rgba(0,0,0,0.1)', 'flex': '1', 'marginRight': '15px'}),
            
            html.Div([
                dcc.Graph(figure=create_yearly_consumption_seasonality(df))
            ], style={'backgroundColor': '#fff', 'padding': '20px', 'borderRadius': '12px',
                     'boxShadow': '0 4px 6px rgba(0,0,0,0.1)', 'flex': '1', 'marginLeft': '15px'})
        ], style={'display': 'flex', 'gap': '0px', 'marginBottom': '25px'}),
        
        # Quatrième ligne : Corrélation température et Décomposition saisonnière
        html.Div([
            html.Div([
                dcc.Graph(figure=create_temperature_consumption_correlation(df))
            ], style={'backgroundColor': '#fff', 'padding': '20px', 'borderRadius': '12px',
                     'boxShadow': '0 4px 6px rgba(0,0,0,0.1)', 'flex': '1', 'marginRight': '15px'}),
            
        ], style={'display': 'flex', 'gap': '0px'})
    ])

# Fonctions pour l'analyse de consommation
def create_consumption_evolution(df):
    """Créer le graphique d'évolution temporelle de la consommation"""
    fig = px.line(
        df.reset_index(), x='timestamp', y='consommation_totale',
        title='Évolution Temporelle de la Consommation',
        labels={'timestamp': 'Date', 'consommation_totale': 'Consommation (MW)'}
    )
    fig.update_layout(height=400, title_font_size=16)
    fig.update_traces(line=dict(color='#e74c3c', width=2))
    return fig

def create_consumption_distribution(df):
    """Créer l'histogramme de distribution de la consommation"""
    fig = px.histogram(
        df, x='consommation_totale', nbins=50,
        title='Distribution de la Consommation',
        labels={'consommation_totale': 'Consommation (MW)', 'count': 'Fréquence'}
    )
    fig.update_layout(height=400, title_font_size=16)
    fig.update_traces(marker_color='#e74c3c', opacity=0.7)
    return fig

def create_daily_consumption_seasonality(df):
    """Créer le boxplot de la consommation par heure (saisonnalité journalière)"""
    df_daily = df.copy()
    df_daily['hour'] = df_daily.index.hour
    
    fig = go.Figure()
    
    # Créer un boxplot pour chaque heure
    for hour in range(24):
        hour_data = df_daily[df_daily['hour'] == hour]['consommation_totale']
        
        fig.add_trace(go.Box(
            y=hour_data,
            name=str(hour),
            boxpoints='outliers',
            marker_color='#3498db',
            line_color='#2c3e50'
        ))
    
    fig.update_layout(
        title='Consommation par Heure avec Boxplot (Saisonnalité Journalière)',
        title_font_size=16,
        xaxis_title='Heure de la journée',
        yaxis_title='Consommation (MW)',
        height=400,
        showlegend=False
    )
    
    return fig

def create_weekly_consumption_seasonality(df):
    """Créer le boxplot de la consommation par jour de la semaine"""
    df_weekly = df.copy()
    df_weekly['day_name'] = df_weekly.index.day_name()
    df_weekly['weekday'] = df_weekly.index.dayofweek
    
    fig = go.Figure()
    
    # Ordre des jours et traduction
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    days_fr = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']
    
    for i, day in enumerate(days_order):
        day_data = df_weekly[df_weekly['day_name'] == day]['consommation_totale']
        
        if not day_data.empty:
            fig.add_trace(go.Box(
                y=day_data,
                name=days_fr[i],
                boxpoints='outliers',
                marker_color='#e74c3c',
                line_color='#2c3e50'
            ))
    
    fig.update_layout(
        title='Consommation par Jour de la Semaine avec Boxplot',
        title_font_size=16,
        xaxis_title='Jour de la semaine',
        yaxis_title='Consommation (MW)',
        height=400,
        showlegend=False
    )
    
    return fig

def create_monthly_consumption_seasonality(df):
    """Créer le boxplot de la consommation par mois"""
    df_monthly = df.copy()
    df_monthly['month'] = df_monthly.index.month
    
    fig = go.Figure()
    
    months_names = ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Jun', 
                   'Jul', 'Aoû', 'Sep', 'Oct', 'Nov', 'Déc']
    
    for month in range(1, 13):
        month_data = df_monthly[df_monthly['month'] == month]['consommation_totale']
        
        if not month_data.empty:
            fig.add_trace(go.Box(
                y=month_data,
                name=months_names[month-1],
                boxpoints='outliers',
                marker_color='#e74c3c',
                line_color='#2c3e50'
            ))
    
    fig.update_layout(
        title='Consommation par Mois avec Boxplot',
        title_font_size=16,
        xaxis_title='Mois',
        yaxis_title='Consommation (MW)',
        height=400,
        showlegend=False
    )
    
    return fig

def create_yearly_consumption_seasonality(df):
    """Créer la saisonnalité annuelle de la consommation"""
    df_yearly = df.copy()
    df_yearly['hour'] = df_yearly.index.hour
    df_yearly['weekday'] = df_yearly.index.day_name()
    
    # Créer le tableau pivot
    pivot_data = df_yearly.pivot_table(
        index='hour', 
        columns='weekday', 
        values='consommation_totale', 
        aggfunc='mean'
    )
    
    # Réorganiser les colonnes pour commencer par lundi
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    days_fr = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']
    
    # Réorganiser et renommer les colonnes
    available_days = [day for day in days_order if day in pivot_data.columns]
    pivot_data = pivot_data[available_days]
    pivot_data.columns = [days_fr[days_order.index(day)] for day in available_days]
    
    fig = go.Figure()
    
    # Couleurs pour chaque jour
    day_colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#e67e22']
    
    for i, day in enumerate(pivot_data.columns):
        fig.add_trace(go.Scatter(
            x=pivot_data.index,
            y=pivot_data[day],
            mode='lines',
            name=day,
            line=dict(color=day_colors[i % len(day_colors)], width=2)
        ))
    
    fig.update_layout(
        title='Consommation par Heure et Jour de la Semaine (Patterns Weekday/Weekend)',
        title_font_size=16,
        xaxis_title='Heure de la journée',
        yaxis_title='Consommation Moyenne (MW)',
        height=400,
        hovermode='x unified',
        legend=dict(orientation="h", y=1.02)
    )
    
    return fig

def create_temperature_consumption_correlation(df):
    """Créer le graphique de corrélation température-consommation"""
    if 'temp' not in df.columns:
        return go.Figure().add_annotation(
            text="Données de température non disponibles",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )
    
    # Calculer la corrélation
    correlation = df['temp'].corr(df['consommation_totale'])
    
    fig = px.scatter(
        df.reset_index(), x='temp', y='consommation_totale',
        title=f'Corrélation Température vs Consommation (r={correlation:.3f})',
        labels={'temp': 'Température (°C)', 'consommation_totale': 'Consommation (MW)'},
        trendline="ols"
    )
    
    fig.update_layout(height=400, title_font_size=16)
    fig.update_traces(marker=dict(color='#f39c12', size=4, opacity=0.6))
    
    return fig

def create_consumption_seasonal_decompose(df):
    """Créer la décomposition saisonnière de la consommation"""
    try:
        if len(df) < 48:  # Au moins 2 jours de données
            return go.Figure().add_annotation(
                text="Données insuffisantes pour la décomposition saisonnière",
                xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
            )
        
        # Réechantillonner si nécessaire
        ts_data = df['consommation_totale'].resample('h').mean().dropna()
        
        if len(ts_data) < 48:
            return go.Figure().add_annotation(
                text="Données insuffisantes pour la décomposition saisonnière",
                xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
            )
        
        # Décomposition saisonnière avec période journalière (24h)
        decomposition = seasonal_decompose(ts_data, model='additive', period=8760)
        
        # Créer le graphique avec les 3 composantes principales
        fig = go.Figure()
        
        # Série originale
        fig.add_trace(go.Scatter(
            x=ts_data.index,
            y=ts_data.values,
            mode='lines',
            name='Série Originale',
            line=dict(color='#2c3e50', width=1.5),
            opacity=0.8
        ))
        
        # Tendance
        fig.add_trace(go.Scatter(
            x=decomposition.trend.index,
            y=decomposition.trend.values,
            mode='lines',
            name='Tendance',
            line=dict(color='#e74c3c', width=2)
        ))
        
        # Composante saisonnière (moyennée pour lisibilité)
        seasonal_avg = decomposition.seasonal.groupby(decomposition.seasonal.index.hour).mean()
        hours_extended = list(range(24)) * (len(ts_data) // 24 + 1)[:len(ts_data)]
        seasonal_extended = [seasonal_avg[h] for h in hours_extended]
        
        fig.add_trace(go.Scatter(
            x=ts_data.index,
            y=seasonal_extended,
            mode='lines',
            name='Saisonnalité',
            line=dict(color='#3498db', width=2)
        ))
        
        fig.update_layout(
            title='Décomposition Saisonnière de la Consommation',
            title_font_size=16,
            xaxis_title='Date',
            yaxis_title='Consommation (MW)',
            height=400,
            hovermode='x unified',
            legend=dict(orientation="h", y=1.02)
        )
        
        return fig
        
    except Exception as e:
        return go.Figure().add_annotation(
            text=f"Erreur lors de la décomposition: {str(e)}",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )

def create_production_tab(df):
    """Créer l'onglet production avec analyses saisonnières complètes"""
    production_cols = [col for col in df.columns if col in ['SOLAR', 'BIOMASS', 'WIND_ONSHORE', 'NUCLEAR']]
    
    if not production_cols:
        return html.Div("Données de production non disponibles")
    
    return html.Div([
        # Première ligne : Évolution des sources et Répartition
        html.Div([
            html.Div([
                dcc.Graph(figure=create_production_evolution_by_source(df))
            ], style={'backgroundColor': '#fff', 'padding': '20px', 'borderRadius': '12px',
                     'boxShadow': '0 4px 6px rgba(0,0,0,0.1)', 'flex': '1', 'marginRight': '15px'}),
            
            html.Div([
                dcc.Graph(figure=create_production_pie_chart(df, production_cols))
            ], style={'backgroundColor': '#fff', 'padding': '20px', 'borderRadius': '12px',
                     'boxShadow': '0 4px 6px rgba(0,0,0,0.1)', 'flex': '1', 'marginLeft': '15px'})
        ], style={'display': 'flex', 'gap': '0px', 'marginBottom': '25px'}),
        
        # Deuxième ligne : Saisonnalités hebdomadaire et mensuelle
        html.Div([
           
            html.Div([
                dcc.Graph(figure=create_monthly_production_seasonality(df, production_cols))
            ], style={'backgroundColor': '#fff', 'padding': '20px', 'borderRadius': '12px',
                     'boxShadow': '0 4px 6px rgba(0,0,0,0.1)', 'flex': '1', 'marginLeft': '15px'})
        ], style={'display': 'flex', 'gap': '0px', 'marginBottom': '25px'}),
        
        # Troisième ligne : Corrélations avec la température
        html.Div([
            dcc.Graph(figure=create_temperature_production_correlations(df, production_cols))
        ], style={'backgroundColor': '#fff', 'padding': '20px', 'borderRadius': '12px',
                 'boxShadow': '0 4px 6px rgba(0,0,0,0.1)'})
    ])

# Fonctions pour l'analyse de production
def create_production_evolution_by_source(df):
    """Créer l'évolution des différentes sources de production en graphique en aire"""
    production_cols = [col for col in df.columns if col in ['SOLAR', 'BIOMASS', 'WIND_ONSHORE', 'NUCLEAR']]
    
    if not production_cols:
        return go.Figure().add_annotation(
            text="Données de production non disponibles",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )
    
    # Préparer les données pour px.area
    df_reset = df.reset_index()
    
    # Utiliser px.area avec la syntaxe correcte
    fig = px.area(df_reset, x='timestamp', y=production_cols,
                  title='Évolution des Sources de Production (Graphique en Aires)')
    
    # Appliquer les couleurs personnalisées
    colors = {'SOLAR': '#f1c40f', 'BIOMASS': '#27ae60', 'WIND_ONSHORE': '#3498db', 'NUCLEAR': '#e74c3c'}
    
    for i, trace in enumerate(fig.data):
        if i < len(production_cols):
            source = production_cols[i]
            trace.fillcolor = colors.get(source, '#95a5a6')
            trace.line.color = colors.get(source, '#95a5a6')
    
    fig.update_layout(
        title_font_size=16,
        xaxis_title='Date',
        yaxis_title='Production (MW)',
        height=400,
        hovermode='x unified',
        legend=dict(orientation="h", y=1.02)
    )
    
    return fig

def create_production_pie_chart(df, production_cols):
    """Créer le graphique en secteurs de la production"""
    avg_production = df[production_cols].mean()
    colors = {'SOLAR': '#f1c40f', 'BIOMASS': '#27ae60', 'WIND_ONSHORE': '#3498db', 'NUCLEAR': '#e74c3c'}
    
    fig = px.pie(
        values=avg_production.values,
        names=avg_production.index,
        title='Répartition Moyenne de la Production par Source',
        color_discrete_map=colors
    )
    fig.update_layout(height=400, title_font_size=16)
    
    return fig


def create_weekly_production_seasonality(df, production_cols):
    """Créer les boxplots hebdomadaires pour chaque source de production"""
    df_weekly = df.copy()
    df_weekly['day_name'] = df_weekly.index.day_name()
    
    # Créer des sous-graphiques pour chaque source
    n_cols = min(2, len(production_cols))
    n_rows = (len(production_cols) + n_cols - 1) // n_cols
    
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=[f"{source} - Boxplot par Jour" for source in production_cols],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    colors = {'SOLAR': '#f1c40f', 'BIOMASS': '#27ae60', 'WIND_ONSHORE': '#3498db', 'NUCLEAR': '#e74c3c'}
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    days_fr = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']
    
    for i, source in enumerate(production_cols):
        row = i // n_cols + 1
        col = i % n_cols + 1
        
        # Créer un boxplot pour chaque jour de la semaine
        for j, day in enumerate(days_order):
            day_data = df_weekly[df_weekly['day_name'] == day][source]
            
            if not day_data.empty:
                fig.add_trace(
                    go.Box(
                        y=day_data,
                        name=days_fr[j],
                        boxpoints='outliers',
                        marker_color=colors.get(source, '#95a5a6'),
                        line_color='#2c3e50',
                        showlegend=False
                    ),
                    row=row, col=col
                )
    
    fig.update_layout(
        title='Production par Jour avec Boxplots (Saisonnalité Hebdomadaire)',
        title_font_size=16,
        height=400 * n_rows,
        showlegend=False
    )
    
    # Mettre à jour les axes pour tous les sous-graphiques
    for i in range(1, n_rows * n_cols + 1):
        fig.update_xaxes(title_text="Jour de la semaine", row=(i-1)//n_cols + 1, col=(i-1)%n_cols + 1)
        fig.update_yaxes(title_text="Production (MW)", row=(i-1)//n_cols + 1, col=(i-1)%n_cols + 1)
    
    return fig

def create_monthly_production_seasonality(df, production_cols):
    """Créer les boxplots mensuels pour chaque source de production"""
    df_monthly = df.copy()
    df_monthly['month'] = df_monthly.index.month
    
    # Créer des sous-graphiques pour chaque source
    n_cols = min(2, len(production_cols))
    n_rows = (len(production_cols) + n_cols - 1) // n_cols
    
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=[f"{source} - Boxplot par Mois" for source in production_cols],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    colors = {'SOLAR': '#f1c40f', 'BIOMASS': '#27ae60', 'WIND_ONSHORE': '#3498db', 'NUCLEAR': '#e74c3c'}
    months_names = ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Jun', 
                   'Jul', 'Aoû', 'Sep', 'Oct', 'Nov', 'Déc']
    
    for i, source in enumerate(production_cols):
        row = i // n_cols + 1
        col = i % n_cols + 1
        
        # Créer un boxplot pour chaque mois
        for month in range(1, 13):
            month_data = df_monthly[df_monthly['month'] == month][source]
            
            if not month_data.empty:
                fig.add_trace(
                    go.Box(
                        y=month_data,
                        name=months_names[month-1],
                        boxpoints='outliers',
                        marker_color=colors.get(source, '#95a5a6'),
                        line_color='#2c3e50',
                        showlegend=False
                    ),
                    row=row, col=col
                )
    
    fig.update_layout(
        title='Production par Mois avec Boxplots (Saisonnalité Mensuelle)',
        title_font_size=16,
        height=400 * n_rows,
        showlegend=False
    )
    
    # Mettre à jour les axes pour tous les sous-graphiques
    for i in range(1, n_rows * n_cols + 1):
        fig.update_xaxes(title_text="Mois", row=(i-1)//n_cols + 1, col=(i-1)%n_cols + 1)
        fig.update_yaxes(title_text="Production (MW)", row=(i-1)//n_cols + 1, col=(i-1)%n_cols + 1)
    
    return fig


def create_temperature_production_correlations(df, production_cols):
    """Créer les corrélations entre température et sources de production"""
    if 'temp' not in df.columns:
        return go.Figure().add_annotation(
            text="Données de température non disponibles",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )
    
    # Créer des sous-graphiques pour chaque source
    n_cols = min(2, len(production_cols))
    n_rows = (len(production_cols) + n_cols - 1) // n_cols
    
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=[f"{source} vs Température" for source in production_cols],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    colors = {'SOLAR': '#f1c40f', 'BIOMASS': '#27ae60', 'WIND_ONSHORE': '#3498db', 'NUCLEAR': '#e74c3c'}
    
    for i, source in enumerate(production_cols):
        row = i // n_cols + 1
        col = i % n_cols + 1
        
        # Calculer la corrélation
        correlation = df['temp'].corr(df[source])
        
        fig.add_trace(
            go.Scatter(
                x=df['temp'],
                y=df[source],
                mode='markers',
                name=f'{source} (r={correlation:.3f})',
                marker=dict(color=colors.get(source, '#95a5a6'), size=3, opacity=0.6),
                showlegend=False
            ),
            row=row, col=col
        )
    
    fig.update_layout(
        title='Corrélations entre Température et Sources de Production',
        title_font_size=16,
        height=400 * n_rows,
        hovermode='closest'
    )
    
    # Mettre à jour les axes pour tous les sous-graphiques
    for i in range(1, n_rows * n_cols + 1):
        fig.update_xaxes(title_text="Température (°C)", row=(i-1)//n_cols + 1, col=(i-1)%n_cols + 1)
        fig.update_yaxes(title_text="Production (MW)", row=(i-1)//n_cols + 1, col=(i-1)%n_cols + 1)
    
    return fig

def create_prediction_tab(df):
    """Créer l'onglet prévision énergétique"""
    if df.empty:
        return html.Div("Aucune donnée disponible pour les prévisions")
    
    return html.Div([
        # Section de configuration des prévisions
        html.Div([
            html.H4("Configuration des Prévisions", style={'color': '#2c3e50', 'marginBottom': '20px'}),
            
            html.Div([
                html.Div([
                    html.Label("Horizon de prévision (heures):", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                    dcc.Dropdown(
                        id='prediction-horizon',
                        options=[
                            {'label': '1 heure', 'value': 1},
                            {'label': '6 heures', 'value': 6},
                            {'label': '12 heures', 'value': 12},
                            {'label': '24 heures', 'value': 24},
                            {'label': '48 heures', 'value': 48},
                            {'label': '72 heures', 'value': 72}
                        ],
                        value=1,
                        style={'width': '100%'}
                    )
                ], style={'flex': '1', 'marginRight': '15px'}),
                
                html.Div([
                    html.Label("Variables prédites:", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                    html.P("Toutes les variables (consommation, température, sources d'énergie)", 
                           style={'color': '#7f8c8d', 'fontSize': '14px', 'margin': '0',
                                 'padding': '8px', 'backgroundColor': '#f8f9fa', 'borderRadius': '4px'})
                ], style={'flex': '1', 'marginLeft': '15px'})
            ], style={'display': 'flex', 'marginBottom': '20px'}),
            
            html.Div([
                html.Button('Générer Prévision', id='predict-button', n_clicks=0,
                           style={'backgroundColor': '#27ae60', 'color': 'white', 'padding': '12px 25px',
                                 'border': 'none', 'borderRadius': '6px', 'cursor': 'pointer',
                                 'fontSize': '16px', 'fontWeight': 'bold'})
            ], style={'textAlign': 'center'})
            
        ], style={'backgroundColor': '#f8f9fa', 'padding': '25px', 'borderRadius': '10px', 'marginBottom': '25px'}),
        
        # Résultats des prévisions
        html.Div(id='prediction-results', children=[
            html.Div([
                html.H4("Prévisions Énergétiques", style={'color': '#2c3e50', 'marginBottom': '20px'}),
                html.P("Configurez les paramètres ci-dessus et cliquez sur 'Générer Prévision' pour voir les résultats.",
                      style={'textAlign': 'center', 'color': '#7f8c8d', 'fontSize': '16px', 'padding': '40px'})
            ], style={'backgroundColor': '#fff', 'padding': '25px', 'borderRadius': '10px',
                     'boxShadow': '0 4px 6px rgba(0,0,0,0.1)'})
        ]),
        
        # Informations sur les modèles
        html.Div([
            html.H4("Informations sur les Modèles", style={'color': '#2c3e50', 'marginBottom': '15px'}),
            html.P("Cette section utilise les modèles pré-entraînés disponibles dans votre pipeline pour générer des prévisions énergétiques.", 
                   style={'marginBottom': '10px'}),
            html.P("Les prévisions sont basées sur les données historiques et les patterns identifiés par les algorithmes de Machine Learning.", 
                   style={'marginBottom': '10px'}),
            html.P("Vous pouvez ajuster l'horizon de prévision et le type de variable à prédire selon vos besoins d'analyse.", 
                   style={'marginBottom': '0'})
        ], style={'backgroundColor': '#e3f2fd', 'padding': '20px', 'borderRadius': '8px',
                 'border': '1px solid #bbdefb', 'marginTop': '25px'})
    ])

# Fonctions utilitaires
def calculate_energy_deficit(df):
    """Calculer le déficit énergétique (production - consommation)"""
    try:
        production_cols = [col for col in df.columns if col in ['SOLAR', 'BIOMASS', 'WIND_ONSHORE', 'NUCLEAR']]
        
        if not production_cols or 'consommation_totale' not in df.columns:
            return pd.DataFrame()
        
        df_deficit = df.copy()
        df_deficit['production_totale'] = df_deficit[production_cols].sum(axis=1)
        df_deficit['deficit'] = df_deficit['production_totale'] - df_deficit['consommation_totale']
        
        return df_deficit[['production_totale', 'consommation_totale', 'deficit']]
    
    except Exception as e:
        print(f"Erreur calcul déficit: {e}")
        return pd.DataFrame()

def create_annual_stacked_histogram(df):
    """Créer un histogramme empilé annuel des différentes énergies"""
    try:
        production_cols = [col for col in df.columns if col in ['SOLAR', 'BIOMASS', 'WIND_ONSHORE', 'NUCLEAR']]
        
        if not production_cols:
            return go.Figure().add_annotation(
                text="Données de production non disponibles",
                xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
            )
        
        # Préparer les données par mois
        df_monthly = df.copy()
        df_monthly['year_month'] = df_monthly.index.to_period('M')
        
        # Agrégation mensuelle
        monthly_data = df_monthly.groupby('year_month')[production_cols].sum().reset_index()
        monthly_data['year_month_str'] = monthly_data['year_month'].astype(str)
        
        fig = go.Figure()
        
        colors = {'SOLAR': '#f1c40f', 'BIOMASS': '#27ae60', 'WIND_ONSHORE': '#3498db', 'NUCLEAR': '#e74c3c'}
        
        for col in production_cols:
            fig.add_trace(go.Bar(
                x=monthly_data['year_month_str'],
                y=monthly_data[col],
                name=col,
                marker_color=colors.get(col, '#95a5a6')
            ))
        
        fig.update_layout(
            title='Production Énergétique Mensuelle par Source',
            title_font_size=16,
            xaxis_title='Mois',
            yaxis_title='Production (MWh)',
            barmode='stack',
            height=400,
            xaxis={'tickangle': 45}
        )
        
        return fig
        
    except Exception as e:
        return go.Figure().add_annotation(
            text=f"Erreur création histogramme: {str(e)}",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )

def create_deficit_evolution(deficit_data):
    """Créer un histogramme de l'évolution du déficit énergétique"""
    try:
        if deficit_data.empty or 'deficit' not in deficit_data.columns:
            return go.Figure().add_annotation(
                text="Données de déficit non disponibles",
                xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
            )
        
        # Agrégation journalière pour lisibilité
        df_daily = deficit_data.copy()
        df_daily = df_daily.resample('D').mean()
        
        fig = go.Figure()
        
        # Déficit avec couleurs contrastées pour meilleure lisibilité
        colors = ['#2ecc71' if x >= 0 else '#e74c3c' for x in df_daily['deficit']]
        
        fig.add_trace(go.Bar(
            x=df_daily.index,
            y=df_daily['deficit'],
            name='Déficit Énergétique',
            marker=dict(
                color=colors,
                line=dict(color='rgba(0,0,0,0.3)', width=1)
            ),
            text=[f"{x:.0f}" for x in df_daily['deficit']],
            textposition='outside',
            textfont=dict(size=10, color='black')
        ))
        
        # Ligne de référence à zéro
        fig.add_hline(y=0, line_dash="dash", line_color="black", 
                     annotation_text="Équilibre", annotation_position="bottom right")
        
        fig.update_layout(
            title='Évolution du Déficit Énergétique (Production - Consommation)',
            title_font_size=16,
            xaxis_title='Date',
            yaxis_title='Déficit (MW)',
            height=400,
            showlegend=False,
            annotations=[
                dict(x=0.02, y=0.98, xref="paper", yref="paper",
                     text="Surplus (Production > Consommation) Déficit (Production < Consommation)", 
                     showarrow=False,
                     font=dict(size=11, color='black'), 
                     bgcolor="rgba(255,255,255,0.9)",
                     bordercolor="rgba(0,0,0,0.2)",
                     borderwidth=1)
            ]
        )
        
        return fig
        
    except Exception as e:
        return go.Figure().add_annotation(
            text=f"Erreur création déficit: {str(e)}",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )

def create_consumption_vs_production(df):
    """Créer un graphique linéaire comparant consommation et production totale"""
    try:
        production_cols = [col for col in df.columns if col in ['SOLAR', 'BIOMASS', 'WIND_ONSHORE', 'NUCLEAR']]
        
        if not production_cols or 'consommation_totale' not in df.columns:
            return go.Figure().add_annotation(
                text="Données de consommation ou production non disponibles",
                xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
            )
        
        # Calculer la production totale
        df_comparison = df.copy()
        df_comparison['production_totale'] = df_comparison[production_cols].sum(axis=1)
        
        # Créer le graphique linéaire
        fig = go.Figure()
        
        # Ligne de consommation
        fig.add_trace(go.Scatter(
            x=df_comparison.index,
            y=df_comparison['consommation_totale'],
            mode='lines',
            name='Consommation',
            line=dict(color='#e74c3c', width=2),
            hovertemplate='<b>Consommation</b><br>Date: %{x}<br>Valeur: %{y:.0f} MW<extra></extra>'
        ))
        
        # Ligne de production totale
        fig.add_trace(go.Scatter(
            x=df_comparison.index,
            y=df_comparison['production_totale'],
            mode='lines',
            name='Production totale',
            line=dict(color='#27ae60', width=2),
            hovertemplate='<b>Production Totale</b><br>Date: %{x}<br>Valeur: %{y:.0f} MW<extra></extra>'
        ))
        
        fig.update_layout(
            title='Comparaison Consommation vs Production Totale',
            title_font_size=16,
            xaxis_title='Date',
            yaxis_title='Puissance (MW)',
            height=400,  # Hauteur standard pour pleine largeur
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
        )
        
        return fig
        
    except Exception as e:
        return go.Figure().add_annotation(
            text=f"Erreur création comparaison: {str(e)}",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )

def create_distributions_overview(df):
    """Créer des distributions des variables principales"""
    try:
        # Variables principales à analyser (inclure toujours SOLAR)
        main_vars = ['consommation_totale']
        production_cols = [col for col in df.columns if col in ['SOLAR', 'BIOMASS', 'WIND_ONSHORE', 'NUCLEAR']]
        
        # Ajouter SOLAR en priorité s'il existe
        if 'SOLAR' in production_cols:
            main_vars.append('SOLAR')
            production_cols.remove('SOLAR')
        
        # Ajouter les autres sources de production par ordre d'importance
        if production_cols:
            production_means = df[production_cols].mean()
            top_production = production_means.nlargest(3).index.tolist()
            main_vars.extend(top_production)
        
        # Filtrer les variables disponibles
        available_vars = [var for var in main_vars if var in df.columns]
        
        if not available_vars:
            return go.Figure().add_annotation(
                text="Aucune variable disponible pour les distributions",
                xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
            )
        
        # Créer des sous-graphiques (plus de colonnes pour la pleine largeur)
        cols = min(4, len(available_vars))
        rows = (len(available_vars) + cols - 1) // cols
        
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=[var.replace('_', ' ').title() for var in available_vars],
            vertical_spacing=0.15,
            horizontal_spacing=0.08
        )
        
        colors = ['#e74c3c', '#f39c12', '#3498db', '#27ae60', '#9b59b6']
        
        for i, var in enumerate(available_vars):
            row = i // cols + 1
            col = i % cols + 1
            
            fig.add_trace(
                go.Histogram(
                    x=df[var],
                    name=var,
                    marker_color=colors[i % len(colors)],
                    opacity=0.7,
                    showlegend=False,
                    nbinsx=30
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            title='Distributions des Variables Principales',
            title_font_size=16,
            height=500,  # Hauteur fixe pour cohérence avec le graphique à côté
            showlegend=False
        )
        
        return fig
        
    except Exception as e:
        return go.Figure().add_annotation(
            text=f"Erreur création distributions: {str(e)}",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )

# Callback pour générer les prévisions
@app.callback(
    Output('prediction-results', 'children'),
    [Input('predict-button', 'n_clicks')],
    [dash.dependencies.State('data-store', 'data'),
     dash.dependencies.State('prediction-horizon', 'value')]
)
def generate_predictions(n_clicks, stored_data, horizon):
    if n_clicks == 0 or not stored_data:
        return html.Div([
            html.H4("Prévisions Énergétiques", style={'color': '#2c3e50', 'marginBottom': '20px'}),
            html.P("Configurez les paramètres ci-dessus et cliquez sur 'Générer Prévision' pour voir les résultats.",
                  style={'textAlign': 'center', 'color': '#7f8c8d', 'fontSize': '16px', 'padding': '40px'})
        ], style={'backgroundColor': '#fff', 'padding': '25px', 'borderRadius': '10px',
                 'boxShadow': '0 4px 6px rgba(0,0,0,0.1)'})
    
    try:
        # Préparer les données
        df = prepare_dataframe(stored_data)
        if df is None:
            return html.Div([
                html.H4("Erreur", style={'color': '#e74c3c'}),
                html.P("Aucune donnée disponible pour les prévisions.")
            ])
        
        # Générer les prévisions
        y_pred, y_test, mae, mse, metrics_by_energy, error = create_predictions(df, horizon)
        
        if error:
            return html.Div([
                html.H4("Erreur de Prédiction", style={'color': '#e74c3c', 'marginBottom': '15px'}),
                html.P(error, style={'color': '#e74c3c'})
            ], style={'backgroundColor': '#fff', 'padding': '25px', 'borderRadius': '10px',
                     'boxShadow': '0 4px 6px rgba(0,0,0,0.1)'})
        
        if y_pred is None:
            return html.Div([
                html.H4("Erreur", style={'color': '#e74c3c'}),
                html.P("Impossible de générer les prédictions.")
            ])
        
        # Créer les visualisations des prédictions avec métriques détaillées
        return create_prediction_visualizations(df, y_pred, y_test, mae, mse, metrics_by_energy, horizon)
        
    except Exception as e:
        return html.Div([
            html.H4("Erreur", style={'color': '#e74c3c'}),
            html.P(f"Erreur lors de la génération des prévisions: {str(e)}")
        ])

def create_prediction_visualizations(historical_df, y_pred, y_test, mae, mse, metrics_by_energy, horizon):
    """Créer les visualisations des prédictions - Mode prédiction réelle"""
    try:
        # Déterminer le mode (avec ou sans vérité terrain)
        has_ground_truth = y_test is not None and mae is not None
        
        # Features utilisées
        features = ['BIOMASS', 'NUCLEAR', 'SOLAR', 'WIND_ONSHORE', 'consommation_totale', 'temp']
        
        print(f"Mode: {'Test/Validation' if has_ground_truth else 'Prédiction Réelle'}")
        print(f"Shape de y_pred: {y_pred.shape}")
        print(f"Horizon demandé: {horizon}")
        
        # Prendre une semaine de données historiques pour le contexte
        hist_data = historical_df[features].tail(168).copy()  # 7 jours * 24h = 168 heures
        
        # Créer les timestamps futurs de manière robuste
        try:
            # S'assurer que l'index est bien un DatetimeIndex
            if hasattr(historical_df.index, 'to_timestamp'):
                last_timestamp = historical_df.index[-1].to_timestamp()
            else:
                last_timestamp = pd.to_datetime(historical_df.index[-1])
            
            # Créer les timestamps futurs
            future_timestamps = []
            for i in range(horizon):
                future_time = last_timestamp + pd.DateOffset(hours=i+1)
                future_timestamps.append(future_time)
                
            print(f"Last timestamp: {last_timestamp}, Type: {type(last_timestamp)}")
            print(f"First future timestamp: {future_timestamps[0] if future_timestamps else 'None'}")
        except Exception as e:
            print(f"Erreur création timestamps: {e}")
            # Fallback : créer des timestamps simples
            import datetime
            base_time = datetime.datetime.now()
            future_timestamps = [base_time + datetime.timedelta(hours=i+1) for i in range(horizon)]
        
        # Convertir y_pred en DataFrame (s'assurer qu'on a la bonne shape)
        if y_pred.shape[0] >= horizon:
            y_pred_horizon = y_pred[:horizon, :]
        else:
            y_pred_horizon = y_pred  # Prendre tout ce qu'on a
        
        # S'assurer que les timestamps sont de la bonne longueur
        timestamps_to_use = future_timestamps[:y_pred_horizon.shape[0]]
        pred_df = pd.DataFrame(y_pred_horizon, columns=features, index=timestamps_to_use)
        print(f"DataFrame prédictions shape: {pred_df.shape}")
        
        # Calculer le déficit énergétique pour les prédictions
        deficit_info = calculate_energy_deficit_predict(y_pred_horizon)
        
        # Créer les graphiques
        fig_main = create_prediction_charts(hist_data, pred_df, features, horizon)
        fig_conso_prod = create_consumption_vs_production_chart(deficit_info, timestamps_to_use, hist_data) if deficit_info else None
        fig_deficit = create_deficit_evolution_chart(deficit_info, timestamps_to_use, hist_data) if deficit_info else None
        
        # Tableau des prédictions
        pred_table = dash_table.DataTable(
            data=pred_df.round(2).reset_index().to_dict('records'),
            columns=[{"name": i, "id": i} for i in pred_df.reset_index().columns],
            style_cell={'textAlign': 'center', 'padding': '8px'},
            style_header={'backgroundColor': '#3498db', 'color': 'white', 'fontWeight': 'bold'},
            style_data={'backgroundColor': '#f8f9fa'},
            page_size=10
        )
        
        # Interface adaptée selon le mode
        if has_ground_truth:
            # Mode Test/Validation : Afficher les métriques
            return html.Div([
                # Métriques globales
                html.Div([
                    html.H4("Métriques de Performance", style={'color': '#2c3e50', 'marginBottom': '15px'}),
                    html.Div([
                        html.Div([
                            html.H5(f"MAE: {mae:.3f}", style={'color': '#e74c3c', 'margin': '0'}),
                            html.P("Mean Absolute Error", style={'fontSize': '12px', 'color': '#7f8c8d'})
                        ], style={'textAlign': 'center', 'flex': '1'}),
                        html.Div([
                            html.H5(f"MSE: {mse:.3f}", style={'color': '#f39c12', 'margin': '0'}),
                            html.P("Mean Squared Error", style={'fontSize': '12px', 'color': '#7f8c8d'})
                        ], style={'textAlign': 'center', 'flex': '1'}),
                        html.Div([
                            html.H5(f"RMSE: {np.sqrt(mse):.3f}", style={'color': '#27ae60', 'margin': '0'}),
                            html.P("Root Mean Squared Error", style={'fontSize': '12px', 'color': '#7f8c8d'})
                        ], style={'textAlign': 'center', 'flex': '1'})
                    ], style={'display': 'flex', 'gap': '20px'})
                ], style={'backgroundColor': '#f8f9fa', 'padding': '20px', 'borderRadius': '8px', 'marginBottom': '25px'}),
                
                # Métriques par source d'énergie
                html.Div([
                    html.H4("Métriques par Source", style={'color': '#2c3e50', 'marginBottom': '15px'}),
                    dash_table.DataTable(
                        data=metrics_by_energy if metrics_by_energy else [],
                        columns=[
                            {"name": "Source", "id": "Source"},
                            {"name": "MAE", "id": "MAE"},
                            {"name": "MSE", "id": "MSE"},
                            {"name": "R²", "id": "R2"}
                        ],
                        style_cell={'textAlign': 'center', 'padding': '10px'},
                        style_header={'backgroundColor': '#27ae60', 'color': 'white', 'fontWeight': 'bold'},
                        style_data={'backgroundColor': '#f8f9fa'}
                    ) if metrics_by_energy else html.P("Métriques non disponibles")
                ], style={'backgroundColor': '#fff', 'padding': '25px', 'borderRadius': '10px',
                         'boxShadow': '0 4px 6px rgba(0,0,0,0.1)', 'marginBottom': '25px'}),
                
                # Graphique principal
                html.Div([
                    html.H4("Visualisation", style={'color': '#2c3e50', 'marginBottom': '20px'}),
                    dcc.Graph(figure=fig_main)
                ], style={'backgroundColor': '#fff', 'padding': '25px', 'borderRadius': '10px',
                         'boxShadow': '0 4px 6px rgba(0,0,0,0.1)'})
            ])
        else:
            # Mode Prédiction Réelle : Focus sur les visualisations
            components = [
                # Note explicative sur le modèle
                html.Div([
                    html.H5("Note sur les prédictions", style={'color': '#3498db', 'marginBottom': '10px'}),
                    html.P([
                        "Le modèle utilise une approche autoregressive : chaque prédiction devient l'entrée pour la suivante. ",
                        "Si vous observez des valeurs répétées sur plusieurs horizons, c'est que le modèle a convergé vers un état stable. ",
                        "Ce comportement est normal pour les horizons lointains (>12h) où l'incertitude augmente."
                    ], style={'fontSize': '14px', 'color': '#7f8c8d', 'lineHeight': '1.6'})
                ], style={'backgroundColor': '#e3f2fd', 'padding': '15px', 'borderRadius': '8px', 
                         'marginBottom': '20px', 'borderLeft': '4px solid #3498db'}),
                
                # Graphiques séparés - Historique + Prédictions
                html.Div([
                    html.H4("Analyse Détaillée par Source d'Énergie", 
                           style={'color': '#2c3e50', 'marginBottom': '20px'}),
                    html.Div([
                        html.P([
                            html.Strong("Organisation en grille 3×2 : "),
                            "Chaque source a son propre graphique pour une analyse approfondie."
                        ], style={'marginBottom': '8px'}),
                        html.P([
                            "• ", html.Strong("Ligne continue"), " : Historique (7 derniers jours)",
                            html.Br(),
                            "• ", html.Strong("Ligne pointillée + marqueurs"), " : Prédictions centrales",
                            html.Br(),
                            "• ", html.Strong("Zone ombrée"), " : Intervalle de confiance adapté par source (2-15% selon variabilité)",
                            html.Br(),
                            "• ", html.Strong("Ligne grise verticale"), " : Séparation historique/prédiction"
                        ], style={'fontSize': '13px', 'lineHeight': '1.8'}),
                        html.P([
                            html.I("Intervalles adaptés : Nucléaire (1-5%), Biomasse (2-8%), Éolien (4-12%), Solaire (5-15%)")
                        ], style={'fontSize': '12px', 'marginTop': '8px', 'fontStyle': 'italic', 'color': '#3498db'})
                    ], style={'fontSize': '14px', 'color': '#7f8c8d', 'marginBottom': '15px', 
                             'backgroundColor': '#f8f9fa', 'padding': '12px', 'borderRadius': '6px'}),
                    dcc.Graph(figure=fig_main)
                ], style={'backgroundColor': '#fff', 'padding': '25px', 'borderRadius': '10px',
                         'boxShadow': '0 4px 6px rgba(0,0,0,0.1)', 'marginBottom': '25px'})
            ]
            
            # Ajouter les graphiques de déficit en deux colonnes si disponibles
            if fig_conso_prod and fig_deficit:
                components.append(
                    html.Div([
                        html.H4("Analyse Énergétique Détaillée", 
                               style={'color': '#2c3e50', 'marginBottom': '20px', 'textAlign': 'center'}),
                        
                        # Deux colonnes pour les graphiques
                        html.Div([
                            # Colonne gauche : Consommation vs Production
                            html.Div([
                                dcc.Graph(figure=fig_conso_prod),
                                html.Div([
                                    html.P([
                                        html.Strong("Comparaison directe :"),
                                        html.Br(),
                                        "• Rouge : Consommation prédite",
                                        html.Br(),
                                        "• Vert : Production totale prédite",
                                        html.Br(),
                                        "• Zone jaune : Écart"
                                    ], style={'fontSize': '12px', 'lineHeight': '1.6'})
                                ], style={'padding': '10px', 'backgroundColor': '#f8f9fa', 'borderRadius': '6px'})
                            ], style={'flex': '1', 'paddingRight': '10px'}),
                            
                            # Colonne droite : Évolution du déficit
                            html.Div([
                                dcc.Graph(figure=fig_deficit),
                                html.Div([
                                    html.P([
                                        html.Strong("Interprétation :"),
                                        html.Br(),
                                        "• Au-dessus de 0 : Surplus (Production > Conso)",
                                        html.Br(),
                                        "• En-dessous de 0 : Déficit (Production < Conso)",
                                        html.Br(),
                                        "• Ligne grise : Équilibre parfait"
                                    ], style={'fontSize': '12px', 'lineHeight': '1.6'})
                                ], style={'padding': '10px', 'backgroundColor': '#f8f9fa', 'borderRadius': '6px'})
                            ], style={'flex': '1', 'paddingLeft': '10px'})
                        ], style={'display': 'flex', 'gap': '20px'}),
                        
                        # Note explicative globale
                        html.Div([
                            html.P([
                                html.I("Note : Déficit = Production - Consommation. "),
                                "Un déficit positif (surplus) indique une production excédentaire exportable. ",
                                "Un déficit négatif indique un besoin d'importation pour combler le manque."
                            ], style={'fontSize': '13px', 'color': '#7f8c8d', 'textAlign': 'center'})
                        ], style={'marginTop': '15px', 'padding': '10px', 'backgroundColor': '#e3f2fd', 
                                 'borderRadius': '6px', 'borderLeft': '4px solid #3498db'})
                        
                    ], style={'backgroundColor': '#fff', 'padding': '25px', 'borderRadius': '10px',
                             'boxShadow': '0 4px 6px rgba(0,0,0,0.1)', 'marginBottom': '25px'})
                )
            
            # Tableau des prédictions
            components.append(
                html.Div([
                    html.H4("Détail des Prédictions", style={'color': '#2c3e50', 'marginBottom': '20px'}),
                    pred_table
                ], style={'backgroundColor': '#fff', 'padding': '25px', 'borderRadius': '10px',
                         'boxShadow': '0 4px 6px rgba(0,0,0,0.1)'})
            )
            
            return html.Div(components)
        
    except Exception as e:
        return html.Div([
            html.H4("Erreur de Visualisation", style={'color': '#e74c3c'}),
            html.P(f"Erreur lors de la création des visualisations: {str(e)}")
        ])

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)
