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
# Configuration pour intégration avec FastAPI
app = dash.Dash(__name__, 
                suppress_callback_exceptions=True,
                requests_pathname_prefix="/dashboard/")

# Styles CSS
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
    "fontSize": "15px"
}

CONTENT_STYLE = {
    "marginLeft": "19.5rem",
    "marginRight": "2rem",
    "padding": "2rem 1rem",
}

# Sidebar avec navigation en haut et configuration en bas
sidebar = html.Div([
    html.H2("Energy Forecast", className="display-6", 
           style={'textAlign': 'center', 'marginBottom': '30px', 'color': '#ecf0f1'}),
    
    # Navigation en haut
    html.Div(id='sidebar-nav'),
    
    html.Hr(style={'borderColor': '#34495e', 'margin': '20px 0'}),
    
    # Configuration des données en bas
    html.Div([
        html.H4("Configuration des Données", 
               style={'color': '#ecf0f1', 'fontSize': '16px', 'marginBottom': '15px'}),
        
        html.Div([
            html.Label("Début:", 
                      style={'color': '#bdc3c7', 'fontSize': '14px', 'marginBottom': '5px', 'display': 'block'}),
            dcc.DatePickerSingle(
                id='start-date-picker',
                date='2024-01-01',
                display_format='YYYY-MM-DD',
                style={'width': '100%', 'fontSize': '13px'}
            )
        ], style={'marginBottom': '12px'}),
        
        html.Div([
            html.Label("Fin:", 
                      style={'color': '#bdc3c7', 'fontSize': '14px', 'marginBottom': '5px', 'display': 'block'}),
            dcc.DatePickerSingle(
                id='end-date-picker',
                date='2024-01-07',
                display_format='YYYY-MM-DD',
                style={'width': '100%', 'fontSize': '13px'}
            )
        ], style={'marginBottom': '15px'}),
        
        html.Button(
            "Charger les Données",
            id="load-button",
            n_clicks=0,
            style={
                'backgroundColor': '#3498db',
                'color': 'white',
                'border': 'none',
                'padding': '8px 16px',
                'borderRadius': '5px',
                'cursor': 'pointer',
                'width': '100%',
                'fontSize': '14px',
                'marginBottom': '15px'
            }
        ),
        
        html.Div(id='status-div-sidebar')
    ])
], style=SIDEBAR_STYLE)

# Layout principal
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    sidebar,
    html.Div(id='page-content', style=CONTENT_STYLE),
    dcc.Store(id='data-store', data={}),
])

# Navigation
nav_links = [
    {"label": "🏠 Accueil", "href": "/dashboard/"},
    {"label": "📊 Vue d'ensemble", "href": "/dashboard/overview"},
    {"label": "⚡ Analyse Consommation", "href": "/dashboard/consumption"},
    {"label": "🔋 Analyse Production", "href": "/dashboard/production"},
    {"label": "🔮 Prévision", "href": "/dashboard/prediction"}
]

# Caches globaux
_graph_cache = {}
_processed_data_cache = {}

# Fonctions utilitaires
def prepare_dataframe(stored_data):
    """Préparer le DataFrame à partir des données stockées"""
    if not stored_data:
        return None
    
    try:
        df = pd.DataFrame(stored_data)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
        return df
    except Exception as e:
        print(f"Erreur préparation DataFrame: {e}")
        return None

def create_home_page():
    """Page d'accueil"""
    return html.Div([
        html.H1("🌟 Tableau de Bord - Prévision Énergétique", 
               style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '30px'}),
        
        html.Div([
            html.H3("🎯 Application Intégrée", style={'color': '#3498db', 'marginBottom': '15px'}),
            html.P([
                "Cette application combine votre API FastAPI et votre Dashboard Dash en une seule solution intégrée. ",
                "Accédez aux fonctionnalités via la navigation de gauche."
            ], style={'fontSize': '16px', 'lineHeight': '1.6'}),
            
            html.Div([
                html.A("📊 API Documentation", href="/docs", target="_blank",
                      style={'backgroundColor': '#3498db', 'color': 'white', 'padding': '10px 20px',
                            'textDecoration': 'none', 'borderRadius': '5px', 'marginRight': '15px'}),
                html.A("🔮 Dashboard", href="/dashboard/overview",
                      style={'backgroundColor': '#27ae60', 'color': 'white', 'padding': '10px 20px',
                            'textDecoration': 'none', 'borderRadius': '5px'})
            ], style={'textAlign': 'center', 'marginTop': '20px'})
        ], style={
            'backgroundColor': '#fff', 'padding': '30px', 'borderRadius': '10px',
            'boxShadow': '0 4px 6px rgba(0,0,0,0.1)', 'maxWidth': '600px', 'margin': '0 auto'
        })
    ])

def create_overview_page():
    """Page vue d'ensemble simplifiée"""
    return html.Div([
        html.H1("📊 Vue d'ensemble", style={'color': '#2c3e50', 'marginBottom': '30px'}),
        html.Div(id='overview-content', children=[
            html.P("Chargez des données pour voir les analyses.", 
                  style={'textAlign': 'center', 'color': '#7f8c8d', 'fontSize': '16px'})
        ])
    ])

def create_prediction_page():
    """Page de prévision"""
    return html.Div([
        html.H1("🔮 Prévisions Énergétiques", style={'color': '#2c3e50', 'marginBottom': '30px'}),
        
        html.Div([
            html.H3("Configuration des Prévisions", style={'color': '#34495e', 'marginBottom': '20px'}),
            
            html.Div([
                html.Label("Horizon de prévision (heures):", style={'fontWeight': 'bold', 'marginBottom': '10px'}),
                dcc.Dropdown(
                    id='prediction-horizon',
                    options=[
                        {'label': '6 heures', 'value': 6},
                        {'label': '12 heures', 'value': 12},
                        {'label': '24 heures', 'value': 24},
                        {'label': '48 heures', 'value': 48}
                    ],
                    value=24,
                    style={'width': '200px', 'marginBottom': '20px'}
                )
            ]),
            
            html.Button(
                "Générer Prévision",
                id="predict-button",
                n_clicks=0,
                style={
                    'backgroundColor': '#3498db',
                    'color': 'white',
                    'border': 'none',
                    'padding': '12px 30px',
                    'borderRadius': '5px',
                    'cursor': 'pointer',
                    'fontSize': '16px'
                }
            )
        ], style={
            'backgroundColor': '#fff', 'padding': '25px', 'borderRadius': '10px',
            'boxShadow': '0 4px 6px rgba(0,0,0,0.1)', 'marginBottom': '30px'
        }),
        
        html.Div(id='prediction-results')
    ])

# Callbacks
@app.callback([Output('page-content', 'children'),
               Output('sidebar-nav', 'children')],
              [Input('url', 'pathname')])
def display_page(pathname):
    # Navigation
    nav_elements = []
    for link in nav_links:
        is_active = (pathname == link['href']) or (pathname == '/dashboard' and link['href'] == '/dashboard/')
        style = {
            'display': 'block', 'padding': '12px 15px', 'color': '#ecf0f1',
            'textDecoration': 'none', 'borderRadius': '5px', 'marginBottom': '8px',
            'fontSize': '15px', 'transition': 'all 0.3s ease'
        }
        if is_active:
            style.update({'backgroundColor': '#3498db', 'color': 'white', 'fontWeight': 'bold'})
        else:
            style.update({'backgroundColor': 'transparent'})
        
        nav_elements.append(
            html.A(link['label'], href=link['href'], style=style)
        )
    
    # Contenu de la page
    if pathname == '/dashboard/overview':
        page_content = create_overview_page()
    elif pathname == '/dashboard/consumption':
        page_content = html.Div([html.H1("⚡ Analyse Consommation"), html.P("Page en développement...")])
    elif pathname == '/dashboard/production':
        page_content = html.Div([html.H1("🔋 Analyse Production"), html.P("Page en développement...")])
    elif pathname == '/dashboard/prediction':
        page_content = create_prediction_page()
    else:  # Default to home
        page_content = create_home_page()
    
    return page_content, nav_elements

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
        # Convertir les dates
        start_str = start_date.split('T')[0] if isinstance(start_date, str) else str(start_date)
        end_str = end_date.split('T')[0] if isinstance(end_date, str) else str(end_date)
        
        # Charger les données
        df = concat_all_data(start_str, end_str)
        data_dict = df.reset_index().to_dict('records')
        
        status_message = html.Div([
            html.I(className="fas fa-check-circle", style={'color': '#27ae60', 'marginRight': '5px'}),
            html.Small(f"Succès - {len(data_dict)} enregistrements", style={'color': '#27ae60'})
        ], style={'backgroundColor': 'rgba(39, 174, 96, 0.2)', 'padding': '8px', 'borderRadius': '4px'})
        
        return data_dict, status_message
        
    except Exception as e:
        error_message = html.Div([
            html.I(className="fas fa-exclamation-triangle", style={'color': '#e74c3c', 'marginRight': '5px'}),
            html.Small(f"Erreur: {str(e)}", style={'color': '#e74c3c'})
        ], style={'backgroundColor': 'rgba(231, 76, 60, 0.2)', 'padding': '8px', 'borderRadius': '4px'})
        
        return {}, error_message

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
            html.P("Chargez des données et cliquez sur 'Générer Prévision'.",
                  style={'textAlign': 'center', 'color': '#7f8c8d', 'fontSize': '16px'})
        ], style={'backgroundColor': '#fff', 'padding': '25px', 'borderRadius': '10px',
                 'boxShadow': '0 4px 6px rgba(0,0,0,0.1)'})
    
    try:
        df = prepare_dataframe(stored_data)
        if df is None:
            return html.Div([html.H4("Erreur"), html.P("Aucune donnée disponible.")])
        
        # Utiliser les modèles locaux pour la prédiction
        features = ['BIOMASS', 'NUCLEAR', 'SOLAR', 'WIND_ONSHORE', 'consommation_totale', 'temp']
        
        # Vérifier les features
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            return html.Div([
                html.H4("Erreur", style={'color': '#e74c3c'}),
                html.P(f"Features manquantes: {missing_features}")
            ])
        
        # Charger les modèles
        model = load_object("final_models/model.pkl")
        preprocessor = load_object("final_models/preprocessor.pkl")
        
        if model is None or preprocessor is None:
            return html.Div([
                html.H4("Erreur", style={'color': '#e74c3c'}),
                html.P("Impossible de charger les modèles.")
            ])
        
        # Prédiction
        forecast_model = ForecastModel(preprocessor=preprocessor, model=model)
        df_features = df[features].tail(36).copy()
        y_pred, y_test = forecast_model.predict_multistep(x=df_features, n_futur=horizon)
        
        # Affichage simple des résultats
        return html.Div([
            html.H4("Résultats des Prévisions", style={'color': '#2c3e50', 'marginBottom': '20px'}),
            html.P(f"Prédiction générée pour {horizon} heures", style={'color': '#27ae60'}),
            html.P(f"Shape des prédictions: {y_pred.shape}", style={'color': '#7f8c8d'}),
            
            # Tableau simple des prédictions
            html.Div([
                html.H5("Premières Prédictions", style={'marginBottom': '15px'}),
                dash_table.DataTable(
                    data=pd.DataFrame(y_pred[:min(10, len(y_pred))], columns=features).round(2).to_dict('records'),
                    columns=[{"name": col, "id": col} for col in features],
                    style_cell={'textAlign': 'center'},
                    style_header={'backgroundColor': '#3498db', 'color': 'white'}
                )
            ])
        ], style={'backgroundColor': '#fff', 'padding': '25px', 'borderRadius': '10px',
                 'boxShadow': '0 4px 6px rgba(0,0,0,0.1)'})
        
    except Exception as e:
        return html.Div([
            html.H4("Erreur", style={'color': '#e74c3c'}),
            html.P(f"Erreur lors de la génération: {str(e)}")
        ])

# Ne pas exécuter directement si importé
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8050))
    app.run_server(debug=False, host='0.0.0.0', port=port)
