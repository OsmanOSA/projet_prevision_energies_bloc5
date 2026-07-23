# Image de l'app Streamlit EnergIA (KPI en lecture sur PostgreSQL).
FROM python:3.12-slim

WORKDIR /app

# libgomp1 : runtime OpenMP requis par LightGBM / XGBoost pour désérialiser et
# exécuter le modèle (backtesting H+1 glissant calculé dans l'app).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY docker/streamlit/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY . /app

EXPOSE 8501

# Le code projet est monté sur /app (volume) ; PYTHONPATH=/app le rend importable.
CMD ["streamlit", "run", "streamlit_app/app_main.py", \
     "--server.port=8501", "--server.address=0.0.0.0"]
