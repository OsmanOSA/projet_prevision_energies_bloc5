# Image Airflow de la plateforme EnergIA : Airflow officiel + dépendances
# runtime du projet (collecte, persistance, prévision).
ARG AIRFLOW_VERSION=2.10.4
FROM apache/airflow:${AIRFLOW_VERSION}-python3.12

# libgomp1 : runtime OpenMP requis par LightGBM / XGBoost (absent de l'image slim)
USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
USER airflow

COPY docker/airflow/requirements-airflow.txt /requirements-airflow.txt
RUN pip install --no-cache-dir -r /requirements-airflow.txt
