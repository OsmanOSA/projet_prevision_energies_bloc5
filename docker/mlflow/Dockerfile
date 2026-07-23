# Serveur de suivi MLflow (backend PostgreSQL + artefacts servis par le serveur).
# La version est alignée sur le client installé côté projet (mlflow 3.14.x)
# afin d'éviter tout écart de schéma du backend store.
FROM python:3.12-slim

RUN pip install --no-cache-dir "mlflow==3.14.0" "psycopg2-binary>=2.9"

EXPOSE 5000

# --serve-artifacts : les clients envoient/téléchargent les artefacts via le
# serveur (schéma mlflow-artifacts://), ce qui évite d'exiger un chemin partagé.
CMD ["sh", "-c", "mlflow server \
  --host 0.0.0.0 --port 5000 \
  --backend-store-uri \"$MLFLOW_BACKEND_STORE_URI\" \
  --serve-artifacts --artifacts-destination /mlartifacts"]
