FROM python:3.12-slim

WORKDIR /app

RUN apt-get update     && apt-get install -y --no-install-recommends build-essential libgomp1     && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

ENV PORT=8000
EXPOSE 8000

# API FastAPI historique. Les endpoints de prédiction exigent final_models/
# (volume ou artefact restauré par le déploiement).
CMD ["/bin/sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
