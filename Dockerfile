FROM python:3.10-slim

# Définir répertoire de travail
WORKDIR /app

# Installer les dépendances système pour LightGBM
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copier les fichiers de l'application
COPY . /app

# Installer dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Exposer le port (Heroku utilise $PORT automatiquement)
EXPOSE 80

# Lancer FastAPI avec Uvicorn (Heroku définit $PORT)
CMD uvicorn app:app --host 0.0.0.0 --port $PORT
