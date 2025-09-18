FROM python:3.10-slim

# Définir répertoire de travail
WORKDIR /app

# Copier les fichiers
COPY . /app

# Installer dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Exposer le port (Heroku utilise PORT automatiquement, mais on met 8000 par défaut)
EXPOSE 8000

# Lancer FastAPI avec Uvicorn (Heroku définit $PORT)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
