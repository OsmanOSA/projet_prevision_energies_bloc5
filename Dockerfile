FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MALLOC_ARENA_MAX=2

WORKDIR /app
COPY requirements.txt .

# Installez uniquement ce qui est nécessaire aux wheels binaires
RUN apt-get update \
 && apt-get install -y --no-install-recommends awscli \
 && pip install --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r requirements.txt \
 && apt-get purge -y --auto-remove \
 && rm -rf /var/lib/apt/lists/*

COPY . .
CMD ["python3", "app.py"]