# Pont d'alerte Grafana -> Airflow (service minimal).
FROM python:3.12-slim

WORKDIR /srv

RUN pip install --no-cache-dir "fastapi>=0.110" "uvicorn[standard]>=0.27" "requests>=2.31"

COPY docker/alert_bridge/app.py /srv/app.py

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
