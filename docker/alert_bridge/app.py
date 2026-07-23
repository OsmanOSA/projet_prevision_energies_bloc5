"""Pont d'alerte Grafana → Airflow.

Grafana émet son propre format de notification (`status`, `alerts[]`, `labels`…)
alors que l'API Airflow attend un corps `{"conf": {...}}`. Ce service traduit
l'un vers l'autre, journalise l'alerte reçue (traçabilité) et déclenche le DAG
de réentraînement uniquement quand l'alerte est en cours (`firing`).
"""

import logging
import os

import requests
from fastapi import FastAPI, Request

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("alert-bridge")

AIRFLOW_URL = os.getenv("AIRFLOW_API_URL", "http://airflow-webserver:8080")
DAG_ID = os.getenv("RETRAIN_DAG_ID", "retrain_on_degradation")
AIRFLOW_USER = os.getenv("AIRFLOW_USER", "admin")
AIRFLOW_PASSWORD = os.getenv("AIRFLOW_PASSWORD", "admin")

app = FastAPI(title="EnergIA — pont d'alerte Grafana → Airflow", version="1.0.0")


@app.get("/health")
def health():
    return {"status": "ok", "dag": DAG_ID, "airflow": AIRFLOW_URL}


@app.post("/grafana")
async def grafana_webhook(request: Request):
    """Reçoit une notification Grafana et déclenche le DAG si l'alerte est active."""
    payload = await request.json()
    status = payload.get("status", "unknown")
    alerts = payload.get("alerts", []) or []

    logger.info("Alerte Grafana reçue : status=%s, %d alerte(s)", status, len(alerts))

    if status != "firing":
        logger.info("Statut '%s' (pas 'firing') : aucun réentraînement déclenché.", status)
        return {"triggered": False, "reason": f"status={status}"}

    # Contexte transmis au DAG (visible dans l'UI Airflow et les logs).
    first = alerts[0] if alerts else {}
    conf = {
        "source": "grafana",
        "status": status,
        "alertname": (first.get("labels") or {}).get("alertname"),
        "summary": (first.get("annotations") or {}).get("summary"),
        "values": first.get("values"),
    }

    url = f"{AIRFLOW_URL}/api/v1/dags/{DAG_ID}/dagRuns"
    try:
        response = requests.post(
            url, json={"conf": conf},
            auth=(AIRFLOW_USER, AIRFLOW_PASSWORD),
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
    except Exception as exc:
        logger.exception("Appel Airflow en échec")
        return {"triggered": False, "error": str(exc)}

    ok = response.status_code in (200, 201)
    logger.info("Déclenchement %s du DAG %s (HTTP %s)",
                "réussi" if ok else "en échec", DAG_ID, response.status_code)
    return {"triggered": ok, "airflow_status": response.status_code,
            "detail": response.text[:300]}
