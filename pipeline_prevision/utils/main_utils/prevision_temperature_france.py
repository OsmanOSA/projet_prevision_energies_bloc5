"""Prévision de température France pondérée — indice national prévu (Open-Meteo).

Pourquoi ce module
------------------
Le modèle ne disposait d'**aucune température future**. Toutes ses features
thermiques étaient gelées à l'instant d'origine (`temp_0`, ses lags et rollings,
et les croisements `temp_0 × target_hour_*`), la seule source étant Meteostat,
qui est de l'observé. RTE (J-1) s'appuie lui sur une prévision météo réelle.

Conséquence mesurée sur les origines ≥ 2026-07-23 : par météo stable
(|Δtemp| < 2 °C entre l'origine et la moyenne du jour cible) nous battions RTE de
+16 % en MAE ; par bascule thermique (≥ 2 °C) nous perdions de -15 %. La canicule
du 27-29/07/2026 a exposé cet angle mort. Ce module fournit l'information
manquante.

Pourquoi pondérée, et pas simplement Paris
------------------------------------------
Mesuré sur 17 sites, mars→juillet 2026 (`python -m
scripts.backfill_prevision_temperature --diagnostic` pour rejouer la mesure) :

|  échéance | RMSE par site | RMSE indice FR pondéré | facteur |
|-----------|---------------|------------------------|---------|
|  J-1      |    1,04 °C    |       **0,45 °C**      |  ×2,34  |
|  J-2      |    1,54 °C    |         0,58 °C        |  ×2,64  |
|  J-3      |    1,73 °C    |         0,71 °C        |  ×2,43  |

Les erreurs de prévision entre sites sont quasi indépendantes (corrélation
moyenne 0,076), donc l'agrégation pondérée les gomme d'un facteur ~2,3. À
l'inverse, substituer Paris à la France coûte 1,20 °C d'écart absolu moyen (et
jusqu'à 5,2 °C sur une heure de canicule : le pic du 29/07/2026 était largement
parisien). **La prévision pondérée est donc plus précise que l'erreur qu'on
commettrait en la remplaçant par un point unique** — la pondération n'est pas un
raffinement, c'est ce qui rend la prévision exploitable.

Absence de fuite
----------------
Piège classique : s'entraîner sur la température *observée* à l'heure cible. Le
modèle apprend alors une météo future parfaite, puis reçoit en production une
prévision à ~1 °C près, et déçoit. On s'entraîne donc sur de **vraies prévisions
passées**, via les variables `temperature_2m_previous_dayN` de l'API Historical
Forecast (archive des runs, disponible depuis 2023-01-01 — soit tout notre
historique). Leur erreur par site croît avec l'échéance (×1,66 de J-1 à J-3),
signature d'une vraie prévision et non d'une analyse déguisée.

`previous_dayN` n'existe PAS sur l'API live (vérifié : 0 valeur non nulle) :
- entraînement / backfill -> `archive_prevision_france()`, échéance contrôlée ;
- inférence en production   -> `prevision_france()`, prévision courante.
La prévision courante est *plus fraîche* que celle sur laquelle on s'entraîne
(run du jour vs run de la veille), donc l'écart joue en notre faveur. C'est le
sens prudent : jamais l'inverse.

Une seule source des deux côtés : pourquoi
------------------------------------------
Le biais entre cette prévision (grille Open-Meteo) et `temp_fr` (stations
Meteostat) vaut +0,40 °C en moyenne — inoffensif s'il était constant, un modèle à
arbres absorbant un décalage fixe. **Mesuré sur 12 mois (août 2025 → juillet
2026, n = 8 712 h), il ne l'est pas :**

| découpage                      | amplitude du biais |
|--------------------------------|--------------------|
| par heure UTC                  | 0,78 °C (+0,41 la nuit, -0,31 à 08 h, +0,47 à 18 h) |
| par niveau de température       | 0,63 °C (+0,15 en régime moyen, **+0,78 au-delà de 30 °C**, +0,57 sous 0 °C) |
| par mois                        | 0,38 °C |

Rapporté à l'écart-type de l'erreur de prévision elle-même (0,56 °C), la
variation du biais donne un ratio de 1,39 : elle dépasse le bruit qu'elle est
censée être. Le décalage se creuse dans les **extrêmes thermiques** — là où la
thermosensibilité décide — et porte un **cycle diurne** qui se confondrait avec
les features horaires. Mélanger les sources ferait donc apprendre l'écart
grille/station comme de la physique de consommation.

D'où la règle : le côté origine ET le côté cible viennent d'Open-Meteo
(`temp_fr_om` / `temp_fr_prev`). `temp_fr` (Meteostat) est conservée pour rejouer
la comparaison, cf. `select_temperature` dans `feature_engineering.py`. Le coût
de la bascule est faible : la grille Open-Meteo suit l'observé Meteostat à
+0,15 °C de biais et 0,50 °C de RMSE, corrélation 0,997.

Convention temporelle : index horaire **UTC naïf**, comme tout le pipeline.
"""

import logging
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

from pipeline_prevision.utils.main_utils.temperature_france import (
    POIDS_TOTAL, SEUIL_POIDS, VILLES, ponderer,
)

# Prévision courante (aucune clé requise en usage non commercial).
URL_LIVE = "https://api.open-meteo.com/v1/forecast"
# Archive des prévisions telles qu'émises (pour l'entraînement).
URL_ARCHIVE = "https://historical-forecast-api.open-meteo.com/v1/forecast"

VARIABLE = "temperature_2m"
# Début de couverture de l'archive des PRÉVISIONS (`temperature_2m_previous_dayN`),
# borné par requête réelle : vide au 2021-03-15, complète au 2021-03-25.
#
# Cette constante valait 2023-01-01, valeur retenue d'un premier sondage qui
# n'avait testé que le début de notre historique d'alors. Elle a silencieusement
# tronqué la première extension d'historique : 15 504 lignes RTE/Meteostat
# écrites, et 100 % de NaN sur les deux colonnes Open-Meteo. Une borne posée par
# commodité finit toujours par être prise pour une limite de la source.
#
# L'analyse (`temperature_2m`) remonte bien plus loin (>= 2016), mais c'est la
# prévision qui contraint : sans elle, `select_forecast_temperature` désactive
# les features de température cible.
ARCHIVE_DEBUT = datetime(2021, 3, 25)

ESSAIS = 3
ATTENTE_S = 3
# L'archive est découpée par année civile : une requête 17 points × 3,5 ans
# dépasse les garde-fous de l'API (et le quota gratuit compte au volume).
DELAI_ENTRE_LOTS_S = 1.0

# Tolérance d'appariement entre le point demandé et la maille renvoyée.
# Open-Meteo aligne sur sa grille (~1 à 11 km) : au-delà de 0,25° (~28 km),
# l'appariement est douteux et on préfère échouer.
TOLERANCE_DEG = 0.25


def _requete(url: str, params: dict, verbeux: bool = False) -> list[dict]:
    """Appel HTTP avec tentatives, normalisé en liste de blocs."""
    for essai in range(ESSAIS):
        try:
            r = requests.get(url, params=params, timeout=180)
            r.raise_for_status()
            charge = r.json()
            # L'API renvoie un objet seul pour une coordonnée, une liste sinon.
            return charge if isinstance(charge, list) else [charge]
        except Exception as err:
            if verbeux:
                print(f"    {type(err).__name__}, tentative {essai + 1}/{ESSAIS}")
            if essai == ESSAIS - 1:
                raise
            time.sleep(ATTENTE_S)
    raise AssertionError("inatteignable")


def _apparier(blocs: list[dict]) -> dict[str, dict]:
    """Associe chaque bloc renvoyé à sa ville, par proximité géographique.

    L'API renvoie une liste et **rien ne garantit contractuellement qu'elle suit
    l'ordre de la requête**. Plutôt que de supposer, on apparie chaque bloc à la
    ville la plus proche et on exige une bijection : une permutation silencieuse
    mélangerait les poids entre régions, ce qu'aucun test d'aval ne rattraperait.
    """
    if len(blocs) != len(VILLES):
        raise ValueError(f"{len(blocs)} blocs reçus pour {len(VILLES)} villes demandées")

    noms = list(VILLES)
    distances = np.array([
        [np.hypot(bloc["latitude"] - VILLES[nom].lat,
                  bloc["longitude"] - VILLES[nom].lon) for nom in noms]
        for bloc in blocs
    ])

    attribution: dict[str, dict] = {}
    for i, bloc in enumerate(blocs):
        j = int(distances[i].argmin())
        nom, ecart = noms[j], distances[i, j]
        if ecart > TOLERANCE_DEG:
            raise ValueError(
                f"bloc lat={bloc['latitude']} lon={bloc['longitude']} à {ecart:.3f}° "
                f"de {nom} (plus proche) : au-delà de la tolérance {TOLERANCE_DEG}°")
        if nom in attribution:
            raise ValueError(f"deux blocs appariés à {nom} : appariement ambigu")
        attribution[nom] = bloc

    manquantes = set(noms) - set(attribution)
    if manquantes:
        raise ValueError(f"villes sans bloc apparié : {sorted(manquantes)}")
    return attribution


def _table_large(blocs: list[dict], variable: str) -> pd.DataFrame:
    """Table large : index = heure UTC naïve, colonnes = noms de villes."""
    attribution = _apparier(blocs)
    colonnes = {}
    for nom, bloc in attribution.items():
        horaire = bloc.get("hourly") or {}
        if variable not in horaire:
            raise ValueError(f"variable `{variable}` absente de la réponse pour {nom}")
        colonnes[nom] = pd.Series(
            horaire[variable],
            index=pd.DatetimeIndex(horaire["time"]).tz_localize(None),
            dtype=float,
        )
    large = pd.DataFrame(colonnes)
    large.index.name = "timestamp"
    return large


def _coordonnees() -> tuple[str, str]:
    """Latitudes/longitudes des 17 points, en une paire de listes séparées par
    des virgules (forme multi-coordonnées de l'API, vérifiée sur les deux
    endpoints : 17 points en un seul appel)."""
    noms = list(VILLES)
    return (",".join(f"{VILLES[n].lat:.4f}" for n in noms),
            ",".join(f"{VILLES[n].lon:.4f}" for n in noms))


def _controler_couverture(indice: pd.Series, contexte: str) -> None:
    """Trace les heures où trop de poids manque pour que l'indice soit
    représentatif. Silencieux tant que tout va bien, comme côté Meteostat."""
    fraction = indice.attrs.get("fraction_poids")
    if fraction is None:
        return
    faibles = int((fraction < SEUIL_POIDS).sum())
    if faibles:
        logging.warning(
            "%s : %d heures sous %.0f %% du poids total (indice peu fiable sur "
            "ces heures)", contexte, faibles, 100 * SEUIL_POIDS)


def prevision_france(debut: datetime, fin: datetime, verbeux: bool = False
                     ) -> pd.Series:
    """Prévision courante de l'indice national pondéré, pour l'**inférence**.

    `debut`/`fin` sont des heures UTC naïves. La fenêtre peut mordre sur le
    passé récent : `past_days` est calculé pour la couvrir (l'API le plafonne à
    92 jours). Retourne une série horaire réindexée sur [debut, fin], donc les
    heures non couvertes sont explicitement NaN plutôt que silencieusement
    absentes.
    """
    # UTC naïf, convention du pipeline (`datetime.utcnow()` est déprécié en 3.12).
    maintenant = datetime.now(timezone.utc).replace(tzinfo=None)
    jours_passes = max(0, min(92, (maintenant.date() - debut.date()).days + 1))
    jours_futurs = max(1, min(16, (fin.date() - maintenant.date()).days + 1))

    lat, lon = _coordonnees()
    blocs = _requete(URL_LIVE, {
        "latitude": lat, "longitude": lon, "hourly": VARIABLE,
        "timezone": "GMT", "past_days": jours_passes,
        "forecast_days": jours_futurs,
    }, verbeux=verbeux)

    indice = ponderer(_table_large(blocs, VARIABLE))
    _controler_couverture(indice, "prévision live")

    indice = indice.reindex(pd.date_range(debut, fin, freq="h"))
    indice.name = "temp_fr_prev"
    return indice


def archive_prevision_france(debut: datetime, fin: datetime, lead_jours: int = 1,
                             verbeux: bool = False) -> pd.Series:
    """Prévision telle qu'émise `lead_jours` avant l'heure cible, pour
    l'**entraînement**. C'est la variante sans fuite.

    `lead_jours=1` correspond à nos origines : ~21-23 h UTC le jour J, cibles sur
    J+1. La prévision archivée à J-1 est légèrement plus ancienne que celle dont
    on disposera en production (run de la veille vs run du jour), ce qui rend
    l'entraînement prudent plutôt qu'optimiste.

    Découpé par année civile : une requête sur tout l'historique dépasse les
    garde-fous de l'API.
    """
    if not 1 <= lead_jours <= 7:
        raise ValueError(f"lead_jours doit être entre 1 et 7, reçu {lead_jours}")

    variable = f"{VARIABLE}_previous_day{lead_jours}"
    indice = _archive(variable, debut, fin, f"archive prévision J-{lead_jours}",
                      verbeux=verbeux)
    indice.name = "temp_fr_prev"
    return indice


def analyse_france(debut: datetime, fin: datetime, verbeux: bool = False
                   ) -> pd.Series:
    """Indice national **observé** selon Open-Meteo (côté origine des features).

    C'est le pendant de `temperature_france()` (Meteostat) sur la même grille que
    la prévision. Nécessaire pour que l'écart `temp_cible - temp_0` ne franchisse
    jamais une frontière de source : cf. « Une seule source des deux côtés » en
    tête de module — le biais grille/station varie de 0,78 °C selon l'heure et de
    0,63 °C selon le niveau de température, donc il ne peut pas être absorbé.
    """
    indice = _archive(VARIABLE, debut, fin, "analyse Open-Meteo", verbeux=verbeux)
    indice.name = "temp_fr_om"
    return indice


def _archive(variable: str, debut: datetime, fin: datetime, contexte: str,
             verbeux: bool = False) -> pd.Series:
    """Interroge l'archive pour une variable, découpée par année civile.

    Le découpage est nécessaire : une requête 17 points × 3,5 ans dépasse les
    garde-fous de l'API, et le quota gratuit se compte au volume.
    """
    if debut < ARCHIVE_DEBUT:
        logging.warning(
            "L'archive démarre au %s : la demande à partir du %s sera tronquée "
            "(les heures antérieures resteront NaN).",
            ARCHIVE_DEBUT.date(), debut.date())

    lat, lon = _coordonnees()
    morceaux = []
    for an in range(max(debut, ARCHIVE_DEBUT).year, fin.year + 1):
        d = max(debut, ARCHIVE_DEBUT, datetime(an, 1, 1))
        f = min(fin, datetime(an, 12, 31, 23))
        if d > f:
            continue
        if verbeux:
            print(f"  {contexte} {an} : {d.date()} -> {f.date()}")
        blocs = _requete(URL_ARCHIVE, {
            "latitude": lat, "longitude": lon, "hourly": variable,
            "timezone": "GMT",
            "start_date": d.strftime("%Y-%m-%d"),
            "end_date": f.strftime("%Y-%m-%d"),
        }, verbeux=verbeux)
        morceaux.append(_table_large(blocs, variable))
        time.sleep(DELAI_ENTRE_LOTS_S)

    if not morceaux:
        raise ValueError(f"Aucune donnée récupérée ({contexte})")

    large = pd.concat(morceaux)
    large = large[~large.index.duplicated(keep="last")].sort_index()

    indice = ponderer(large)
    _controler_couverture(indice, contexte)
    return indice.reindex(pd.date_range(debut, fin, freq="h"))


def diagnostic(debut: datetime, fin: datetime, leads: tuple[int, ...] = (1, 2, 3)
               ) -> pd.DataFrame:
    """Rejoue la mesure qui justifie la pondération : erreur par site contre
    erreur de l'indice agrégé, par échéance.

    Sert à re-vérifier qu'`previous_dayN` reste une vraie prévision (erreur par
    site ~1 °C à J-1, croissante avec l'échéance) et non une analyse. À relancer
    si Open-Meteo change ses modèles sous-jacents.
    """
    lat, lon = _coordonnees()
    variables = [VARIABLE] + [f"{VARIABLE}_previous_day{n}" for n in leads]
    blocs = _requete(URL_ARCHIVE, {
        "latitude": lat, "longitude": lon, "hourly": ",".join(variables),
        "timezone": "GMT",
        "start_date": debut.strftime("%Y-%m-%d"),
        "end_date": fin.strftime("%Y-%m-%d"),
    })

    reference = _table_large(blocs, VARIABLE)
    reference_indice = ponderer(reference)

    lignes = []
    for lead in leads:
        prevu = _table_large(blocs, f"{VARIABLE}_previous_day{lead}")
        ecart_site = prevu - reference
        rmse_site = np.sqrt((ecart_site ** 2).mean())

        indice = ponderer(prevu)
        commun = indice.notna() & reference_indice.notna()
        rmse_indice = float(np.sqrt(((indice - reference_indice)[commun] ** 2).mean()))

        lignes.append({
            "échéance": f"J-{lead}",
            "RMSE par site (°C)": round(float(rmse_site.mean()), 3),
            "RMSE indice FR (°C)": round(rmse_indice, 3),
            "facteur agrégation": round(float(rmse_site.mean()) / rmse_indice, 2),
            "corr. erreurs inter-sites": round(float(
                ecart_site.corr().values[~np.eye(len(ecart_site.columns), dtype=bool)].mean()), 3),
        })

    resume = pd.DataFrame(lignes)
    resume.attrs["sqrt_n_theorique"] = round(float(np.sqrt(len(VILLES))), 2)
    resume.attrs["poids_total"] = POIDS_TOTAL
    return resume
