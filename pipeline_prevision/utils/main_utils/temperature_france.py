"""Température France pondérée — indice national à partir de 17 stations.

Pourquoi ce module
------------------
Le pipeline utilisait jusqu'ici la température d'**une seule station** (la plus
proche des coordonnées `LAT`/`LON` du `.env`, en pratique Paris-Montsouris). La
consommation électrique française est un agrégat national : sa thermosensibilité
dépend du froid *pondéré par la population*, pas du froid parisien. Une vague de
froid continentale sur le Grand Est ou une canicule méditerranéenne sont
invisibles pour une station unique.

Construction
------------
1. **17 villes couvrant les 12 régions métropolitaines.** Chaque station reçoit
   pour poids la population de la région qu'elle représente (partagée entre les
   villes d'une même région). C'est une approximation d'un découpage de Voronoï
   pondéré par la population — l'approche des indices de température nationaux.
   Pondérer par la population de l'aire urbaine *de chaque ville* serait un
   contresens : cela donnerait 45 % du poids à Paris parce qu'on n'échantillonne
   que 17 villes, et non parce que Paris pèse 45 % de la France.

2. **Une station principale et une station de repli par ville.** Le classement
   par distance de `stations.nearby` n'est pas fiable : à Lille il désigne une
   station couverte à 18 %, à Strasbourg une station vide. Les identifiants ont
   donc été **figés après mesure de la couverture réelle** sur 2023-2026, et
   restreints aux stations synoptiques Météo-France (`07xxx`).

3. **Repli débiaisé.** Plusieurs stations n'ont aucune donnée sur 2023 entier
   (Nantes, Strasbourg). Les trous sont comblés par la station de repli, corrigée
   du biais moyen mesuré sur les heures communes (une station d'altitude ou de
   bord de mer n'a pas le même niveau qu'une station urbaine).

4. **Renormalisation des poids heure par heure.** Si une station manque à une
   heure donnée, son poids est redistribué sur les stations présentes. Sans cela
   une panne de station décale mécaniquement l'indice national.

Convention temporelle : index horaire **UTC naïf**, identique au reste du
pipeline (cf. `extract_conso` / `extract_temperature`).
"""

import time
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from meteostat import Parameter, hourly

# meteostat 2.x refuse plus de 10 stations par requête et plus de 3 ans
# d'historique horaire. On découpe donc par lot ET par année civile — qui est
# aussi la granularité des fichiers sources côté Meteostat.
LOT_STATIONS = 8
ESSAIS = 3

# Fraction de poids disponible en dessous de laquelle une heure est jugée peu
# fiable (trop de stations manquantes pour que la moyenne soit représentative).
SEUIL_POIDS = 0.80


@dataclass(frozen=True)
class Ville:
    region: str
    poids: float          # millions d'habitants représentés
    station: str          # station principale (identifiant Meteostat)
    repli: str | None     # station de secours pour combler les trous
    nom_station: str
    # Coordonnées de la station principale, figées depuis `stations.meta()`.
    # Elles servent à interroger les API sur grille (cf.
    # `prevision_temperature_france.py`) au point de mesure exact, et non au
    # centre-ville. Écart moyen station/maille mesuré : 4,3 m d'altitude.
    lat: float
    lon: float


# Poids = population régionale (INSEE 2021, millions), répartie entre les villes
# d'une même région au prorata de leur poids démographique relatif.
VILLES: dict[str, Ville] = {
    "Paris":            Ville("Île-de-France",            12.32, "07156", "07149", "Paris-Montsouris",       48.8167,  2.3333),
    "Lyon":             Ville("Auvergne-Rhône-Alpes",      4.48, "07480", "07481", "Lyon / Bron",            45.7167,  4.9500),
    "Grenoble":         Ville("Auvergne-Rhône-Alpes",      2.03, "07487", "07486", "Grenoble / Le Versoud",  45.2172,  5.8486),
    "Clermont-Ferrand": Ville("Auvergne-Rhône-Alpes",      1.63, "07460", "07374", "Clermont-Ferrand",       45.7833,  3.1667),
    "Bordeaux":         Ville("Nouvelle-Aquitaine",        6.05, "07510", "07502", "Bordeaux / Mérignac",    44.8333, -0.7000),
    "Toulouse":         Ville("Occitanie",                 3.29, "07630", "07631", "Toulouse / Blagnac",     43.6333,  1.3667),
    "Montpellier":      Ville("Occitanie",                 2.70, "07643", "07645", "Montpellier",            43.5833,  3.9667),
    "Lille":            Ville("Hauts-de-France",           5.98, "07015", "07017", "Lille",                  50.5667,  3.1000),
    "Strasbourg":       Ville("Grand Est",                 3.05, "07190", "07186", "Strasbourg",             48.5500,  7.6333),
    "Nancy":            Ville("Grand Est",                 2.50, "07180", "07090", "Nancy / Essey",          48.6833,  6.2167),
    "Marseille":        Ville("Provence-Alpes-Côte d'Azur", 3.33, "07650", "07653", "Marseille / Marignane", 43.4500,  5.2333),
    "Nice":             Ville("Provence-Alpes-Côte d'Azur", 1.80, "07690", "07684", "Nice",                  43.6500,  7.2000),
    "Nantes":           Ville("Pays de la Loire",          3.86, "07222", "07306", "Nantes",                 47.1667, -1.6000),
    "Rennes":           Ville("Bretagne",                  3.41, "07130", "07125", "Rennes",                 48.0667, -1.7333),
    "Rouen":            Ville("Normandie",                 3.32, "07037", "07038", "Rouen",                  49.3833,  1.1833),
    "Dijon":            Ville("Bourgogne-Franche-Comté",   2.80, "07280", "07386", "Dijon",                  47.2667,  5.0833),
    "Tours":            Ville("Centre-Val de Loire",       2.57, "07240", "07235", "Tours",                  47.4500,  0.7167),
}

POIDS_TOTAL = sum(v.poids for v in VILLES.values())

# Poids indexés par nom de ville — forme directement utilisable par `ponderer`.
POIDS = pd.Series({nom: v.poids for nom, v in VILLES.items()})


def ponderer(par_ville: pd.DataFrame) -> pd.Series:
    """Agrège des températures par ville en indice national pondéré.

    **Point d'entrée unique de la pondération**, partagé par la série observée
    (Meteostat, ce module) et par la série prévue (Open-Meteo, cf.
    `prevision_temperature_france.py`). Les deux séries DOIVENT passer par ici :
    la feature qui porte le signal est l'écart `temp_cible - temp_0`, une
    différence entre futur prévu et passé observé. Deux définitions spatiales
    différentes y injecteraient un décalage systématique à la place du signal,
    et une marche artificielle dans les lags qui enjambent l'origine.

    Les poids manquants sont renormalisés heure par heure sur les villes
    réellement présentes : sans cela l'absence d'une ville décale mécaniquement
    le niveau de l'indice.
    """
    colonnes = [c for c in POIDS.index if c in par_ville.columns]
    par_ville = par_ville[colonnes]
    poids = POIDS[colonnes]

    poids_dispo = par_ville.notna().mul(poids, axis=1).sum(axis=1)
    somme = par_ville.fillna(0).mul(poids, axis=1).sum(axis=1)
    indice = (somme / poids_dispo).where(poids_dispo > 0)
    indice.attrs["fraction_poids"] = poids_dispo / POIDS_TOTAL
    return indice


def _identifiants() -> list[str]:
    ids = {v.station for v in VILLES.values()}
    ids |= {v.repli for v in VILLES.values() if v.repli}
    return sorted(ids)


def recuperer_stations(identifiants: list[str], debut: datetime,
                       fin: datetime, verbeux: bool = False) -> pd.DataFrame:
    """Températures horaires en table large (index = heure UTC, colonnes = stations).

    Découpe la requête par lot de stations et par année civile pour rester sous
    les garde-fous de meteostat 2.x, avec quelques tentatives en cas d'échec
    réseau ou mémoire.
    """
    morceaux = []
    for an in range(debut.year, fin.year + 1):
        d = max(debut, datetime(an, 1, 1))
        f = min(fin, datetime(an, 12, 31, 23))
        if d > f:
            continue
        for i in range(0, len(identifiants), LOT_STATIONS):
            lot = identifiants[i:i + LOT_STATIONS]
            for essai in range(ESSAIS):
                try:
                    bloc = hourly(lot, d, f, parameters=[Parameter.TEMP]).fetch()
                    if bloc is not None and not bloc.empty:
                        morceaux.append(bloc)
                    break
                except Exception as err:
                    if verbeux:
                        print(f"    {an} lot {i//LOT_STATIONS+1} : "
                              f"{type(err).__name__}, tentative {essai+1}/{ESSAIS}")
                    if essai == ESSAIS - 1:
                        raise
                    time.sleep(3)

    if not morceaux:
        raise ValueError("Aucune donnée de température récupérée")

    brut = pd.concat(morceaux)
    brut = brut[~brut.index.duplicated(keep="last")]
    large = brut["temp"].unstack("station").astype(float)
    large.index = pd.DatetimeIndex(large.index).tz_localize(None)
    large.index.name = "timestamp"
    return large.reindex(pd.date_range(debut, fin, freq="h"))


def _serie_ville(large: pd.DataFrame, ville: Ville) -> tuple[pd.Series, float, int]:
    """Série d'une ville : station principale, trous comblés par le repli débiaisé.

    Retourne (série, biais appliqué, nombre d'heures comblées).
    """
    principale = large[ville.station] if ville.station in large.columns else None
    if principale is None:
        principale = pd.Series(np.nan, index=large.index)

    if not ville.repli or ville.repli not in large.columns:
        return principale, 0.0, 0

    repli = large[ville.repli]
    commun = principale.notna() & repli.notna()
    # Sans heures communes, on ne sait pas recaler le repli : mieux vaut ne rien
    # combler que d'injecter le niveau d'une autre station tel quel.
    if commun.sum() < 24 * 30:
        return principale, 0.0, 0

    biais = float((principale[commun] - repli[commun]).mean())
    a_combler = principale.isna() & repli.notna()
    serie = principale.copy()
    serie[a_combler] = repli[a_combler] + biais
    return serie, biais, int(a_combler.sum())


def temperature_france(debut: datetime, fin: datetime, verbeux: bool = False
                       ) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """Indice de température France pondéré par la population régionale.

    Retourne :
      - `temp_fr` : série horaire (°C), index UTC naïf ;
      - `par_ville` : températures par ville, après comblement ;
      - `qualite` : rapport de qualité par ville (couverture, biais, comblements).
    """
    large = recuperer_stations(_identifiants(), debut, fin, verbeux=verbeux)

    colonnes, rapport = {}, []
    for nom, ville in VILLES.items():
        serie, biais, comble = _serie_ville(large, ville)
        colonnes[nom] = serie
        brute = (large[ville.station].notna().mean()
                 if ville.station in large.columns else 0.0)
        rapport.append({
            "ville": nom, "région": ville.region, "poids": ville.poids,
            "poids %": 100 * ville.poids / POIDS_TOTAL,
            "station": ville.station, "couverture brute %": 100 * brute,
            "heures comblées": comble, "biais repli °C": biais,
            "couverture finale %": 100 * serie.notna().mean(),
        })

    par_ville = pd.DataFrame(colonnes)

    # Renormalisation heure par heure sur les stations réellement disponibles.
    temp_fr = ponderer(par_ville)
    temp_fr.name = "temp_fr"
    fraction_poids = temp_fr.attrs["fraction_poids"]

    qualite = pd.DataFrame(rapport).sort_values("poids", ascending=False)
    qualite.attrs["fraction_poids"] = fraction_poids
    qualite.attrs["heures_faibles"] = int((fraction_poids < SEUIL_POIDS).sum())
    return temp_fr, par_ville, qualite
