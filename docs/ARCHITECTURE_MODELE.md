# Architecture du modèle — `direct_multihorizon_residual`

Formalisation mathématique de la chaîne de prévision d'EnergIA, de
l'extraction des features à l'intervalle de confiance publié.

Ce document décrit **ce que le code fait**, pas une intention de conception.
Chaque équation renvoie à son implémentation.

| Composant | Fichier |
|---|---|
| Extraction des features (`Φ`) | `pipeline_prevision/utils/main_utils/feature_engineering.py` |
| Entraînement (θ, α, w) | `pipeline_prevision/components/model_trainer.py` |
| Inférence, intervalles conformes | `pipeline_prevision/utils/ml_utils/model/local_forecaster.py` |
| Garde-fou correction de biais (§6) | `scripts/validate_bias_params.py` |

---

## 1. Résumé en une équation

Pour une cible `s` et un horizon `h`, la prévision publiée s'écrit :

```
ŷ⁽ˢ⁾_{t+h}  =  max( 0 ,  (1 − w⁽ˢ⁾_h) · [ y⁽ˢ⁾_t + α⁽ˢ⁾_h · f⁽ˢ⁾_h( x⁽ˢ⁾_{t,h} ) ]
                        +      w⁽ˢ⁾_h  ·   y⁽ˢ⁾_{t+h−24}                        )

                          └───── expert 1 ─────┘      └─── expert 2 ───┘
                         persistance + résidu GBDT   persistance saisonnière
```

accompagnée de l'intervalle conforme `[ ŷ − q⁽ˢ⁾_h , ŷ + q⁽ˢ⁾_h ]`.

> **Note de lecture.** `α`, `w` et les paramètres de `f` sont tous *estimés*
> sur les données. Un quatrième étage a existé — une correction de biais en
> ligne `b(t ; λ, κ, c, W)`, seul composant du système dont les paramètres
> étaient fixés à la main. Il a été mesuré, puis **retiré** : voir §6.

C'est une **agrégation convexe statique de deux experts**, dont le premier
est une correction résiduelle apprise au-dessus d'une persistance naïve, le
tout enveloppé d'un intervalle calibré empiriquement.

---

## 2. Notation

| Symbole | Sens | Valeur / domaine |
|---|---|---|
| `t` | origine — instant de la dernière observation réelle | — |
| `h` | horizon de prévision | `1..24` (`HORIZON_MAX`) |
| `s` | cible modélisée | 5 cibles (§3) |
| `y⁽ˢ⁾_t` | valeur observée de `s` à l'instant `t` | GW |
| `x⁽ˢ⁾_{t,h}` | vecteur de features | ℝᵈ |
| `f⁽ˢ⁾_h` | modèle GBDT résiduel | LightGBM |
| `α⁽ˢ⁾_h` | gain appliqué au résidu prédit | `[0.70, 1.40]` |
| `w⁽ˢ⁾_h` | poids de l'expert saisonnier | `[0.00, 0.50]` |
| `q⁽ˢ⁾_h` | demi-largeur conforme | quantile 95 % |
| ~~`b⁽ˢ⁾_h(t)`~~ | correction de biais EWMA — **retirée** (§6) | — |

**Cibles modélisées** (`TARGET_PREFIXES`) — 5 cibles × 24 horizons =
**120 modèles LightGBM** :

```
S = { SOLAR , BIOMASS , WIND_ONSHORE , NUCLEAR , consommation_totale }
```

`production_total` n'est **pas** une cible entraînée : c'est une somme
a posteriori (§8).

---

## 3. Étage 1 — Extraction des features `Φ`

`Φ` est **entièrement déterministe** : aucun paramètre appris, aucun
`fit` sur les données d'entraînement. Il n'y a donc ni imputer ni scaler à
sérialiser, et aucune fuite possible par ce canal.

```
x⁽ˢ⁾_{t,h} = Φ( {y⁽ˢ⁾_{t−k}}_{k ∈ L_s} , {T_{t−k}}_{k ∈ L_T} ,
                {W_{t−k}}_{k ∈ L_T} , C_t , C_{t+h} )

  L_s = {1..24} ∪ {25, 48, 72, 168, 336}     lags de la cible
  L_T = {1, 2, 3, 6, 12, 24, 48, 168}        lags météo
  W   = (wspd, wdir, cldc)  —  uniquement si s = WIND_ONSHORE
```

### Familles de features

| Famille | Contenu |
|---|---|
| Autorégressif | lags `L_s` ; rolling {mean, std, min, max, median} sur {3, 6, 12, 24, 48, 168} ; deltas d'ordre 1–3, accélérations ; trends `(y_t − y_{t−k})/k` ; écarts `y_t − y_{t−24}`, `y_t − y_{t−168}` |
| Thermique | mêmes dérivées sur `T`, plus `heating_degree = max(0, 18 − T_t)`, `cooling_degree = max(0, T_t − 18)`, `T_t²`, et interactions `T_t × {sin h, cos h, weekend}` |
| Éolien (WIND_ONSHORE) | `wspd`, `wspd²`, `wspd³` (loi puissance-vitesse ∝ v³), `wspd³_{t−24}`, direction encodée en `(sin, cos)`, couverture nuageuse |
| Calendaire origine `C_t` | heure/jour encodés cycliquement, weekend, régimes `night` / `morning_ramp` / `midday_peak` |
| Calendaire cible `C_{t+h}` | **mêmes encodages appliqués à `t+h`** + interactions avec `delta_1` |

### La séparation `C_t` / `C_{t+h}`

C'est le point structurant de l'architecture. Toutes les grandeurs
*physiques* sont lues à l'origine `t` ; seul le **calendaire** est projeté à
l'heure cible `t+h`, ce qui est légitime car un calendrier est connu à
l'avance quel que soit `h`.

Conséquence : **aucune réinjection autorégressive**. Le modèle de l'horizon
24 ne consomme pas la sortie du modèle de l'horizon 23. Chaque horizon est
un problème d'apprentissage indépendant, et l'erreur ne se propage pas.

**Corollaire — voir aussi §11 :** aucune prévision météo n'entre dans le
modèle. Les features `T` et `W` sont indexées en `t−k`, jamais sur
`[t, t+h]`.

---

## 4. Étage 2 — Modèle résiduel GBDT

La cible d'apprentissage n'est pas la valeur future mais son **écart à la
persistance** :

```
r⁽ˢ⁾_{t,h} = y⁽ˢ⁾_{t+h} − y⁽ˢ⁾_t
```

Le modèle est un gradient boosting d'arbres :

```
f⁽ˢ⁾_h(x) = Σ_{m=1}^{M⁽ˢ⁾_h}  ν · g_m(x)        g_m = arbre de régression CART
```

estimé sous perte **L1** (`objective="regression_l1"`) avec régularisation
élastique sur les poids de feuilles :

```
f̂⁽ˢ⁾_h = argmin_f   Σ_t | r⁽ˢ⁾_{t,h} − f(x⁽ˢ⁾_{t,h}) |  +  λ₁‖f‖₁ + λ₂‖f‖₂²
```

`M⁽ˢ⁾_h` (nombre d'arbres) n'est pas un hyperparamètre libre : il est fixé
par early stopping sur la partition de validation (patience 150), puis le
modèle est **réentraîné sur `train ∪ valid`** avec `n_estimators = M⁽ˢ⁾_h`
figé. Cela récupère les données de validation sans risquer de re-sur-ajuster
le nombre d'arbres.

Le choix de la perte L1 aligne l'entraînement sur la métrique
d'évaluation (MAE) et rend l'ajustement robuste aux pics de production.

---

## 5. Étage 3 — Agrégation de deux experts

```
Ŷ⁽ˢ⁾_{t+h} = max( 0 , (1 − w⁽ˢ⁾_h)·[ y⁽ˢ⁾_t + α⁽ˢ⁾_h · f̂⁽ˢ⁾_h(x⁽ˢ⁾_{t,h}) ]
                    +      w⁽ˢ⁾_h ·   y⁽ˢ⁾_{t+h−24} )
```

| Expert | Définition | Rôle |
|---|---|---|
| 1 — persistance corrigée | `y_t + α · f̂(x)` | capte la dynamique récente |
| 2 — persistance saisonnière | `y_{t+h−24}`, soit `lag_{24−h}` | capte le cycle journalier quand la dynamique récente est peu informative |

**Rôle de `α`.** C'est un coefficient de *shrinkage* sur la correction
apprise. Si le GBDT sur-corrige systématiquement à un horizon donné, la
validation retient `α < 1` et amortit. C'est une recalibration a posteriori
de l'amplitude, que la perte L1 seule ne garantit pas.

**Rôle de `w`.** Poids d'expert, borné à 0,50 : le modèle appris ne peut
jamais être minoritaire dans le mélange. La borne est un garde-fou —
elle empêche l'optimiseur de dégénérer vers une pure persistance
saisonnière sur un horizon où le GBDT aurait mal appris.

Le `max(0, ·)` impose la contrainte physique de positivité (une production
ou une consommation ne peut être négative).

---

## 6. Correction de biais en ligne — *retirée après mesure*

Un quatrième étage a existé : une correction de niveau `b` estimée en ligne
sur le biais signé récent, ajoutée à `ŷ` avant persistance.

```
e_i      = y⁽ˢ⁾_i − ŷ⁽ˢ⁾_i                    erreurs signées de calibration
β_λ(t)   = (1 − λ) · Σ_{i≥0} λⁱ · e_{t−i}     biais EWMA (demi-vie ≈ 69 h)
τ_W(t)   = (1/W) · Σ_{i<W} | y_{t−i} |        échelle typique

b⁽ˢ⁾_h(t ; λ, κ, c, W) = clip( κ · β_λ(t) , −c·τ_W(t) , +c·τ_W(t) )
                         avec λ=0.99, κ=0.5, c=0.05, W=168
```

C'était le seul composant du système dont les paramètres étaient posés à la
main. Soumis au même traitement que `(α, w)` — grid search + backtest — il a
été **retiré**. `scripts/validate_bias_params.py` conserve la mesure et sert
de garde-fou avant toute réintroduction.

### 6.1 Le backtest lisait le futur

Deux variantes coexistaient, présentées comme équivalentes en causalité.
Elles ne l'étaient pas :

| Fonction | Décalage | Causale ? |
|---|---|---|
| `_live_bias_correction` | calibration arrêtée à `t − h` | ✅ oui |
| `_walkforward_bias_correction` | `.shift(1)` | ❌ **non, pour `h ≥ 2`** |

Ses séries sont indexées par **origine** (`series.shift(-h)`), donc `.shift(1)`
recule d'une origine, soit 1 h — alors que l'erreur de l'origine `i−k` n'est
observable qu'à `i−k+h`. La causalité exige `k ≥ h`, c'est-à-dire
`.shift(horizon)`. À `h = 24`, la correction lisait un réalisé arrivant 23 h
plus tard, ce qui gonflait `backtest_direct` (le graphe du dashboard) et les
résidus de calibration de §7.

Signature de la fuite — gain MAE en fonction de l'horizon, `consommation` :

| | h=1 | h=6 | h=12 | h=24 |
|---|---|---|---|---|
| `.shift(1)` (ancien) | −0,01 % | +1,32 % | +3,28 % | **+7,19 %** |
| `.shift(h)` (causal) | −0,01 % | +0,14 % | +0,56 % | **+1,87 %** |

À `h = 1` les deux décalages coïncident par construction, et les gains y sont
effectivement identiques. **C'est aussi le seul horizon où la correction était
prouvablement causale — et elle n'y apportait rien** (−0,01 % à −0,87 % selon
la cible). Toute sa valeur apparente venait des horizons où elle trichait.

### 6.2 Kill-test : échec sur les 5 cibles

Partition `test` coupée en deux dans le temps — sélection sur la première
moitié, confirmation sur la seconde, jamais regardée pendant la sélection.
Grille de 312 configurations `(λ, κ, c, W)` incluant `κ = 0` comme hypothèse
nulle. IC 95 % par bootstrap à blocs mobiles de 7 j sur le gain quotidien
apparié (les erreurs horaires sont trop autocorrélées pour un bootstrap
i.i.d.). Décalage causal `.shift(h)`.

| Cible | gain sélection | gain confirmation | IC 95 % gain/j | verdict |
|---|---|---|---|---|
| SOLAR | −4,57 % | −4,54 % | [−41,6 ; −27,0] | ❌ |
| BIOMASS | −4,27 % | −3,30 % | [−0,73 ; −0,12] | ❌ |
| WIND_ONSHORE | −1,24 % | −0,57 % | [−29,0 ; +9,0] | ❌ |
| NUCLEAR | +1,70 % | −1,13 % | [−41,0 ; +1,6] | ❌ |
| consommation | +2,51 % | −1,17 % | [−35,1 ; +11,2] | ❌ |

NUCLEAR et consommation illustrent exactement ce que la moitié de
confirmation devait attraper : positifs en sélection, négatifs en
confirmation. Sans elle, la correction aurait été « validée ». Le taux de
correction nuisible — fraction des points où `b` éloigne la prévision du réel
— vaut ~50 % sur toutes les cibles. Le grid search causal converge vers
`λ→0.995, κ→0.25, c→0.02` : « ne rien corriger » sur les quatre axes à la
fois.

Un biais de niveau qui persiste doit désormais remonter par
`retrain_on_degradation` (§11), pas être rustiné en ligne.

### 6.3 Ce qui n'a pas bougé

La couverture des intervalles conformes était soupçonnée d'être dégradée par
la fuite (les résidus de calibration étaient calculés sur des prévisions
corrigées par la variante fautive). **Mesure : non.** 93,85 % avec fuite
contre 93,86 % sans, sur SOLAR ; écarts du même ordre sur les autres cibles.
Le sous-couvrement de ~2 points par rapport au nominal 95 % existe
indépendamment et relève d'un autre sujet (non-échangeabilité des résidus).

---

## 7. Étage 4 — Intervalles conformes

Demi-largeur empirique, par horizon, sur les résidus absolus de calibration :

```
R_i      = | y⁽ˢ⁾_i − ŷ⁽ˢ⁾_i |
q⁽ˢ⁾_h   = Quantile_{1−α}( R_{n−200..n} )              α = 0.05

IC⁽ˢ⁾_{t+h} = [ max(0, ŷ⁽ˢ⁾_{t+h} − q⁽ˢ⁾_h) ,  ŷ⁽ˢ⁾_{t+h} + q⁽ˢ⁾_h ]
```

L'intervalle est donc **symétrique par construction mais dépendant de
l'horizon** : `q` est réestimé pour chaque `h`, ce qui produit
l'élargissement naturel avec l'horizon sans qu'aucune loi paramétrique ne
soit imposée. La fenêtre glissante de 200 origines le rend adaptatif : une
période récemment difficile élargit mécaniquement les bornes.

> ⚠️ **Couverture réelle mesurée : 92–94 %, pour un nominal de 95 %** (§6.3).
> L'écart tient à la non-échangeabilité des résidus, que la garantie conforme
> suppose. Les bornes sont donc légèrement optimistes.

Le retrait de la correction de biais (§6) fait apparaître un invariant
vérifié par les tests : `predict_direct`, `predict_with_conformal_intervals`
et `backtest_direct` renvoient désormais **la même** valeur pour une origine
donnée. Autrement dit, le backtest affiché réattribue exactement la prévision
qui aurait été émise en direct — ce qui était faux tant que les chemins live
et backtest appliquaient deux corrections différentes.

---

## 8. Agrégat dérivé

```
ŷ^{production_total}_{t+h} = Σ_{s ∈ {SOLAR, BIOMASS, WIND_ONSHORE, NUCLEAR}} ŷ⁽ˢ⁾_{t+h}
```

Les bornes sont sommées de la même façon. C'est une hypothèse conservatrice
(elle suppose les erreurs des quatre filières parfaitement corrélées) et
produit donc un intervalle **plus large** que ne le justifierait
l'indépendance — un choix assumé côté sûreté.

Motivation du découpage par filière : entraîner directement sur l'agrégat
noyait l'éolien — seule composante réellement volatile — sous des
composantes stables ou calendaires (nucléaire, biomasse).

---

## 9. Les deux niveaux d'apprentissage

### Niveau 1 — hyperparamètres `θ`, **un seul jeu par cible**

Optuna/TPE, 25 essais, `TimeSeriesSplit(n_splits=3, gap=24)`, pruner médian.
L'objectif est un **ratio à la persistance**, moyenné sur 5 horizons
représentatifs :

```
θ⁽ˢ⁾ = argmin_θ   (1/3) Σ_{k=1}^{3}  (1/5) Σ_{h ∈ {1,6,12,18,24}}
                       MAE_k( ŷ⁽ˢ⁾_{·,h} ; θ ) / MAE_k( y⁽ˢ⁾_{·} )
```

Deux décisions de conception ici :

- **Un ratio, pas une MAE absolue** — rend le score comparable entre cibles
  d'échelles très différentes (nucléaire ~50 GW vs solaire 0–10 GW) et
  ancre l'optimisation sur le gain réel apporté par le modèle.
- **`θ` partagé sur les 24 horizons** — tuner par horizon multiplierait le
  coût par 24. Seuls 5 horizons sont évalués par essai. C'est un compromis
  coût/finesse explicite, pas un oubli.

Espace de recherche : `learning_rate` (log, 0.005–0.05), `num_leaves`
(8–64), `max_depth` (4–9), `min_child_samples` (30–300), `subsample`,
`colsample_bytree` (0.60–1.00), `reg_alpha`, `reg_lambda` (log),
`min_split_gain`, `max_bin`.

### Niveau 2 — coefficients de mélange `(α, w)`, **par couple (s, h)**

Grid search exhaustif sur la partition de validation — 71 × 11 = 781
combinaisons par couple :

```
(α⁽ˢ⁾_h , w⁽ˢ⁾_h) = argmin_{α,w}  (1/|V|) Σ_{t ∈ V} | y⁽ˢ⁾_{t+h} − Ŷ⁽ˢ⁾_{t+h}(α,w) |

     α ∈ [0.70, 1.40]  pas 0.01
     w ∈ [0.00, 0.50]  pas 0.05
```

Le grid est exhaustif et non stochastique : le problème est en dimension 2
et le coût d'évaluation est négligeable (les prédictions résiduelles sont
déjà calculées), donc rien ne justifie une recherche approchée.

---

## 10. Protocole de validation

```
|<──── train ────>|═══|<─ valid ─>|═══|<─── test ───>|
                   24h             24h
                 embargo         embargo
```

| Élément | Choix | Raison |
|---|---|---|
| Découpage | strictement chronologique | aucun mélange aléatoire : les features sont autocorrélées à 336 h |
| Embargo | 24 h à chaque frontière | `HORIZON_MAX` : empêche une cible de chevaucher la partition suivante |
| CV interne | `TimeSeriesSplit(gap=24)` | même purge à l'intérieur du tuning |
| Valeurs manquantes | `dropna` après features | déterministe et identique entre partitions ; pas d'imputer ajusté |
| Test | jamais utilisé pour la sélection | ni `θ`, ni `α`, ni `w`, ni `M` |

**Double baseline**, reportée par cible et par horizon dans
`metadata.json` :

```
persistance simple      P⁽ˢ⁾_{t,h} = y⁽ˢ⁾_t
persistance saisonnière S⁽ˢ⁾_{t,h} = y⁽ˢ⁾_{t+h−24}
gain (%) = 100 · ( MAE(P) − MAE(modèle) ) / MAE(P)
```

Comparer aux deux est nécessaire : sur une série à cycle journalier fort,
battre la persistance simple est facile et ne prouve rien. La persistance
saisonnière est la baseline exigeante.

---

## 11. Propriétés et limites structurelles

**Aucune prévision météo n'est consommée.** Toutes les features `T` et `W`
sont indexées en `t−k`. Le modèle extrapole donc implicitement une météo
persistante sur `[t, t+h]`. À `h = 24` sur SOLAR et WIND_ONSHORE — dont la
production est physiquement pilotée par l'irradiance et le vent *futurs* —
c'est la contrainte dominante de l'architecture. Brancher un NWP est le
levier d'amélioration le mieux documenté dans la littérature.

**Aucun jour férié.** Le bloc calendaire encode heure, jour de semaine,
weekend et régimes horaires, mais pas les jours fériés ni les vacances
scolaires françaises. Sur la consommation nationale, c'est une source
d'erreur systématique connue et bon marché à corriger.

**Aucune adaptation en ligne.** `α` et `w` sont estimés une fois sur la
validation puis gelés jusqu'au prochain réentraînement, et la correction de
niveau `b` qui constituait le seul mécanisme adaptatif a été retirée (§6).
Entre deux entraînements, la seule quantité qui bouge avec les données
récentes est `q`, la demi-largeur de l'intervalle. La littérature de
référence sur la prévision de charge française (agrégation d'experts, EDF
R&D) fait évoluer le **poids** `w` en ligne selon la performance récente de
chaque expert — c'est précisément ce qui procure la robustesse aux ruptures
de régime, et c'est le bon endroit où porter l'adaptativité. Corriger un
niveau, comme le faisait `b`, ne réalloue rien entre experts : §6 mesure
que cette voie-là ne payait pas.

**Pas de sortie probabiliste native.** L'intervalle de §7 est un
post-traitement conforme, pas une distribution prédictive : le modèle
n'estime aucun quantile conditionnel aux features. Un même `q⁽ˢ⁾_h`
s'applique à une heure calme et à un pic.

**Pas de contrainte de cohérence inter-filières.** Les 5 cibles sont
entraînées indépendamment ; rien ne garantit la cohérence de leur somme
avec une contrainte de bilan (réconciliation hiérarchique).

---

## 12. Inventaire des paramètres

| Paramètre | Portée | Estimé sur | Nombre |
|---|---|---|---|
| `θ` (hyperparamètres LightGBM) | par cible | CV purgée (train+valid) | 5 |
| `M` (nombre d'arbres) | par (cible, horizon) | early stopping / valid | 120 |
| `f̂` (arbres) | par (cible, horizon) | train ∪ valid | 120 modèles |
| `α` (gain résiduel) | par (cible, horizon) | valid | 120 |
| `w` (poids saisonnier) | par (cible, horizon) | valid | 120 |
| `q` (demi-largeur conforme) | par (cible, horizon) | 200 dernières origines, à l'inférence | recalculé |

Les cinq premières lignes sont figées dans l'artefact `model.pkl` et hashées
en SHA-256 dans `metadata.json` ; `q` est recalculé à chaque prévision à
partir de l'historique en base.

**Tout paramètre du système provient désormais d'une procédure de sélection
explicite, mesurée sur une partition dédiée.** L'unique exception — les
quatre constantes `(λ, κ, c, W)` de la correction de biais — a été supprimée
avec elle (§6). C'était aussi la seule que `metadata.json` ne consignait
pas : deux versions du code avec des `κ` différents produisaient des
prévisions différentes sous un même hash de modèle. Ce trou de traçabilité
est refermé.

> Corollaire à préserver : toute reprise d'un paramètre non estimé doit
> passer par un protocole de kill-test comme celui de §6.2 — sélection et
> confirmation sur deux moitiés temporelles distinctes, hypothèse nulle
> « ne rien faire » présente dans la grille.

---

## 13. Généalogie de l'architecture

Cette conception se situe, dans la taxonomie de la littérature de prévision
de charge, comme une **prévision directe multi-horizon à cible résiduelle,
avec agrégation convexe statique de deux experts** :

- *directe* (par opposition à récursive) — un modèle par horizon, pas de
  rollout ; c'est le choix majoritaire pour les horizons courts, car il
  évite l'accumulation d'erreur ;
- *résiduelle* — apprendre `y_{t+h} − y_t` plutôt que `y_{t+h}` retire la
  composante de niveau, non stationnaire, et laisse au GBDT un signal
  centré ;
- *agrégation d'experts* — la forme `(1−w)·A + w·B` est la version statique
  et à deux experts de l'agrégation en ligne utilisée en production sur la
  charge française.

Les gains rapportés vs persistance figurent dans le `metadata.json` de
chaque artefact entraîné, par cible et par horizon.
