from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from pipeline_prevision.components.data_ingestion import DataIngestion
from pipeline_prevision.components.data_transformation import DataTransformation
from pipeline_prevision.components.data_validation import DataValidation
from pipeline_prevision.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
)
from pipeline_prevision.utils.main_utils.feature_engineering import (
    ANCHOR_NAMES,
    DAYTYPE_SUFFIX,
    DEFAULT_ANCHOR,
    HORIZON_MAX,
    JOUR_CHOME,
    JOUR_OUVRE,
    add_target_features,
    anchor_values,
    build_daytype_anchor_columns,
    build_forecast_temperature_columns,
    build_origin_feature_frame,
    build_series_by_target,
    complementary_anchor,
    day_types,
    select_forecast_temperature,
    select_temperature,
)
from pipeline_prevision.utils.main_utils.utils import load_object, window_generator
from pipeline_prevision.utils.ml_utils.model.local_forecaster import (
    backtest_direct,
    forecast_origin,
    get_anchor,
    predict_with_conformal_intervals,
)


def frame(start: str, rows: int, offset: float = 0.0) -> pd.DataFrame:
    index = pd.date_range(start, periods=rows, freq="h", name="timestamp")
    base = np.arange(rows, dtype=float) + offset
    solar = np.maximum(0, base)
    biomass = 300 + base * 0.01
    wind_onshore = 2_000 + base
    nuclear = 40_000 + base
    return pd.DataFrame(
        {
            "temp": 10 + base * 0.01,
            # Température France pondérée (17 stations) : légèrement décalée de
            # `temp` pour que les tests distinguent laquelle des deux colonnes
            # les features consomment réellement (cf. select_temperature).
            "temp_fr": 10.4 + base * 0.01,
            # Couple Open-Meteo : observé et prévu J-1, sur la même grille. Chaque
            # série a son propre décalage, pour la même raison que ci-dessus — un
            # test qui confondrait deux sources de température passerait sans
            # rien prouver. `temp_fr_om` est la source d'origine préférée, et
            # `temp_fr_prev` alimente les features à l'heure cible.
            "temp_fr_om": 10.8 + base * 0.01,
            "temp_fr_prev": 11.3 + base * 0.01,
            "SOLAR": solar,
            "BIOMASS": biomass,
            "WIND_ONSHORE": wind_onshore,
            "NUCLEAR": nuclear,
            "consommation_totale": 50_000 + base,
            "production_total": solar + biomass + wind_onshore + nuclear,
        },
        index=index,
    )


def test_temporal_split_creates_three_distinct_ordered_files(tmp_path):
    config = SimpleNamespace(
        train_test_split_ratio=0.2,
        train_valid_split_ratio=0.1,
        training_file_path=str(tmp_path / "train.csv"),
        submission_file_path=str(tmp_path / "valid.csv"),
        testing_file_path=str(tmp_path / "test.csv"),
    )
    ingestion = DataIngestion(config)
    train, valid, test = ingestion.split_data_as_train_test_valid(
        frame("2024-01-01", 100)
    )

    assert (len(train), len(valid), len(test)) == (72, 8, 20)
    assert train.index.max() < valid.index.min() < test.index.min()
    assert Path(config.submission_file_path).read_bytes() != Path(
        config.testing_file_path
    ).read_bytes()


def test_temporal_split_rejects_duplicate_timestamps(tmp_path):
    data = frame("2024-01-01", 20)
    data.index = data.index[:-1].append(pd.DatetimeIndex([data.index[-2]]))
    config = SimpleNamespace(
        train_test_split_ratio=0.2,
        train_valid_split_ratio=0.1,
        training_file_path=str(tmp_path / "train.csv"),
        submission_file_path=str(tmp_path / "valid.csv"),
        testing_file_path=str(tmp_path / "test.csv"),
    )
    with pytest.raises(Exception, match="dupliqué"):
        DataIngestion(config).split_data_as_train_test_valid(data)


def validation_config(tmp_path):
    valid_dir = tmp_path / "validated"
    invalid_dir = tmp_path / "invalid"
    return SimpleNamespace(
        valid_train_file_path=str(valid_dir / "train.csv"),
        valid_submission_file_path=str(valid_dir / "valid.csv"),
        valid_test_file_path=str(valid_dir / "test.csv"),
        invalid_train_file_path=str(invalid_dir / "train.csv"),
        invalid_submission_file_path=str(invalid_dir / "valid.csv"),
        invalid_test_file_path=str(invalid_dir / "test.csv"),
        drift_report_file_path=str(tmp_path / "reports" / "quality.yaml"),
    )


def ingestion_artifact(tmp_path):
    train_path = tmp_path / "source_train.csv"
    valid_path = tmp_path / "source_valid.csv"
    test_path = tmp_path / "source_test.csv"
    frame("2024-01-01", 60).to_csv(train_path)
    frame("2024-01-03 12:00", 20, 60).to_csv(valid_path)
    frame("2024-01-04 08:00", 20, 80).to_csv(test_path)
    return DataIngestionArtifact(
        trained_file_path=str(train_path),
        submission_file_path=str(valid_path),
        test_file_path=str(test_path),
    )


def test_validation_enforces_schema_chronology_and_distinct_partitions(tmp_path):
    result = DataValidation(
        ingestion_artifact(tmp_path), validation_config(tmp_path)
    ).initiate_data_validation()

    assert result.validation_status is True
    assert Path(result.valid_train_file_path).exists()
    report = yaml.safe_load(Path(result.drift_report_file_path).read_text("utf-8"))
    assert report["validation_status"] is True
    assert not report["chronology_errors"]
    assert len(set(report["fingerprints"].values())) == 3


def test_validation_rejects_missing_column(tmp_path):
    artifact = ingestion_artifact(tmp_path)
    broken = pd.read_csv(artifact.submission_file_path).drop(columns=["NUCLEAR"])
    broken.to_csv(artifact.submission_file_path, index=False)

    with pytest.raises(Exception, match="colonnes manquantes"):
        DataValidation(
            artifact, validation_config(tmp_path)
        ).initiate_data_validation()


def test_window_generator_includes_last_eligible_target_without_future_leakage():
    values = np.arange(8, dtype=float).reshape(-1, 1)
    x_values, y_values = window_generator(values, lookback=3, horizon=2)

    assert x_values.shape == (4, 3, 1)
    assert y_values.shape == (4, 2, 1)
    np.testing.assert_array_equal(x_values[0, :, 0], [0, 1, 2])
    np.testing.assert_array_equal(y_values[0, :, 0], [3, 4])
    np.testing.assert_array_equal(y_values[-1, :, 0], [6, 7])


def test_transformation_purges_embargo_and_prevents_horizon_leakage(tmp_path, monkeypatch):
    """L'architecture actuelle (direct multi-horizon) n'utilise ni imputer ni
    scaler ajustés globalement (cf. pipeline_prevision/components/
    data_transformation.py) : le point critique à couvrir est l'embargo
    HORIZON_MAX entre partitions, qui empêche la cible target_h{HORIZON_MAX}
    d'une ligne de train/valid de déborder dans la partition suivante."""
    train_path = tmp_path / "train.csv"
    valid_path = tmp_path / "valid.csv"
    test_path = tmp_path / "test.csv"

    # Historique contigu suffisant : ~336h de préchauffe (plus grand lag) +
    # marge confortable pour que chaque partition purgée reste non vide.
    train = frame("2024-01-01", 1_000)
    valid = frame(train.index[-1] + pd.Timedelta(hours=1), 100)
    test = frame(valid.index[-1] + pd.Timedelta(hours=1), 100)
    train.to_csv(train_path)
    valid.to_csv(valid_path)
    test.to_csv(test_path)

    validation = DataValidationArtifact(
        validation_status=True,
        valid_train_file_path=str(train_path),
        valid_submission_file_path=str(valid_path),
        valid_test_file_path=str(test_path),
        invalid_train_file_path="",
        invalid_submission_file_path="",
        invalid_test_file_path="",
        drift_report_file_path="",
    )
    config = SimpleNamespace(
        transformed_object_file_path=str(tmp_path / "transform" / "bundle.pkl"),
        transformed_train_file_path=str(tmp_path / "transform" / "train.npy"),
        transformed_submission_file_path=str(tmp_path / "transform" / "valid.npy"),
        transformed_test_file_path=str(tmp_path / "transform" / "test.npy"),
    )
    monkeypatch.setenv("MODEL_OUTPUT_DIR", str(tmp_path / "models"))
    artifact = DataTransformation(validation, config).initiate_data_transformation()
    bundle = load_object(artifact.transformed_object_file_path)

    for target in ["consommation_totale", "SOLAR", "NUCLEAR"]:
        part = bundle[target]
        train_part, valid_part, test_part = part["train"], part["valid"], part["test"]

        assert len(train_part) > 0 and len(valid_part) > 0 and len(test_part) > 0
        assert train_part.index.max() < valid_part.index.min() < test_part.index.min()

        # Embargo : la cible la plus lointaine d'une ligne (t + HORIZON_MAX
        # heures) ne doit jamais atteindre la partition suivante.
        assert (valid_part.index.min() - train_part.index.max()) >= pd.Timedelta(hours=HORIZON_MAX)
        assert (test_part.index.min() - valid_part.index.max()) >= pd.Timedelta(hours=HORIZON_MAX)


DATASET = Path(__file__).resolve().parents[1] / "datasets" / "data.csv"
MODEL = Path(__file__).resolve().parents[1] / "final_models" / "model.pkl"


@pytest.mark.skipif(not (DATASET.is_file() and MODEL.is_file()),
                    reason="requires datasets/data.csv and final_models/model.pkl")
@pytest.mark.parametrize("target", ["SOLAR", "consommation_totale"])
def test_backtest_reproduces_the_live_forecast_of_the_same_origin(target):
    """Le backtest doit réattribuer à une origine EXACTEMENT la prévision qui
    aurait été émise en direct depuis cette origine.

    C'est la propriété qui rend honnête le backtest affiché au dashboard, et
    c'est celle que violait la correction de biais retirée : le chemin live
    (`_live_bias_correction`) et le chemin backtest
    (`_walkforward_bias_correction`) appliquaient deux corrections
    différentes, la seconde décalée d'une seule origine alors que l'erreur
    d'une origine n'est observable que `horizon` heures plus tard. Le backtest
    lisait donc du futur et surestimait la qualité du modèle.
    """
    observations = pd.read_csv(
        DATASET, sep=None, engine="python", parse_dates=["timestamp"], index_col="timestamp"
    ).sort_index().asfreq("h")

    # On coupe l'historique : la prévision « live » est émise depuis une
    # origine pour laquelle le réel des 24 heures suivantes existe malgré tout
    # dans le jeu complet -> le backtest peut être confronté au même point.
    truncated = observations.iloc[:-200]

    live = predict_with_conformal_intervals(truncated, target)
    origin = forecast_origin(live)

    for horizon in (1, 12, 24):
        replayed = backtest_direct(observations, target, horizon)
        target_ts = origin + pd.Timedelta(hours=horizon)
        assert target_ts in replayed.index

        expected = float(live.loc[target_ts, "y_pred"])
        assert replayed.loc[target_ts, "y_pred"] == pytest.approx(expected, rel=1e-9, abs=1e-6)


def test_seasonal_anchor_reads_the_target_hour_of_the_previous_day():
    """`seasonal_24` doit valoir la série 24 h avant l'heure CIBLE, pour tout
    horizon -- et rester causale (jamais postérieure à l'origine)."""
    data = frame("2024-01-01", 400)
    series = data["consommation_totale"]
    features, prefix = build_origin_feature_frame(
        build_series_by_target(data), data["temp"], "consommation_totale"
    )
    features = features.dropna()

    for horizon in (1, 6, 23, 24):
        values = anchor_values(features, horizon, prefix, "seasonal_24")
        expected = series.reindex(features.index + pd.Timedelta(hours=horizon - 24)).to_numpy()
        assert values == pytest.approx(expected)


def test_both_anchors_coincide_at_horizon_24():
    """À h=24 l'ancre saisonnière retombe sur l'origine : les deux ancres sont
    le même vecteur, le mélange y est donc dégénéré (cf. `complementary_anchor`)."""
    data = frame("2024-01-01", 400)
    features, prefix = build_origin_feature_frame(
        build_series_by_target(data), data["temp"], "consommation_totale"
    )
    features = features.dropna()

    assert anchor_values(features, 24, prefix, "seasonal_24") == pytest.approx(
        anchor_values(features, 24, prefix, "persistence")
    )
    assert anchor_values(features, 12, prefix, "seasonal_24") != pytest.approx(
        anchor_values(features, 12, prefix, "persistence")
    )


def test_composite_without_anchors_key_falls_back_to_persistence():
    """Un `model.pkl` entraîné avant la sélection d'ancre n'a pas la clé
    `anchors` : il doit rester ancré sur la persistance, donc produire
    exactement les mêmes prévisions qu'avant."""
    assert get_anchor({}, 1) == DEFAULT_ANCHOR == "persistence"
    assert get_anchor({"anchors": {}}, 12) == "persistence"
    assert get_anchor({"anchors": {12: "seasonal_24"}}, 12) == "seasonal_24"
    assert get_anchor({"anchors": {12: "seasonal_24"}}, 13) == "persistence"


def test_complementary_anchor_pairing():
    """L'appariement des ancres de mélange.

    Ce test vérifiait auparavant une INVOLUTION (`c(c(x)) == x`). C'était une
    propriété du monde à deux ancres : avec trois candidates, elle ne peut plus
    tenir qu'en appariant une ancre avec elle-même — ce qui rendrait le mélange
    dégénéré, puisque `seasonal_weight` pondérerait alors deux termes identiques.
    On verrouille donc ce qui a un sens fonctionnel :

    - le couple historique persistance <-> saisonnière 24 h est PRÉSERVÉ ;
    - le complément est toujours une ancre connue ;
    - une ancre n'est jamais son propre complément.
    """
    assert complementary_anchor("persistence") == "seasonal_24"
    assert complementary_anchor("seasonal_24") == "persistence"
    for name in ANCHOR_NAMES:
        autre = complementary_anchor(name)
        assert autre in ANCHOR_NAMES
        assert autre != name, f"{name} serait mélangée avec elle-même"


def test_decision_quantile_shifts_consumption_upward_only():
    """§ 9 : la prévision ENGAGÉE vise le quantile de coût, pas la médiane.

    Deux propriétés à verrouiller, parce qu'elles sont contre-intuitives et
    qu'une régression y serait silencieuse (`y_decision` resterait plausible) :

    1. sur la consommation, le quantile est > 0,5 -> `y_decision >= y_pred`,
       puisque sous-estimer coûte plus cher que surestimer ;
    2. sur les filières de PRODUCTION, aucune règle n'est définie -> AUCUNE
       retouche. Surtout pas q0,5 : la médiane des résidus signés n'étant pas
       nulle (+836 MW mesurés sur NUCLEAR à h+24), viser 0,5 réintroduirait la
       correction de niveau retirée de ce module après mesure, parce qu'elle
       perdait sur les cinq cibles.
    """
    from pipeline_prevision.utils.ml_utils.model.local_forecaster import (
        COST_OVER_FORECAST, COST_UNDER_FORECAST, decision_quantile,
    )

    q_conso = decision_quantile("consommation_totale")
    assert q_conso == pytest.approx(
        COST_UNDER_FORECAST / (COST_UNDER_FORECAST + COST_OVER_FORECAST))
    assert q_conso > 0.5, "sous-estimer la consommation doit coûter plus cher"

    for source in ("SOLAR", "BIOMASS", "WIND_ONSHORE", "NUCLEAR"):
        assert decision_quantile(source) is None, (
            "un quantile par défaut rétablirait une correction de biais mesurée perdante")

    production = predict_with_conformal_intervals(
        frame("2024-01-01", 900), "NUCLEAR", horizons=[1, 24])
    assert production["decision_q"].isna().all()
    assert production["y_decision"].to_numpy() == pytest.approx(
        production["y_pred"].to_numpy())

    data = frame("2024-01-01", 900)
    prediction = predict_with_conformal_intervals(
        data, "consommation_totale", horizons=[1, 6, 24])

    assert {"y_decision", "decision_q"} <= set(prediction.columns)
    assert prediction["decision_q"].to_numpy() == pytest.approx(q_conso)
    # Le décalage se lit sur le quantile des résidus signés : il ne peut pas
    # rendre la prévision engagée inférieure à la prévision ponctuelle.
    assert (prediction["y_decision"] >= prediction["y_pred"] - 1e-6).all()
    assert (prediction["y_decision"] >= 0).all()


# --- Température prévue à l'heure cible -------------------------------------
# Ces features sont la seule information dont RTE (J-1) disposait et nous pas :
# toutes nos features thermiques étaient gelées à l'origine. Ce qui est testé ici
# est d'abord la CAUSALITÉ (ne lire que la série de prévision, jamais l'observé
# futur) et la DISCIPLINE PAR HORIZON (chacun des 24 modèles ne voit que sa
# propre échéance).


def serie_prevision(start: str, rows: int) -> pd.Series:
    """Série de prévision volontairement différente de `temp`/`temp_fr` du
    `frame()` ci-dessus, pour que tout mélange de sources se voie."""
    index = pd.date_range(start, periods=rows, freq="h", name="timestamp")
    return pd.Series(100.0 + np.arange(rows, dtype=float), index=index)


def test_forecast_temperature_column_reads_the_target_hour():
    """`temp_prev_h{h}` à l'origine t doit valoir la prévision à t+h."""
    index = pd.date_range("2024-01-01", periods=50, freq="h")
    temp_prev = serie_prevision("2024-01-01", 80)   # dépasse l'index d'origine

    cols = build_forecast_temperature_columns(temp_prev, index)

    assert list(cols.columns) == [f"temp_prev_h{h}" for h in range(1, HORIZON_MAX + 1)]
    for h in (1, 7, 24):
        attendu = temp_prev.reindex(index + pd.Timedelta(hours=h)).to_numpy()
        assert cols[f"temp_prev_h{h}"].to_numpy() == pytest.approx(attendu)


def test_forecast_temperature_uses_values_beyond_the_observation_index():
    """Le cas de l'inférence : à l'origine la plus récente, les 24 horizons sont
    tous dans le futur. Un `reindex(index).shift(-h)` les perdrait tous et le
    forecaster reculerait indéfiniment d'origine."""
    index = pd.date_range("2024-01-01", periods=30, freq="h")
    # La prévision couvre l'index PLUS 24 h au-delà, comme l'API live.
    temp_prev = serie_prevision("2024-01-01", 30 + HORIZON_MAX)

    cols = build_forecast_temperature_columns(temp_prev, index)

    derniere = cols.iloc[-1]
    assert derniere.notna().all(), "la dernière origine doit être exploitable"
    for h in (1, 12, 24):
        assert derniere[f"temp_prev_h{h}"] == pytest.approx(
            temp_prev.loc[index[-1] + pd.Timedelta(hours=h)])


def test_add_target_features_keeps_only_the_current_horizon():
    """Chaque horizon a son modèle : lui montrer les 23 autres prévisions serait
    du bruit, et changerait la largeur de X selon rien."""
    data = frame("2024-01-01", 400)
    temp_prev = serie_prevision("2024-01-01", 400 + HORIZON_MAX)
    features, prefix = build_origin_feature_frame(
        build_series_by_target(data), data["temp"], "consommation_totale",
        temp_prev=temp_prev)
    features = features.dropna()

    for horizon in (1, 6, 24):
        X = add_target_features(features, horizon, f"{prefix}_delta_1")

        assert not [c for c in X.columns if c.startswith("temp_prev_h")]
        assert "temp_prev_target" in X.columns
        assert X["temp_prev_target"].to_numpy() == pytest.approx(
            features[f"temp_prev_h{horizon}"].to_numpy())


def test_forecast_temperature_delta_is_same_source_on_both_sides():
    """`temp_prev_delta` est l'écart prévu(cible) - observé(origine). C'est LE
    signal ; il doit être exactement cette différence, sans recalage caché."""
    data = frame("2024-01-01", 400)
    temp_prev = serie_prevision("2024-01-01", 400 + HORIZON_MAX)
    features, prefix = build_origin_feature_frame(
        build_series_by_target(data), data["temp"], "consommation_totale",
        temp_prev=temp_prev)
    features = features.dropna()

    X = add_target_features(features, 12, f"{prefix}_delta_1")

    assert X["temp_prev_delta"].to_numpy() == pytest.approx(
        (X["temp_prev_target"] - X["temp_0"]).to_numpy())
    # Degrés-jours à l'heure cible : non linéaires et complémentaires.
    assert (X["heating_degree_target"] >= 0).all()
    assert (X["cooling_degree_target"] >= 0).all()
    assert (X["heating_degree_target"] * X["cooling_degree_target"] == 0).all()


def test_frame_without_forecast_temperature_is_unchanged():
    """Rétro-compatibilité : sans prévision, le cadre doit être exactement celui
    d'avant l'introduction de ces features, pour que les champions archivés se
    rejouent à l'identique."""
    data = frame("2024-01-01", 400)
    series = build_series_by_target(data)

    sans, prefix = build_origin_feature_frame(series, data["temp"], "consommation_totale")
    avec, _ = build_origin_feature_frame(series, data["temp"], "consommation_totale",
                                         temp_prev=serie_prevision("2024-01-01", 430))

    assert not [c for c in sans.columns if c.startswith("temp_prev")]
    assert list(avec.columns)[:len(sans.columns)] == list(sans.columns)
    pd.testing.assert_frame_equal(avec[sans.columns], sans)

    # add_target_features doit être un no-op sur le cadre sans prévision.
    X = add_target_features(sans.dropna(), 6, f"{prefix}_delta_1")
    assert not [c for c in X.columns if c.startswith("temp_prev")]


def test_partial_forecast_frame_is_rejected():
    """Un cadre amputé d'une seule colonne d'horizon est une incohérence : mieux
    vaut échouer que prédire en silence sur la mauvaise échéance."""
    data = frame("2024-01-01", 400)
    features, prefix = build_origin_feature_frame(
        build_series_by_target(data), data["temp"], "consommation_totale",
        temp_prev=serie_prevision("2024-01-01", 430))
    ampute = features.dropna().drop(columns=["temp_prev_h6"])

    with pytest.raises(KeyError, match="temp_prev_h6"):
        add_target_features(ampute, 6, f"{prefix}_delta_1")


def test_select_temperature_prefers_the_open_meteo_grid():
    """`temp_fr_om` passe devant `temp_fr` pour la cohérence de source avec la
    prévision, et le repli reste en cascade jusqu'à `temp`."""
    data = frame("2024-01-01", 200)
    data["temp_fr_om"] = 20.0 + np.arange(len(data)) * 0.01

    assert select_temperature(data).to_numpy() == pytest.approx(
        data["temp_fr_om"].to_numpy())

    # Colonne présente mais quasi vide = cas dangereux d'une migration en cours.
    partiel = data.copy()
    partiel.loc[partiel.index[10:], "temp_fr_om"] = np.nan
    assert select_temperature(partiel).to_numpy() == pytest.approx(
        data["temp_fr"].to_numpy())

    sans_pondere = data.drop(columns=["temp_fr_om", "temp_fr"])
    assert select_temperature(sans_pondere).to_numpy() == pytest.approx(
        data["temp"].to_numpy())


def test_select_forecast_temperature_returns_none_rather_than_nans():
    """Retourner une série de NaN ferait tomber toutes les lignes au dropna : le
    pipeline « marcherait » sans plus rien apprendre. None désactive proprement
    les features."""
    data = frame("2024-01-01", 200)
    assert select_forecast_temperature(
        data.drop(columns=["temp_fr_prev"])) is None          # colonne absente

    data["temp_fr_prev"] = np.nan
    data.iloc[:5, data.columns.get_loc("temp_fr_prev")] = 12.0
    assert select_forecast_temperature(data) is None          # couverture insuffisante

    data["temp_fr_prev"] = 12.0 + np.arange(len(data)) * 0.01
    serie = select_forecast_temperature(data)
    assert serie is not None and serie.notna().all()


# --- Ancre « dernier jour comparable » --------------------------------------
# Le modèle apprend un RÉSIDU par rapport à l'ancre : une ancre fausse est donc
# un trou que le modèle doit combler, pas un simple décalage. L'ancre
# `seasonal_24` lit « hier », ce qui est faux un lundi (hier = dimanche) et un
# jour férié. Mesuré sur 3,5 ans : 5 766 MW d'erreur d'ancre le lundi, contre
# 2 630 avec le dernier jour ouvré.


def test_daytype_anchor_is_causal():
    """LE test qui compte : l'ancre ne doit jamais lire au-delà de l'origine.

    Une ancre qui fuiterait contaminerait la cible résiduelle elle-même — la
    fuite la plus difficile à détecter, puisqu'elle n'apparaît dans aucune
    feature. On encode la position dans la valeur pour la vérifier directement.
    """
    index = pd.date_range("2026-06-01", periods=24 * 70, freq="h", name="timestamp")
    series = pd.Series(np.arange(len(index), dtype=float), index=index)

    colonnes = build_daytype_anchor_columns(series, "conso")

    assert len(colonnes.columns) == HORIZON_MAX
    positions = np.arange(len(index))
    for horizon in (1, 6, 12, 24):
        valeurs = colonnes[f"conso{DAYTYPE_SUFFIX}{horizon}"].to_numpy()
        connues = ~np.isnan(valeurs)
        assert (valeurs[connues] <= positions[connues]).all(), (
            f"h{horizon} : l'ancre lit au-delà de l'origine")


def test_daytype_anchor_matches_hour_and_daytype():
    """La référence doit être à la MÊME HEURE et d'un type de jour comparable."""
    index = pd.date_range("2026-06-01", periods=24 * 70, freq="h", name="timestamp")
    series = pd.Series(np.arange(len(index), dtype=float), index=index)
    colonnes = build_daytype_anchor_columns(series, "conso")
    types_index = day_types(index)

    for horizon in (1, 12, 24):
        valeurs = colonnes[f"conso{DAYTYPE_SUFFIX}{horizon}"].to_numpy()
        cibles = index + pd.Timedelta(hours=horizon)
        types_cible = day_types(cibles)
        for i in np.flatnonzero(~np.isnan(valeurs)):
            source = int(valeurs[i])
            assert index[source].hour == cibles[i].hour
            if types_cible[i] == JOUR_CHOME:
                # Un dimanche/férié se compare au dernier jour chômé — souvent la
                # veille. Viser le dimanche précédent serait PIRE (3 349 contre
                # 2 104 mesurés) : sept jours d'écart coûtent plus que le
                # changement de type.
                assert types_index[source] != JOUR_OUVRE
            else:
                assert types_index[source] == types_cible[i]


def test_daytype_anchor_skips_sunday_for_a_monday_target():
    """Le cas qui motive l'ancre : un lundi doit se référer au dernier jour
    OUVRÉ (vendredi), jamais au dimanche que lit `seasonal_24`."""
    index = pd.date_range("2026-06-01", periods=24 * 40, freq="h", name="timestamp")
    series = pd.Series(np.arange(len(index), dtype=float), index=index)
    colonnes = build_daytype_anchor_columns(series, "conso")

    origines = [i for i, ts in enumerate(index)
                if ts.hour == 21 and (ts + pd.Timedelta(hours=24)).dayofweek == 0]
    assert origines, "aucune origine de veille de lundi dans l'échantillon"
    for i in origines[1:]:
        source = colonnes[f"conso{DAYTYPE_SUFFIX}24"].iloc[i]
        assert index[int(source)].dayofweek == 4, "la référence doit être un vendredi"


def test_daytype_anchor_reaches_add_target_features_once():
    """Chaque horizon ne doit voir QUE sa propre référence, jamais les 23 autres."""
    data = frame("2024-01-01", 400)
    features, prefix = build_origin_feature_frame(
        build_series_by_target(data), data["temp"], "consommation_totale")
    features = features.dropna()

    for horizon in (1, 6, 24):
        X = add_target_features(features, horizon, f"{prefix}_delta_1")
        assert not [c for c in X.columns if DAYTYPE_SUFFIX in c]
        assert "daytype_ref" in X.columns
        assert X["daytype_ref"].to_numpy() == pytest.approx(
            features[f"{prefix}{DAYTYPE_SUFFIX}{horizon}"].to_numpy())
        assert X["daytype_vs_origin"].to_numpy() == pytest.approx(
            (X["daytype_ref"] - X[f"{prefix}_0"]).to_numpy())


def test_daytype_anchor_values_match_the_column():
    """`anchor_values` doit servir exactement la colonne de son horizon."""
    data = frame("2024-01-01", 400)
    features, prefix = build_origin_feature_frame(
        build_series_by_target(data), data["temp"], "consommation_totale")
    features = features.dropna()

    for horizon in (1, 12, 24):
        assert anchor_values(features, horizon, prefix, "seasonal_daytype") == pytest.approx(
            features[f"{prefix}{DAYTYPE_SUFFIX}{horizon}"].to_numpy())

    with pytest.raises(KeyError):
        anchor_values(features.drop(columns=[f"{prefix}{DAYTYPE_SUFFIX}12"]),
                      12, prefix, "seasonal_daytype")
