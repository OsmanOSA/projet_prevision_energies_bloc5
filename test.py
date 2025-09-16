from pipeline_prevision.utils.main_utils.utils import load_object, load_numpy_array_data, detect_per_day
import pandas as pd
from pipeline_prevision.utils.ml_utils.model.estimator import ForecastModel

preprocessor = load_object("final_models/preprocessor.pkl")
final_model = load_object("final_models/model.pkl")

test_df = pd.read_csv("Artifacts/18_06_2025_20_52_20/data_ingestion/ingested/test.csv", 
                      sep=None, engine="python", parse_dates=["Date"], index_col="Date")
print("Colonnes attendues :", preprocessor.feature_names_in_)
print("Colonnes reçues :", test_df.columns)
print(test_df.describe())
print(test_df.isnull().sum())
print("Nombre total de lignes :", len(test_df))

forecast_model = ForecastModel(preprocessor = preprocessor,
                                       model = final_model)

y_pred, y_test = forecast_model.predict(x=test_df)
