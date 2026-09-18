import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from src.config import EQUIPMENT_COL, MODEL_FEATURES, TARGET


def train_energy_models(df: pd.DataFrame) -> dict:
    models = {}
    for equipment in df[EQUIPMENT_COL].unique():
        subset = df[df[EQUIPMENT_COL] == equipment]
        model = HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.05,
            max_leaf_nodes=31,
            random_state=42,
        )
        model.fit(subset[MODEL_FEATURES], subset[TARGET])
        models[equipment] = model
    return models


def predict_expected_energy(df: pd.DataFrame, models: dict) -> pd.DataFrame:
    df = df.copy()
    df["expected_energy"] = 0.0
    for equipment, model in models.items():
        mask = df[EQUIPMENT_COL] == equipment
        df.loc[mask, "expected_energy"] = model.predict(df.loc[mask, MODEL_FEATURES])
    df["energy_residual"] = df[TARGET] - df["expected_energy"]
    df["energy_deviation_pct"] = (
        df["energy_residual"] / df["expected_energy"].replace(0, pd.NA)
    ) * 100
    return df


def save_energy_models(models: dict, path: str) -> None:
    joblib.dump(models, path)


def load_energy_models(path: str) -> dict:
    return joblib.load(path)
