import os

import pandas as pd

from src.anomaly_model import (
    load_isolation_forest,
    save_isolation_forest,
    score_isolation_forest,
    train_isolation_forest,
)
from src.config import MODELS_DIR
from src.energy_model import (
    load_energy_models,
    predict_expected_energy,
    save_energy_models,
    train_energy_models,
)
from src.explanations import recommendation_for
from src.features import build_features
from src.preprocessing import load_and_clean
from src.scoring import score_all


def run_pipeline(csv_path: str, models_dir: str = MODELS_DIR, retrain: bool = True) -> pd.DataFrame:
    df = load_and_clean(csv_path)
    df = build_features(df)

    energy_model_path = os.path.join(models_dir, "energy_model.pkl")
    iso_path = os.path.join(models_dir, "anomaly_model.pkl")
    scaler_path = os.path.join(models_dir, "scaler.pkl")

    if retrain or not os.path.exists(energy_model_path):
        energy_models = train_energy_models(df)
        os.makedirs(models_dir, exist_ok=True)
        save_energy_models(energy_models, energy_model_path)
    else:
        energy_models = load_energy_models(energy_model_path)

    df = predict_expected_energy(df, energy_models)

    if retrain or not os.path.exists(iso_path):
        iso, scaler = train_isolation_forest(df)
        save_isolation_forest(iso, scaler, iso_path, scaler_path)
    else:
        iso, scaler = load_isolation_forest(iso_path, scaler_path)

    df = score_isolation_forest(df, iso, scaler)
    df = score_all(df)

    df["recommendation"] = df.apply(recommendation_for, axis=1)
    return df


if __name__ == "__main__":
    result = run_pipeline(os.path.join("data", "chiller_data.csv"))
    print(result[["equipment_id", "timestamp", "severity", "ai_score"]].tail(20))
    print(f"\nTotal rows: {len(result)}")
    print(f"Flagged (Attention/Critical): {(result['severity'].isin(['Attention', 'Critical'])).sum()}")
