import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from src.config import ANOMALY_FEATURES


def train_isolation_forest(df: pd.DataFrame):
    scaler = StandardScaler()
    X = scaler.fit_transform(df[ANOMALY_FEATURES])

    iso = IsolationForest(n_estimators=300, contamination="auto", random_state=42)
    iso.fit(X)
    return iso, scaler


def score_isolation_forest(df: pd.DataFrame, iso, scaler) -> pd.DataFrame:
    df = df.copy()
    X = scaler.transform(df[ANOMALY_FEATURES])
    df["iso_prediction"] = iso.predict(X)
    raw_score = -iso.score_samples(X)
    df["iso_score_raw"] = raw_score
    df["iso_score"] = (
        (raw_score - raw_score.min()) / (raw_score.max() - raw_score.min() + 1e-9)
    ) * 100
    return df


def save_isolation_forest(iso, scaler, iso_path: str, scaler_path: str) -> None:
    joblib.dump(iso, iso_path)
    joblib.dump(scaler, scaler_path)


def load_isolation_forest(iso_path: str, scaler_path: str):
    return joblib.load(iso_path), joblib.load(scaler_path)
