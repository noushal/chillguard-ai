import pandas as pd

from src.config import (
    CONTEXT_ANOMALY_QUANTILE,
    EQUIPMENT_COL,
    PERSISTENCE_WINDOW,
    SCORE_WEIGHTS,
    SEVERITY_BINS,
)


def add_context_score(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    abs_resid = df["energy_residual"].abs()
    df["context_score"] = (
        (abs_resid - abs_resid.min()) / (abs_resid.max() - abs_resid.min() + 1e-9)
    ) * 100

    threshold = abs_resid.quantile(CONTEXT_ANOMALY_QUANTILE)
    df["context_anomaly"] = abs_resid > threshold
    return df


def add_persistence(df: pd.DataFrame, window: int = PERSISTENCE_WINDOW) -> pd.DataFrame:
    df = df.copy()
    df["persistent"] = df.groupby(EQUIPMENT_COL)["context_anomaly"].transform(
        lambda x: x.rolling(window, min_periods=1).sum() >= window
    )
    df["persistence_score"] = df["persistent"].astype(int) * 100
    return df


def add_combined_score(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ai_score"] = (
        SCORE_WEIGHTS["context"] * df["context_score"]
        + SCORE_WEIGHTS["iso"] * df["iso_score"]
        + SCORE_WEIGHTS["persistence"] * df["persistence_score"]
    ).clip(0, 100)
    return df


def _severity_for(score: float) -> str:
    for low, high, label in SEVERITY_BINS:
        if low <= score < high:
            return label
    return SEVERITY_BINS[-1][2]


def add_severity(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["severity"] = df["ai_score"].apply(_severity_for)
    df["health_score"] = (100 - df["ai_score"]).clip(0, 100)
    return df


def score_all(df: pd.DataFrame) -> pd.DataFrame:
    df = add_context_score(df)
    df = add_persistence(df)
    df = add_combined_score(df)
    df = add_severity(df)
    return df


def equipment_health_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(EQUIPMENT_COL)
        .agg(
            health_score=("health_score", "mean"),
            anomaly_count=("context_anomaly", "sum"),
            avg_ai_score=("ai_score", "mean"),
        )
        .reset_index()
    )
    summary["health_score"] = summary["health_score"].round(1)
    summary["avg_ai_score"] = summary["avg_ai_score"].round(1)
    return summary
