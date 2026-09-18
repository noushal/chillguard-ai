import pandas as pd

from src.config import (
    COL_ENERGY,
    COL_LOAD,
    EQUIPMENT_COL,
    ROLLING_WINDOW,
    TIMESTAMP_COL,
)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"] = df[TIMESTAMP_COL].dt.hour
    df["dayofweek"] = df[TIMESTAMP_COL].dt.dayofweek
    df["month"] = df[TIMESTAMP_COL].dt.month
    df["day"] = df[TIMESTAMP_COL].dt.day
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    return df


def add_context_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["energy_per_rt"] = df[COL_ENERGY] / df[COL_LOAD].clip(lower=1)
    return df


def add_rolling_features(df: pd.DataFrame, window: int = ROLLING_WINDOW) -> pd.DataFrame:
    df = df.copy()
    grouped = df.groupby(EQUIPMENT_COL)[COL_ENERGY]
    df["energy_rolling_mean"] = grouped.transform(
        lambda x: x.rolling(window, min_periods=1).mean()
    )
    df["energy_rolling_std"] = grouped.transform(
        lambda x: x.rolling(window, min_periods=1).std()
    )
    df["energy_change"] = grouped.transform(lambda x: x.diff())
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_time_features(df)
    df = add_context_features(df)
    df = add_rolling_features(df)
    return df
