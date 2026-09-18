import pandas as pd

from src.config import EQUIPMENT_COL, RAW_NUMERIC_COLS, TIMESTAMP_COL


def load_raw(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL])
    df = df.sort_values([EQUIPMENT_COL, TIMESTAMP_COL]).reset_index(drop=True)
    return df


def add_missing_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in RAW_NUMERIC_COLS:
        if col in df.columns:
            df[f"{col}_missing"] = df[col].isna().astype(int)
    return df


def interpolate_missing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in RAW_NUMERIC_COLS:
        if col in df.columns:
            df[col] = df.groupby(EQUIPMENT_COL)[col].transform(
                lambda x: x.interpolate(limit_direction="both")
            )
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = add_missing_flags(df)
    df = interpolate_missing(df)
    return df


def load_and_clean(csv_path: str) -> pd.DataFrame:
    df = load_raw(csv_path)
    df = clean(df)
    return df
