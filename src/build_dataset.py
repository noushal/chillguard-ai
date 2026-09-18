import pandas as pd

from src.generate_sample_data import generate

REAL_CSV = "data/raw/chiller_01_real_kaggle.csv"
OUTPUT_CSV = "data/chiller_data.csv"

REAL_COLUMN_MAP = {
    "Local Time (Timezone : GMT+8h)": "timestamp",
    "Chilled Water Rate (L/sec)": "chilled_water_rate",
    "Cooling Water Temperature (C)": "cooling_water_temp",
    "Building Load (RT)": "building_load",
    "Chiller Energy Consumption (kWh)": "chiller_energy_consumption",
    "Outside Temperature (F)": "outside_temperature",
    "Dew Point (F)": "dew_point",
    "Humidity (%)": "humidity",
    "Wind Speed (mph)": "wind_speed",
    "Pressure (in)": "pressure",
}


def load_real_chiller_01() -> pd.DataFrame:
    df = pd.read_csv(REAL_CSV)
    df = df.rename(columns=REAL_COLUMN_MAP)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["equipment_id"] = "CHILLER-01"
    return df


def build() -> pd.DataFrame:
    real = load_real_chiller_01()
    synthetic = generate(equipment_ids=["CHILLER-02", "CHILLER-03"])
    combined = pd.concat([real, synthetic], ignore_index=True)
    combined = combined.sort_values(["equipment_id", "timestamp"]).reset_index(drop=True)
    return combined


if __name__ == "__main__":
    df = build()
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {len(df)} rows to {OUTPUT_CSV}")
    print(df["equipment_id"].value_counts())
