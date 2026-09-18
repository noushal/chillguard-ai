import numpy as np
import pandas as pd

EQUIPMENT_IDS = ["CHILLER-01", "CHILLER-02", "CHILLER-03"]
START = "2019-08-18"
END = "2020-06-01"


def generate(seed: int = 42, equipment_ids: list = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range(START, END, freq="30min")
    equipment_ids = equipment_ids or EQUIPMENT_IDS

    rows = []
    for equipment in equipment_ids:
        n = len(timestamps)

        day_frac = np.asarray((timestamps.hour + timestamps.minute / 60) / 24)
        seasonal = np.sin(2 * np.pi * timestamps.dayofyear.to_numpy() / 365)

        outside_temp = 75 + 12 * seasonal + 8 * np.sin(2 * np.pi * day_frac) + rng.normal(0, 2, n)
        humidity = np.clip(55 + 15 * np.sin(2 * np.pi * day_frac + 1) + rng.normal(0, 5, n), 10, 100)
        dew_point = outside_temp - (100 - humidity) / 5 + rng.normal(0, 1, n)
        wind_speed = np.clip(rng.normal(8, 3, n), 0, None)
        pressure = rng.normal(29.9, 0.15, n)

        building_load = np.clip(
            400 + 300 * np.sin(2 * np.pi * day_frac) + 150 * seasonal + rng.normal(0, 40, n),
            50,
            None,
        )
        flow = building_load * 0.9 + rng.normal(0, 15, n)
        cooling_water_temp = 24 + 0.05 * outside_temp + rng.normal(0, 1, n)

        base_energy = 0.75 * building_load + 0.3 * outside_temp + rng.normal(0, 20, n)

        anomaly_mask = rng.random(n) < 0.01
        anomaly_blocks = np.zeros(n, dtype=bool)
        block_starts = rng.choice(n - 6, size=max(1, n // 3000), replace=False)
        for s in block_starts:
            anomaly_blocks[s : s + rng.integers(3, 8)] = True

        energy = base_energy.copy()
        spike = (anomaly_mask | anomaly_blocks)
        energy[spike] *= rng.uniform(1.25, 1.6, spike.sum())

        equipment_ts = pd.DataFrame(
            {
                "timestamp": timestamps,
                "equipment_id": equipment,
                "chilled_water_rate": flow,
                "cooling_water_temp": cooling_water_temp,
                "building_load": building_load,
                "chiller_energy_consumption": energy,
                "outside_temperature": outside_temp,
                "dew_point": dew_point,
                "humidity": humidity,
                "wind_speed": wind_speed,
                "pressure": pressure,
            }
        )

        drop_frac = 0.01
        drop_idx = rng.choice(n, size=int(n * drop_frac), replace=False)
        numeric_cols = equipment_ts.columns.drop(["timestamp", "equipment_id"])
        for col in numeric_cols:
            col_drops = rng.choice(drop_idx, size=len(drop_idx) // len(numeric_cols) + 1, replace=False)
            equipment_ts.loc[equipment_ts.index.isin(col_drops), col] = np.nan

        rows.append(equipment_ts)

    df = pd.concat(rows, ignore_index=True)
    return df


if __name__ == "__main__":
    df = generate()
    df.to_csv("data/chiller_data.csv", index=False)
    print(f"Wrote {len(df)} rows to data/chiller_data.csv")
    print(df["equipment_id"].value_counts())
