TIMESTAMP_COL = "timestamp"
EQUIPMENT_COL = "equipment_id"

COL_FLOW = "chilled_water_rate"
COL_COOLING_TEMP = "cooling_water_temp"
COL_LOAD = "building_load"
COL_ENERGY = "chiller_energy_consumption"
COL_OUTSIDE_TEMP = "outside_temperature"
COL_DEW_POINT = "dew_point"
COL_HUMIDITY = "humidity"
COL_WIND_SPEED = "wind_speed"
COL_PRESSURE = "pressure"

RAW_NUMERIC_COLS = [
    COL_FLOW,
    COL_COOLING_TEMP,
    COL_LOAD,
    COL_ENERGY,
    COL_OUTSIDE_TEMP,
    COL_DEW_POINT,
    COL_HUMIDITY,
    COL_WIND_SPEED,
    COL_PRESSURE,
]

MODEL_FEATURES = [
    COL_FLOW,
    COL_COOLING_TEMP,
    COL_LOAD,
    COL_OUTSIDE_TEMP,
    COL_DEW_POINT,
    COL_HUMIDITY,
    COL_WIND_SPEED,
    COL_PRESSURE,
    "hour",
    "dayofweek",
    "month",
    "is_weekend",
]

ANOMALY_FEATURES = [
    COL_FLOW,
    COL_COOLING_TEMP,
    COL_LOAD,
    COL_ENERGY,
    COL_OUTSIDE_TEMP,
    COL_DEW_POINT,
    COL_HUMIDITY,
    COL_WIND_SPEED,
    COL_PRESSURE,
]

TARGET = COL_ENERGY

ROLLING_WINDOW = 6
PERSISTENCE_WINDOW = 3
CONTEXT_ANOMALY_QUANTILE = 0.99

SCORE_WEIGHTS = {
    "context": 0.5,
    "iso": 0.3,
    "persistence": 0.2,
}

SEVERITY_BINS = [
    (0, 40, "Normal"),
    (40, 70, "Monitor"),
    (70, 85, "Attention"),
    (85, 101, "Critical"),
]

MODELS_DIR = "models"
DATA_DIR = "data"
