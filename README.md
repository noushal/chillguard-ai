# ChillGuard AI - Intelligent Energy & Equipment Monitoring

Context-aware anomaly intelligence for chillers. Learns expected chiller
behaviour under current operating/environmental conditions, flags contextual
deviations, scores severity + persistence, and generates evidence-based
investigation recommendations - instead of a fixed energy threshold.

## Pipeline

```
CSV -> Data Validation -> Feature Engineering -> Expected Energy Model (HistGradientBoosting)
                                                -> Isolation Forest
    -> Contextual Score + Multivariate Score -> Persistence -> Combined AI Score
    -> Severity -> Evidence -> Recommendation -> Streamlit Dashboard
```

## Requirements

- Python 3.10+
- pip

## Installation

```bash
git clone <this-repo-url>
cd chillguard-ai

python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

## Data

The pipeline expects `data/chiller_data.csv` with columns:

```
timestamp, equipment_id, chilled_water_rate, cooling_water_temp, building_load,
chiller_energy_consumption, outside_temperature, dew_point, humidity, wind_speed, pressure
```

Three sources are available, pick one:

**Option A - real + synthetic blend (currently in `data/chiller_data.csv`)**

CHILLER-01 is real data ([Kaggle: Chiller Energy Data](https://www.kaggle.com/datasets/chillerenergy/chiller-energy-data),
Singapore commercial building, 13,615 rows, 2019-08-18 -> 2020-06-01, saved locally at
`data/raw/chiller_01_real_kaggle.csv`). CHILLER-02/03 are synthetic (no public multi-chiller
version of this dataset exists). Rebuild it with:

```bash
python -m src.build_dataset
```

**Option B - fully synthetic (dev/testing only)**

```bash
python -m src.generate_sample_data
```

**Option C - real dataset for all chillers**

Drop the real CSV directly at `data/chiller_data.csv` with the columns above (must include an
`equipment_id` per row). No code changes needed - the pipeline retrains on whatever is there.

## Run

```bash
# pipeline only - trains models, prints summary to console
python -m src.pipeline

# dashboard
streamlit run app.py
# if `streamlit` isn't on PATH (common on Windows):
python -m streamlit run app.py
```

Dashboard opens at `http://localhost:8501`. First load takes ~10-15s (trains models); cached after
that. Navigate via the tabs at the top: Overview, Equipment Monitoring, Anomaly Explorer, Evidence,
Historical Analysis.

`models/*.pkl` isn't in the repo (gitignored - trained binaries, not source). Both commands above
train fresh and write them to `models/` automatically on first run; nothing to download or set up
separately.

## Structure

```
chillguard-ai/
├── data/
│   ├── raw/
│   │   └── chiller_01_real_kaggle.csv  # real CHILLER-01 source data
│   └── chiller_data.csv                # combined dataset the pipeline reads
├── models/                             # trained models (.pkl), generated on first run
├── src/
│   ├── config.py               # column names, feature lists, thresholds
│   ├── preprocessing.py        # load, missing-value flags + interpolation
│   ├── features.py             # time, contextual, rolling features
│   ├── energy_model.py         # per-chiller expected-energy regressor
│   ├── anomaly_model.py        # Isolation Forest multivariate anomaly score
│   ├── scoring.py               # contextual score, persistence, combined score, severity
│   ├── explanations.py         # evidence + rule-based recommendations
│   ├── pipeline.py             # end-to-end orchestrator
│   ├── build_dataset.py        # merges real CHILLER-01 + synthetic CHILLER-02/03
│   └── generate_sample_data.py # fully synthetic dataset for dev/testing
└── app.py                      # Streamlit dashboard
```
