import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import COL_ENERGY, COL_LOAD, EQUIPMENT_COL, TIMESTAMP_COL
from src.explanations import explain_anomaly, top_anomalies
from src.pipeline import run_pipeline
from src.scoring import equipment_health_summary

st.set_page_config(page_title="ChillGuard AI", layout="wide")

DATA_PATH = os.path.join("data", "chiller_data.csv")


@st.cache_data(show_spinner="Running pipeline: clean -> features -> ML -> scores...")
def load_scored_data(csv_path: str, mtime: float) -> pd.DataFrame:
    return run_pipeline(csv_path, retrain=True)


def get_data() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        st.error(f"No dataset found at {DATA_PATH}. Add the CSV, or run "
                  "`python -m src.generate_sample_data` to create a synthetic one.")
        st.stop()
    mtime = os.path.getmtime(DATA_PATH)
    return load_scored_data(DATA_PATH, mtime)


def severity_badge(severity: str) -> str:
    colors = {"Normal": "🟢", "Monitor": "🟡", "Attention": "🟠", "Critical": "🔴"}
    return f"{colors.get(severity, '')} {severity}"


def page_overview(df: pd.DataFrame):
    st.caption("Intelligent Energy & Equipment Monitoring")

    n_equipment = df[EQUIPMENT_COL].nunique()
    n_obs = len(df)
    n_anomalies = int((df["severity"].isin(["Attention", "Critical"])).sum())

    c1, c2, c3 = st.columns(3)
    c1.metric("Chillers", n_equipment)
    c2.metric("Observations", f"{n_obs:,}")
    c3.metric("Flagged Anomalies", n_anomalies)

    st.subheader("Equipment Health Overview", anchor=False)
    summary = equipment_health_summary(df)
    cols = st.columns(len(summary))
    for col, (_, row) in zip(cols, summary.iterrows()):
        col.metric(row[EQUIPMENT_COL], f"{row['health_score']:.0f}", f"{int(row['anomaly_count'])} anomalies")

    st.dataframe(summary, width='stretch')


def page_equipment(df: pd.DataFrame):
    equipment = st.selectbox("Select Equipment", sorted(df[EQUIPMENT_COL].unique()))
    sub = df[df[EQUIPMENT_COL] == equipment].sort_values(TIMESTAMP_COL)

    date_range = st.date_input(
        "Date range",
        value=(sub[TIMESTAMP_COL].min().date(), sub[TIMESTAMP_COL].max().date()),
    )
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start, end = date_range
        sub = sub[(sub[TIMESTAMP_COL].dt.date >= start) & (sub[TIMESTAMP_COL].dt.date <= end)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sub[TIMESTAMP_COL], y=sub[COL_ENERGY], name="Actual Energy", mode="lines"))
    fig.add_trace(go.Scatter(x=sub[TIMESTAMP_COL], y=sub["expected_energy"], name="Expected Energy", mode="lines", line=dict(dash="dash")))
    anomalies = sub[sub["severity"].isin(["Attention", "Critical"])]
    fig.add_trace(go.Scatter(x=anomalies[TIMESTAMP_COL], y=anomalies[COL_ENERGY], name="Anomaly", mode="markers", marker=dict(color="red", size=8)))
    fig.update_layout(title="Actual vs Expected Energy", xaxis_title="Time", yaxis_title="Energy (kWh)")
    st.plotly_chart(fig, width='stretch')

    c1, c2 = st.columns(2)
    with c1:
        fig_load = go.Figure()
        fig_load.add_trace(go.Scatter(x=sub[TIMESTAMP_COL], y=sub[COL_LOAD], name="Building Load"))
        fig_load.update_layout(title="Building Load")
        st.plotly_chart(fig_load, width='stretch')
    with c2:
        fig_temp = go.Figure()
        fig_temp.add_trace(go.Scatter(x=sub[TIMESTAMP_COL], y=sub["outside_temperature"], name="Outside Temp"))
        fig_temp.add_trace(go.Scatter(x=sub[TIMESTAMP_COL], y=sub["cooling_water_temp"], name="Cooling Water Temp"))
        fig_temp.update_layout(title="Temperatures")
        st.plotly_chart(fig_temp, width='stretch')


def page_anomaly_explorer(df: pd.DataFrame):
    flagged = top_anomalies(df, n=200)

    display_cols = [TIMESTAMP_COL, EQUIPMENT_COL, "ai_score", "severity", "energy_deviation_pct", "persistent"]
    st.dataframe(
        flagged[display_cols].rename(columns={"energy_deviation_pct": "energy_deviation_%"}),
        width='stretch',
        height=400,
    )

    if flagged.empty:
        st.info("No Attention/Critical anomalies detected.")
        return

    st.subheader("Anomaly Details", anchor=False)
    options = flagged.apply(lambda r: f"{r[TIMESTAMP_COL]} | {r[EQUIPMENT_COL]} | score {r['ai_score']:.0f}", axis=1)
    choice = st.selectbox("Select anomaly", options, key="anomaly_explorer_select")
    selected_row = flagged.iloc[options.tolist().index(choice)]

    c1, c2, c3 = st.columns(3)
    c1.metric("Equipment", selected_row[EQUIPMENT_COL])
    c2.metric("Severity", severity_badge(selected_row["severity"]))
    c3.metric("AI Score", f"{selected_row['ai_score']:.0f}")

    c1, c2, c3 = st.columns(3)
    c1.metric("Actual Energy", f"{selected_row[COL_ENERGY]:.0f} kWh")
    c2.metric("Expected Energy", f"{selected_row['expected_energy']:.0f} kWh")
    c3.metric("Deviation", f"{selected_row['energy_deviation_pct']:.1f}%")

    st.write(f"**Persistence:** {'Yes - persisted across consecutive observations' if selected_row['persistent'] else 'No - isolated observation'}")
    st.write(f"**Recommendation:** {selected_row['recommendation']}")

    if st.button("Explain This Anomaly"):
        st.text(explain_anomaly(selected_row))


def page_evidence(df: pd.DataFrame):
    st.caption("Why did the AI flag this?")

    flagged = top_anomalies(df, n=50)
    if flagged.empty:
        st.info("No anomalies to show evidence for.")
        return

    options = flagged.apply(lambda r: f"{r[TIMESTAMP_COL]} | {r[EQUIPMENT_COL]}", axis=1)
    choice = st.selectbox("Select anomaly", options, key="evidence_select")
    row = flagged.iloc[options.tolist().index(choice)]

    st.markdown(f"""
    | Signal | Value |
    |---|---|
    | Building Load | {row[COL_LOAD]:.0f} RT |
    | Expected Energy | {row['expected_energy']:.0f} kWh |
    | Actual Energy | {row[COL_ENERGY]:.0f} kWh |
    | Energy deviation | {row['energy_deviation_pct']:.1f}% |
    | Multivariate anomaly | {"Yes" if row['iso_prediction'] == -1 else "No"} |
    | Persistence | {"Persisted across consecutive observations" if row['persistent'] else "Isolated"} |
    | Contextual anomaly score | {row['context_score']:.0f} / 100 |
    """)

    st.subheader("Recommended Investigation", anchor=False)
    st.success(row["recommendation"])


def page_historical(df: pd.DataFrame):
    st.subheader("Anomalies by Equipment", anchor=False)
    by_equipment = df.groupby(EQUIPMENT_COL)["context_anomaly"].sum().reset_index()
    st.bar_chart(by_equipment.set_index(EQUIPMENT_COL))

    st.subheader("Anomalies by Hour of Day", anchor=False)
    by_hour = df.groupby("hour")["context_anomaly"].sum().reset_index()
    st.bar_chart(by_hour.set_index("hour"))

    st.subheader("Energy Efficiency Trend (Energy / RT)", anchor=False)
    equipment = st.selectbox("Equipment", sorted(df[EQUIPMENT_COL].unique()), key="trend_equipment")
    sub = df[df[EQUIPMENT_COL] == equipment].sort_values(TIMESTAMP_COL)
    trend = sub.set_index(TIMESTAMP_COL)["energy_per_rt"].rolling(48).mean()
    st.line_chart(trend)


def main():
    df = get_data()

    st.title("ChillGuard AI", anchor=False)

    tab_overview, tab_equipment, tab_anomaly, tab_evidence, tab_historical = st.tabs(
        ["Overview", "Equipment Monitoring", "Anomaly Explorer", "Evidence", "Historical Analysis"]
    )

    with tab_overview:
        page_overview(df)
    with tab_equipment:
        page_equipment(df)
    with tab_anomaly:
        page_anomaly_explorer(df)
    with tab_evidence:
        page_evidence(df)
    with tab_historical:
        page_historical(df)


if __name__ == "__main__":
    main()
