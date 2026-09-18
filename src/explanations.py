import pandas as pd

from src.config import COL_ENERGY, COL_LOAD, EQUIPMENT_COL, TIMESTAMP_COL


def build_evidence(row: pd.Series) -> dict:
    deviation_pct = row.get("energy_deviation_pct", 0.0)
    direction = "above" if deviation_pct >= 0 else "below"

    evidence = {
        "equipment": row[EQUIPMENT_COL],
        "timestamp": row[TIMESTAMP_COL],
        "severity": row["severity"],
        "ai_score": round(row["ai_score"], 1),
        "actual_energy": round(row[COL_ENERGY], 1),
        "expected_energy": round(row["expected_energy"], 1),
        "deviation_pct": round(deviation_pct, 1),
        "building_load": round(row[COL_LOAD], 1),
        "persistent": bool(row["persistent"]),
        "multivariate_anomaly": bool(row["iso_prediction"] == -1),
        "direction": direction,
    }
    return evidence


def recommendation_for(row: pd.Series) -> str:
    high_energy_deviation = row["context_anomaly"] and row["energy_residual"] > 0
    low_energy_deviation = row["context_anomaly"] and row["energy_residual"] < 0
    persistent = bool(row["persistent"])
    multivariate = row["iso_prediction"] == -1

    if high_energy_deviation and persistent:
        return (
            "Investigation recommended: review chiller energy consumption and "
            "operating conditions during this period. Behaviour has persisted "
            "across multiple consecutive observations."
        )
    if high_energy_deviation:
        return (
            "Energy consumption is higher than expected for the observed "
            "operating conditions. Investigate energy usage during this period."
        )
    if low_energy_deviation and persistent:
        return (
            "Investigation recommended: energy consumption is persistently lower "
            "than expected for the observed load; review flow and sensor readings."
        )
    if multivariate:
        return (
            "Unusual combination of operating measurements detected. Review "
            "chilled-water flow, cooling-water temperature and related conditions."
        )
    return "No investigation required; behaviour is within expected range."


def explain_anomaly(row: pd.Series) -> str:
    evidence = build_evidence(row)
    persistence_text = (
        "The deviation persisted across multiple observations."
        if evidence["persistent"]
        else "This observation was isolated and did not persist."
    )
    return (
        f"{evidence['equipment']} showed abnormal energy behaviour "
        f"at {evidence['timestamp']}.\n\n"
        f"The observed energy consumption ({evidence['actual_energy']} kWh) was "
        f"{abs(evidence['deviation_pct'])}% {evidence['direction']} the expected value "
        f"({evidence['expected_energy']} kWh) predicted from building load, "
        f"chilled-water flow, cooling-water temperature, outdoor conditions and "
        f"historical chiller behaviour.\n\n"
        f"{persistence_text}\n\n"
        f"Recommended action: {recommendation_for(row)}"
    )


def top_anomalies(df: pd.DataFrame, n: int = 50) -> pd.DataFrame:
    flagged = df[df["severity"].isin(["Attention", "Critical"])]
    return flagged.sort_values("ai_score", ascending=False).head(n)
