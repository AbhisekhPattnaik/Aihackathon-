import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from pathlib import Path

from streamlit import status

DATA_FILE = Path(__file__).parent / "train_FD001.txt"
MODEL_FILE = Path(__file__).parent / "rul_model.pkl"
SCALER_FILE = Path(__file__).parent / "scaler.pkl"


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    # Some versions of the file have trailing empty columns; drop if present
    if df.shape[1] > 26:
        cols_to_drop = [c for c in [26, 27] if c in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)

    columns = ["engine_id", "cycle"] + [f"op_setting_{i}" for i in range(1, 4)] + [f"sensor_{i}" for i in range(1, 22)]
    if df.shape[1] == len(columns):
        df.columns = columns
    else:
        # Fallback: create generic names for sensors if column count unexpected
        base = ["engine_id", "cycle"] + [f"op_setting_{i}" for i in range(1, 4)]
        sensors = [f"sensor_{i}" for i in range(1, df.shape[1] - len(base) + 1)]
        df.columns = base + sensors

    # Calculate RUL
    max_cycles = df.groupby("engine_id")["cycle"].max()
    df["max_cycle"] = df["engine_id"].map(max_cycles)
    df["RUL"] = df["max_cycle"] - df["cycle"]
    df.drop("max_cycle", axis=1, inplace=True)
    # Cap RUL for modelling convenience
    df["RUL"] = df["RUL"].clip(upper=125)

    return df


def train_and_save_model(df: pd.DataFrame, model_path: Path, scaler_path: Path):
    sensor_cols = [c for c in df.columns if c.startswith("sensor")]
    X = df[sensor_cols].copy()
    y = df["RUL"].copy()

    # Remove near-constant sensors
    stds = X.std()
    keep = stds[stds >= 0.01].index.tolist()
    if not keep:
        keep = X.columns.tolist()
    X = X[keep]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1, min_samples_leaf=5, max_depth=15)
    model.fit(X_train_scaled, y_train)

    # Evaluate quickly
    y_pred = model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    return model, scaler, keep, rmse, r2


if __name__ == "__main__":
    # If models exist, load them; otherwise train and save
    try:
        model = joblib.load(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)
        # Determine feature list from the saved scaler if available
        df = load_dataset(DATA_FILE)
        sensor_cols = [c for c in df.columns if c.startswith("sensor")]
        if hasattr(scaler, "feature_names_in_"):
            feature_cols = list(scaler.feature_names_in_)
        else:
            feature_cols = sensor_cols
        print("Loaded model and scaler from disk. Using features:", feature_cols)
    except Exception:
        print("Model or scaler not found — training a new model (this may take a while)...")
        df = load_dataset(DATA_FILE)
        model, scaler, feature_cols, rmse, r2 = train_and_save_model(df, MODEL_FILE, SCALER_FILE)
        print(f"Training finished. RMSE={rmse:.3f}, R2={r2:.3f}")

    # Streamlit UI
    try:
        import streamlit as st
        import plotly.express as px
    except Exception:
        print("Streamlit or plotly not installed; UI will not run here.")
        raise

    st.set_page_config(page_title="Jet Engine Health Monitor", layout="wide")
    st.title("✈️ Jet Engine Predictive Maintenance Dashboard")

    # Ensure dataset is loaded for selection
    data = load_dataset(DATA_FILE)
    engine_ids = sorted(data["engine_id"].unique())
    selected_engine = st.selectbox("Select Engine ID", engine_ids)

    engine_data = data[data["engine_id"] == selected_engine]
    latest_row = engine_data.iloc[-1:]
    sensor_choice = st.selectbox(
    "Select Sensor",
    [c for c in engine_data.columns if c.startswith("sensor")]
)

    fig = px.line(engine_data, x="cycle", y=sensor_choice,
              title=f"{sensor_choice} Trend")
    st.plotly_chart(fig, use_container_width=True)



    # Align input columns with scaler's expected feature order
    ordered_features = list(feature_cols)
    missing = [f for f in ordered_features if f not in latest_row.columns]
    if missing:
        st.warning(f"Missing features: {missing}. Filling with zeros for transform.")
        X_input = latest_row.reindex(columns=ordered_features).fillna(0)
    else:
        X_input = latest_row[ordered_features]
    X_input = X_input.astype(float)
    X_scaled = scaler.transform(X_input)
    rul_pred = model.predict(X_scaled)[0]
    
    MAX_RUL = 125
health = (rul_pred / MAX_RUL) * 100
health = max(0, min(100, health))

# ----- DEFINE STATUS STRING FIRST -----
if health > 70:
    engine_status = "Healthy"
elif health > 40:
    engine_status = "Warning"
else:
    engine_status = "Critical"

if engine_status == "Healthy":
    st.success("ENGINE IS HEALTHY")
elif engine_status == "Warning":
    st.warning("ENGINE NEEDS ATTENTION SOON")
else:
    st.error("ENGINE FAILURE IMMINENT — MAINTENANCE REQUIRED")
# ----- SHOW METRICS -----
col1, col2, col3 = st.columns(3)
col1.metric("Predicted RUL (Cycles)", int(rul_pred))
col2.metric("Health Percentage", f"{health:.1f}%")
col3.metric("Engine Status", engine_status)

st.progress(int(health))

st.subheader("Sensor Trend (Sensor 2 vs Cycle)")
fig = px.line(engine_data, x="cycle", y=[c for c in engine_data.columns if c == "sensor_2"], title="Sensor 2 Degradation Trend")
st.plotly_chart(fig, use_container_width=True)

st.info("System predicts failure before breakdown to enable preventive maintenance.")
st.subheader("Important Sensor Trends")

important_sensors = ["sensor_2", "sensor_4", "sensor_7"]

available = [s for s in important_sensors if s in engine_data.columns]

fig = px.line(
    engine_data,
    x="cycle",
    y=available,
    title="Key Sensor Degradation Patterns"
)
st.plotly_chart(fig, use_container_width=True)

engine_data = engine_data.copy()
engine_data["health"] = (engine_data["RUL"] / MAX_RUL) * 100

st.subheader("Engine Health Over Time")
fig2 = px.line(engine_data, x="cycle", y="health", title="Health Degradation Curve")
st.plotly_chart(fig2, use_container_width=True)

st.subheader("🔍 Model Insight: Feature Importance")

if hasattr(model, "feature_importances_"):
    importances = model.feature_importances_
    feat_df = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False).head(15)

    fig_imp = px.bar(
        feat_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Top 15 Important Features Affecting RUL",
    )
    st.plotly_chart(fig_imp, use_container_width=True)
else:
    st.info("Feature importance not available for this model.")
