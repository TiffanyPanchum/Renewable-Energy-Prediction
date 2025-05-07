import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import datetime

from keras.src.saving import load_model

st.set_page_config(page_title="Solar Energy Forecast")

# --- Title & Description ---
st.title("☀️ Solar Energy Production Forecasting")
st.markdown("""
This dashboard uses an LSTM model trained on historical weather and time-based features to forecast solar energy production.  
The data includes weather metrics (temperature, humidity, wind speed, etc.) and time features (hour, day, month) to predict solar output.
""")

# --- Load Data ---
@st.cache_data
def load_data():
    url = "data/processed/Feature_Engineering _Dataset/feature_engineered_data.csv"
    df = pd.read_csv(url)
    df['time'] = pd.to_datetime(df['time'])
    return df

df = load_data()

# --- Load Model & Scalers ---
@st.cache_resource
def load_model_and_scalers():
    data_model = load_model("dashboard/lstm_model/solar_lstm_model.keras", compile=False)
    with open("dashboard/lstm_model/scalers.pkl", "rb") as f:
        scalers = pickle.load(f)
    return data_model, scalers["feature_scaler"], scalers["target_scaler"]

model, feature_scaler, target_scaler = load_model_and_scalers()

# --- Data Preview ---
st.subheader("📊 Data Overview")
st.dataframe(df.tail(10), use_container_width=True)

# --- Time Series Plot ---
st.subheader("📈 Solar Energy Output Over Time")
fig, ax = plt.subplots(figsize=(15, 5))
ax.plot(df["time"], df["Solar"], color='orange')
ax.set_xlabel("Time")
ax.set_ylabel("Solar Output")
st.pyplot(fig)

# --- Prediction ---
st.subheader("🔮 Predict Future Solar Output")

features = ['temp', 'rhum', 'prcp', 'wspd', 'pres', 'hour_sin', 'hour_cos',
            'day_of_week_sin', 'day_of_week_cos', 'month_sin', 'month_cos',
            'Solar_lag_1', 'Solar_lag_24', 'Solar_rolling_24h_mean']

# Make sure no missing values
df[features] = df[features].ffill().bfill()

# Get last 24 rows (most recent full day)
lookback = 24
latest_data = df[features].iloc[-lookback:].values
scaled_input = feature_scaler.transform(latest_data)
scaled_input = scaled_input.reshape(1, lookback, len(features))

# Make prediction
prediction_scaled = model.predict(scaled_input)
prediction = target_scaler.inverse_transform(prediction_scaled)[0][0]

# Show prediction
last_timestamp = df["time"].iloc[-1]
predicted_time = last_timestamp + datetime.timedelta(hours=1)

st.success(f"📍 Predicted solar output for {predicted_time.strftime('%Y-%m-%d %H:%M')} is **{prediction:.2f}** units.")

# --- Footer ---
st.markdown("---")
st.markdown("Built with ❤️ using LSTM + Streamlit. Data source: historical weather and solar metrics for France, Italy, and Spain.")
