import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

st.set_page_config(page_title="NASA Air Quality — EDA + ML App", layout="wide")
st.title("🌍 NASA Air Quality — EDA + Machine Learning App")

DATA_PATH = "Data/NASA_AirQuality_Excel_Analysis.xlsx"

@st.cache_data(show_spinner=False)
def load_data(path=DATA_PATH):
    xls = pd.read_excel(path, sheet_name=None)
    if isinstance(xls, dict) and len(xls) > 1:
        df = pd.concat(xls.values(), ignore_index=True)
    else:
        df = list(xls.values())[0]
    return df

df = load_data(DATA_PATH)

eda_tab, ml_tab = st.tabs(["📊 EDA Dashboard", "🤖 ML Prediction"])

# ==============================================================
# EDA TAB
# ==============================================================
with eda_tab:
    st.header("Exploratory Data Analysis (EDA)")
    st.write(df.head())
    st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    info_df = pd.DataFrame({
        'dtype': df.dtypes.astype(str),
        'missing': df.isna().sum(),
        'unique': df.nunique()
    })
    st.dataframe(info_df)

    st.write("### Missing Value Percentage")
    st.bar_chart(df.isna().mean() * 100)

    st.write("### Descriptive Statistics")
    st.dataframe(df.describe())

    st.write("### Correlation Heatmap")
    num_df = df.select_dtypes(include=[np.number])
    if not num_df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(num_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        st.pyplot(fig)

    st.write("### Pairplot (Sampled)")
    if num_df.shape[1] >= 2:
        sample_df = num_df.sample(min(200, len(num_df)), random_state=42)
        fig = sns.pairplot(sample_df)
        st.pyplot(fig)

    if 'City' in df.columns and 'AQI' in df.columns:
        st.write("### Average AQI by City")
        avg_city = df.groupby('City')['AQI'].mean().sort_values(ascending=False)
        st.bar_chart(avg_city)

# ==============================================================
# ML PREDICTION TAB (Simplified)
# ==============================================================
with ml_tab:
    st.header("🤖 AQI Prediction (Using Pretrained Model)")

    model = joblib.load("aqi_prediction_rf_pipeline.joblib")

    input_features = [
        'City', 'PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3',
        'Temperature', 'Humidity', 'WindSpeed', 'Rainfall',
        'Pressure', 'Visibility', 'UVIndex',
        'Year', 'Month', 'DayOfWeek', 'WeekOfYear', 'IsWeekend'
    ]

    st.subheader("Enter Environmental and Temporal Parameters")

    user_input = {}
    cities = sorted(df['City'].dropna().unique().tolist()) if 'City' in df.columns else ['Delhi']
    user_input['City'] = st.selectbox("City", options=cities, index=0)

    numeric_sliders = [
        'PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'Temperature',
        'Humidity', 'WindSpeed', 'Rainfall', 'Pressure', 'Visibility', 'UVIndex'
    ]

    for col in numeric_sliders:
        min_val = float(df[col].min()) if col in df.columns else 0.0
        max_val = float(df[col].max()) if col in df.columns else 100.0
        median_val = float(df[col].median()) if col in df.columns else 50.0
        user_input[col] = st.slider(col, min_value=round(min_val, 2), max_value=round(max_val, 2), value=round(median_val, 2), step=0.1)

    st.subheader("Date and Time Features")
    user_input['Year'] = st.number_input("Year", min_value=2000, max_value=2100, value=2025, step=1)
    user_input['Month'] = st.slider("Month", 1, 12, 10)
    user_input['DayOfWeek'] = st.slider("Day of Week (0=Mon, 6=Sun)", 0, 6, 0)
    user_input['WeekOfYear'] = st.slider("Week of Year", 1, 52, 43)
    user_input['IsWeekend'] = st.selectbox("Is Weekend", [0, 1], index=0)

    if st.button("🔮 Predict AQI"):
        input_df = pd.DataFrame([user_input])
        prediction = model.predict(input_df)[0]
        st.subheader("🌫️ Predicted Air Quality Index (AQI)")
        st.success(f"**{prediction:.2f}**")

        bins = [0, 50, 100, 150, 200, 300, 500]
        labels = ["Good", "Moderate", "Unhealthy for Sensitive Groups", "Unhealthy", "Very Unhealthy", "Hazardous"]
        category = pd.cut([prediction], bins=bins, labels=labels)[0]
        st.info(f"**Predicted AQI Category:** {category}")
