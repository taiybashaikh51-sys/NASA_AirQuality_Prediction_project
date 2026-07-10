import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="NASA Air Quality Prediction",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background-color: #0B1220; color: #ffffff; }
    .main-header {
        background: linear-gradient(135deg, #1A2333, #0B1220);
        border: 1px solid rgba(59,130,246,0.3);
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 2rem;
        text-align: center;
    }
    .section-title {
        color: #3B82F6;
        font-size: 1.3rem;
        font-weight: 700;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(59,130,246,0.3);
    }
    div[data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid rgba(59,130,246,0.2);
    }
    .aqi-good { color: #22C55E; font-weight: 700; }
    .aqi-moderate { color: #EAB308; font-weight: 700; }
    .aqi-poor { color: #F97316; font-weight: 700; }
    .aqi-hazardous { color: #EF4444; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv('cleaned_aqi_nasa.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    return df

df = load_data()

# Sidebar
with st.sidebar:
    st.markdown("### 🌍 NASA AQI App")
    st.markdown("---")
    
    cities = st.multiselect(
        "🏙️ Select Cities",
        options=sorted(df['City'].unique().tolist()),
        default=sorted(df['City'].unique().tolist())[:5]
    )
    
    year = st.multiselect(
        "📅 Select Year",
        options=sorted(df['year'].unique().tolist()),
        default=sorted(df['year'].unique().tolist())
    )
    
    st.markdown("---")
    page = st.radio(
        "📊 Select Page",
        ["🏠 Overview", "📈 AQI Analysis", "🤖 ML Prediction", "🌆 City Comparison", "📋 Data Explorer"]
    )

# Filter
filtered = df[df['City'].isin(cities) & df['year'].isin(year)]

# Header
st.markdown("""
<div class="main-header">
    <h1 style="color:#3B82F6; margin:0; font-size:2rem;">🌍 NASA Air Quality Prediction</h1>
    <p style="color:#94A3B8; margin:0.5rem 0 0 0;">AI-Powered AQI Analysis & Forecasting using NASA Satellite Data</p>
</div>
""", unsafe_allow_html=True)

# KPIs
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("📊 Avg AQI", f"{filtered['AQI'].mean():.1f}")
with col2:
    st.metric("🏙️ Cities", f"{filtered['City'].nunique()}")
with col3:
    st.metric("📅 Records", f"{len(filtered):,}")
with col4:
    st.metric("🌡️ Avg Temp", f"{filtered['Temperature'].mean():.1f}°C")
with col5:
    worst = filtered.loc[filtered['AQI'].idxmax(), 'City']
    st.metric("⚠️ Worst City", worst)

st.markdown("---")

# OVERVIEW
if page == "🏠 Overview":
    st.markdown('<p class="section-title">📈 AQI Trend Over Time</p>', unsafe_allow_html=True)
    
    daily = filtered.groupby('Date')['AQI'].mean().reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily['Date'], y=daily['AQI'],
        fill='tozeroy',
        fillcolor='rgba(59,130,246,0.1)',
        line=dict(color='#3B82F6', width=2),
        name='AQI'
    ))
    fig.add_hline(y=100, line_dash="dash", line_color="#EAB308", annotation_text="Moderate")
    fig.add_hline(y=200, line_dash="dash", line_color="#F97316", annotation_text="Poor")
    fig.add_hline(y=300, line_dash="dash", line_color="#EF4444", annotation_text="Hazardous")
    fig.update_layout(
        paper_bgcolor='#1A2333', plot_bgcolor='#1A2333',
        font=dict(color='#94A3B8'), height=350,
        xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.05)')
    )
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<p class="section-title">🏙️ AQI by City</p>', unsafe_allow_html=True)
        city_aqi = filtered.groupby('City')['AQI'].mean().sort_values(ascending=False).reset_index()
        fig2 = px.bar(city_aqi, x='City', y='AQI',
            color='AQI', color_continuous_scale='RdYlGn_r')
        fig2.update_layout(
            paper_bgcolor='#1A2333', plot_bgcolor='#1A2333',
            font=dict(color='#94A3B8'),
            xaxis=dict(gridcolor='rgba(255,255,255,0.05)', tickangle=45),
            yaxis=dict(gridcolor='rgba(255,255,255,0.05)'))
        st.plotly_chart(fig2, use_container_width=True)
    
    with col2:
        st.markdown('<p class="section-title">📊 AQI Category Distribution</p>', unsafe_allow_html=True)
        cat_dist = filtered['AQI_Category'].value_counts().reset_index()
        fig3 = px.pie(cat_dist, values='count', names='AQI_Category',
            color_discrete_sequence=['#22C55E', '#EAB308', '#F97316', '#EF4444', '#8B5CF6'])
        fig3.update_layout(paper_bgcolor='#1A2333', font=dict(color='#94A3B8'))
        st.plotly_chart(fig3, use_container_width=True)

# AQI ANALYSIS
elif page == "📈 AQI Analysis":
    st.markdown('<p class="section-title">🔬 Pollutant Analysis</p>', unsafe_allow_html=True)
    
    pollutants = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
    
    col1, col2 = st.columns(2)
    with col1:
        selected_pollutant = st.selectbox("Select Pollutant", pollutants)
        city_poll = filtered.groupby('City')[selected_pollutant].mean().sort_values(ascending=False).reset_index()
        fig = px.bar(city_poll, x='City', y=selected_pollutant,
            color=selected_pollutant, color_continuous_scale='Reds',
            title=f'{selected_pollutant} by City')
        fig.update_layout(
            paper_bgcolor='#1A2333', plot_bgcolor='#1A2333',
            font=dict(color='#94A3B8'),
            xaxis=dict(tickangle=45, gridcolor='rgba(255,255,255,0.05)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.05)'))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        corr_cols = pollutants + ['AQI', 'Temperature', 'Humidity']
        corr = filtered[corr_cols].corr()
        fig2 = px.imshow(corr, color_continuous_scale='RdBu',
            title='Correlation Heatmap')
        fig2.update_layout(paper_bgcolor='#1A2333', font=dict(color='#94A3B8'))
        st.plotly_chart(fig2, use_container_width=True)
    
    st.markdown('<p class="section-title">📅 Monthly AQI Pattern</p>', unsafe_allow_html=True)
    monthly = filtered.groupby('month')['AQI'].mean().reset_index()
    months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    monthly['Month'] = monthly['month'].apply(lambda x: months[x-1])
    fig3 = px.line(monthly, x='Month', y='AQI',
        markers=True, color_discrete_sequence=['#3B82F6'])
    fig3.update_layout(
        paper_bgcolor='#1A2333', plot_bgcolor='#1A2333',
        font=dict(color='#94A3B8'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.05)'))
    st.plotly_chart(fig3, use_container_width=True)

# ML PREDICTION
elif page == "🤖 ML Prediction":
    st.markdown('<p class="section-title">🤖 AQI Prediction Model</p>', unsafe_allow_html=True)
    
    features = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'Temperature', 'Humidity', 'WindSpeed', 'Pressure']
    
    X = df[features].dropna()
    y = df.loc[X.index, 'AQI']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    col1, col2 = st.columns(2)
    with col1:
        model_choice = st.selectbox("🤖 Select Model", ["Random Forest", "Linear Regression"])
    with col2:
        st.markdown("#### Model Training...")
    
    if model_choice == "Random Forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        model = LinearRegression()
    
    with st.spinner("Training model..."):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📊 R² Score", f"{r2:.4f}")
    with col2:
        st.metric("📉 MAE", f"{mae:.2f}")
    with col3:
        st.metric("✅ Accuracy", f"{r2*100:.1f}%")
    
    st.markdown('<p class="section-title">🎯 Real-time AQI Prediction</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        pm25 = st.slider("PM2.5", 0.0, 500.0, 100.0)
        pm10 = st.slider("PM10", 0.0, 600.0, 150.0)
        no2 = st.slider("NO2", 0.0, 200.0, 40.0)
        so2 = st.slider("SO2", 0.0, 100.0, 20.0)
    with col2:
        co = st.slider("CO", 0.0, 50.0, 5.0)
        o3 = st.slider("O3", 0.0, 200.0, 50.0)
        temp = st.slider("Temperature (°C)", -10.0, 50.0, 25.0)
    with col3:
        humidity = st.slider("Humidity (%)", 0.0, 100.0, 60.0)
        wind = st.slider("Wind Speed", 0.0, 30.0, 10.0)
        pressure = st.slider("Pressure", 950.0, 1050.0, 1013.0)
    
    input_data = np.array([[pm25, pm10, no2, so2, co, o3, temp, humidity, wind, pressure]])
    prediction = model.predict(input_data)[0]
    
    if prediction <= 50:
        cat = "Good 🟢"
        color = "#22C55E"
    elif prediction <= 100:
        cat = "Moderate 🟡"
        color = "#EAB308"
    elif prediction <= 200:
        cat = "Poor 🟠"
        color = "#F97316"
    else:
        cat = "Hazardous 🔴"
        color = "#EF4444"
    
    st.markdown(f"""
    <div style="background:#1A2333; border:1px solid {color}; border-radius:16px; padding:2rem; text-align:center; margin-top:1rem;">
        <h2 style="color:{color}; margin:0;">Predicted AQI: {prediction:.1f}</h2>
        <p style="color:#94A3B8; margin:0.5rem 0 0 0;">Category: <strong style="color:{color}">{cat}</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    if model_choice == "Random Forest":
        st.markdown('<p class="section-title">🔍 Feature Importance</p>', unsafe_allow_html=True)
        feat_imp = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_})
        feat_imp = feat_imp.sort_values('Importance', ascending=True)
        fig = px.bar(feat_imp, x='Importance', y='Feature', orientation='h',
            color='Importance', color_continuous_scale='Blues')
        fig.update_layout(
            paper_bgcolor='#1A2333', plot_bgcolor='#1A2333',
            font=dict(color='#94A3B8'),
            xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.05)'))
        st.plotly_chart(fig, use_container_width=True)

# CITY COMPARISON
elif page == "🌆 City Comparison":
    st.markdown('<p class="section-title">🌆 City-wise AQI Comparison</p>', unsafe_allow_html=True)
    
    city1 = st.selectbox("City 1", sorted(df['City'].unique()))
    city2 = st.selectbox("City 2", sorted(df['City'].unique()), index=1)
    
    c1 = df[df['City'] == city1].groupby('Date')['AQI'].mean().reset_index()
    c2 = df[df['City'] == city2].groupby('Date')['AQI'].mean().reset_index()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=c1['Date'], y=c1['AQI'], name=city1,
        line=dict(color='#3B82F6', width=2)))
    fig.add_trace(go.Scatter(x=c2['Date'], y=c2['AQI'], name=city2,
        line=dict(color='#F97316', width=2)))
    fig.update_layout(
        paper_bgcolor='#1A2333', plot_bgcolor='#1A2333',
        font=dict(color='#94A3B8'), height=400,
        xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.05)'))
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"#### 📊 {city1} Stats")
        c1_stats = df[df['City']==city1][['AQI','PM2.5','PM10','Temperature']].describe().round(2)
        st.dataframe(c1_stats, use_container_width=True)
    with col2:
        st.markdown(f"#### 📊 {city2} Stats")
        c2_stats = df[df['City']==city2][['AQI','PM2.5','PM10','Temperature']].describe().round(2)
        st.dataframe(c2_stats, use_container_width=True)

# DATA EXPLORER
elif page == "📋 Data Explorer":
    st.markdown('<p class="section-title">📋 Raw Data Explorer</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        selected_city = st.selectbox("Filter by City", ['All'] + sorted(df['City'].unique().tolist()))
    with col2:
        selected_cat = st.selectbox("Filter by AQI Category", ['All'] + sorted(df['AQI_Category'].unique().tolist()))
    
    exp_df = filtered.copy()
    if selected_city != 'All':
        exp_df = exp_df[exp_df['City'] == selected_city]
    if selected_cat != 'All':
        exp_df = exp_df[exp_df['AQI_Category'] == selected_cat]
    
    st.dataframe(exp_df.head(500), use_container_width=True, height=400)
    
    csv = exp_df.to_csv(index=False)
    st.download_button("📥 Download Data", csv, "nasa_aqi_data.csv", "text/csv")

st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#94A3B8; font-size:0.8rem; padding:1rem;">
    🌍 NASA Air Quality Prediction | Built by <strong style="color:#3B82F6">Taiyba Shaikh</strong> | ML Project
</div>
""", unsafe_allow_html=True)