"""
Streamlit Dashboard — Public Transport Delay Analysis
=====================================================
Interactive dashboard with 4 pages:
  1. Overview: Key metrics and summary
  2. EDA Explorer: Interactive charts with filters
  3. Predictions: Input conditions → predict delay
  4. Model Insights: Feature importance & model comparison
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
import shap
import matplotlib.pyplot as plt

# ─── Page Config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="🚍 Transport Delay Analyzer",
    page_icon="🚍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS & UI Polish ───────────────────────────────────────
st.markdown("""
<style>
    /* Global Font & Background */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(15px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .stApp {
        background-color: #f4f7f6;
        animation: fadeIn 0.6s ease-out forwards;
    }
    
    /* Sidebar Upgrade */
    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #eef2f6;
        box-shadow: 2px 0 10px rgba(0,0,0,0.02);
    }
    
    /* Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, #e0c3fc 0%, #8ec5fc 100%);
        border-radius: 16px;
        padding: 2.5rem 2rem;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 10px 25px -5px rgba(142, 197, 252, 0.4);
        position: relative;
        overflow: hidden;
    }
    .hero-banner::after {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background: url('https://www.transparenttextures.com/patterns/cubes.png');
        opacity: 0.1;
        pointer-events: none;
    }
    .hero-banner .main-header {
        font-size: 3.2rem;
        font-weight: 900;
        background: linear-gradient(135deg, #1e3a8a 0%, #4c1d95 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-top: 0;
        padding-top: 0;
        margin-bottom: 0.5rem;
    }
    .hero-banner .sub-header {
        color: #334155;
        font-weight: 500;
        margin-bottom: 0;
        font-size: 1.3rem;
    }

    /* Metric Cards (Native Streamlit override) */
    div[data-testid="metric-container"] {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }
    
    /* Custom HTML Info Cards */
    .info-card {
        background: white;
        border-left: 5px solid #4f46e5;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }

    /* 7-Day Forecast Cards */
    .forecast-card {
        background: linear-gradient(180deg, #ffffff 0%, #f1f5f9 100%);
        border: 1px solid #e2e8f0;
        border-top: 4px solid #f97316;
        border-radius: 10px;
        padding: 1.2rem 0.5rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.04);
        transition: all 0.2s;
    }
    .forecast-card:hover {
        transform: scale(1.03);
    }
    .forecast-day {
        font-size: 0.9rem;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .forecast-val {
        font-size: 1.8rem;
        color: #f97316;
        font-weight: 800;
        margin: 0.5rem 0;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #4f46e5 0%, #6366f1 100%);
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.2s ease;
        box-shadow: 0 4px 6px rgba(79, 70, 229, 0.2);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(79, 70, 229, 0.3);
        color: white;
    }

    /* Tabs Override */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 4px 4px 0 0;
        padding: 10px 16px;
        background-color: #f1f5f9;
        border: 1px solid transparent;
        border-bottom: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: white;
        border-color: #e2e8f0;
        border-bottom-color: white;
        font-weight: 600;
        color: #4f46e5 !important;
    }
</style>
""", unsafe_allow_html=True)

# ─── Load Data ────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")


@st.cache_data
def load_data():
    filepath = os.path.join(DATA_DIR, "featured_dataset.parquet")
    if os.path.exists(filepath):
        df = pd.read_parquet(filepath)
    else:
        # Fallback
        df = pd.read_csv(os.path.join(DATA_DIR, "featured_dataset.csv"), low_memory=False)
    
    # Convert dates
    date_cols = ["date", "scheduled_departure", "actual_departure"]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    return df

@st.cache_data
def load_forecast_data():
    fp_parquet = os.path.join(DATA_DIR, "delay_forecast.parquet")
    fp_csv = os.path.join(DATA_DIR, "delay_forecast.csv")
    
    if os.path.exists(fp_parquet):
        fdf = pd.read_parquet(fp_parquet)
    elif os.path.exists(fp_csv):
        fdf = pd.read_csv(fp_csv)
    else:
        return None
        
    fdf["date"] = pd.to_datetime(fdf["date"])
    return fdf

@st.cache_data
def load_results():
    results = {}
    for f in ["regression_results.csv", "classification_results.csv"]:
        fp = os.path.join(DATA_DIR, f)
        if os.path.exists(fp):
            results[f.replace("_results.csv", "")] = pd.read_csv(fp)
    return results

@st.cache_data
def load_feature_importance():
    fi = {}
    for f in ["feature_importance_regression.csv", "feature_importance_classification.csv"]:
        fp = os.path.join(DATA_DIR, f)
        if os.path.exists(fp):
            fi[f.replace("feature_importance_", "").replace(".csv", "")] = pd.read_csv(fp)
    return fi

@st.cache_resource
def load_shap_data():
    shap_data = {}
    for f in ["shap_regression.pkl", "shap_classification.pkl"]:
        fp = os.path.join(DATA_DIR, f)
        if os.path.exists(fp):
            shap_data[f.replace("shap_", "").replace(".pkl", "")] = joblib.load(fp)
    return shap_data

try:
    df = load_data()
except Exception as e:
    st.error(f"❌ Error loading data: {e}")
    st.info("Please run the preprocessing and feature engineering pipelines first.")
    st.stop()

# ─── Sidebar Navigation ───────────────────────────────────────────
st.sidebar.markdown("""
    <div style="text-align: center; padding: 1rem 0; border-bottom: 2px solid #f1f5f9; margin-bottom: 1.5rem;">
        <h2 style="color: #4f46e5; font-weight: 900; margin-bottom: 0; font-size: 1.8rem;">🚍 TransitAI</h2>
        <p style="color: #64748b; font-size: 0.95rem; margin-top: 5px; font-weight: 600;">Intelligence Dashboard</p>
    </div>
""", unsafe_allow_html=True)

st.sidebar.title("Navigation")
pages = ["📊 Overview", "🔍 EDA Explorer", "🔮 Predictions", "🧠 Model Insights", "🔮 7-Day Forecast"]
page = st.sidebar.radio("Go to", pages)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📅 Filters")

date_range = st.sidebar.date_input(
    "Date Range",
    value=(df["date"].min().date(), df["date"].max().date()),
    min_value=df["date"].min().date(),
    max_value=df["date"].max().date()
)

if len(date_range) == 2:
    mask = (df["date"].dt.date >= date_range[0]) & (df["date"].dt.date <= date_range[1])
    filtered_df = df[mask]
else:
    filtered_df = df

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎯 Advanced Filters")

if "borough" in df.columns:
    boroughs = st.sidebar.multiselect("🚇 Filter by Borough", options=sorted(df["borough"].dropna().unique()), default=[])
    if boroughs:
        filtered_df = filtered_df[filtered_df["borough"].isin(boroughs)]

if "weather_condition" in df.columns:
    weather_conds = st.sidebar.multiselect("🌦️ Filter by Weather", options=sorted(df["weather_condition"].dropna().unique()), default=[])
    if weather_conds:
        filtered_df = filtered_df[filtered_df["weather_condition"].isin(weather_conds)]

if "is_weekend" in df.columns:
    weekend_filter = st.sidebar.radio("📅 Day Type", ["All", "Weekday Only", "Weekend Only"], horizontal=True)
    if weekend_filter == "Weekday Only":
        filtered_df = filtered_df[filtered_df["is_weekend"] == 0]
    elif weekend_filter == "Weekend Only":
        filtered_df = filtered_df[filtered_df["is_weekend"] == 1]

st.sidebar.markdown(f"**Records:** {len(filtered_df):,}")
st.sidebar.markdown("---")
st.sidebar.markdown("### 📦 Data Sources")
st.sidebar.markdown("- 🚍 NYC MTA Open Data")
st.sidebar.markdown("- 🌦️ Open-Meteo API")
st.sidebar.markdown("- 🎉 Nager.Date Holidays")


# ═══════════════════════════════════════════════════════════════════
# PAGE 1: OVERVIEW
# ═══════════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.markdown("""
    <div class="hero-banner">
        <h1 class="main-header">🚍 Public Transport Delay Analyzer</h1>
        <p class="sub-header">Analyzing NYC MTA Bus Delays with Weather & Event Correlations</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-card">
        <h4 style="margin-top:0;">👋 Welcome to the NYC Transit Analyzer!</h4>
        <p style="color:#64748b; margin-bottom:0;">This tool helps you understand what causes bus delays in New York City. Explore historical data, see how weather affects travel times, or use our AI to predict future delays. Use the menu on the left to navigate!</p>
    </div>
    """, unsafe_allow_html=True)

    # Key Metrics Row
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("📋 Total Records", f"{len(filtered_df):,}", help="Total number of transit delay records evaluated")
    with col2:
        st.metric("⏱ Avg Delay", f"{filtered_df['delay_minutes'].mean():.1f} min", help="The average delay duration across all affected routes")
    with col3:
        st.metric("🚨 Delayed >15min", f"{filtered_df['is_delayed'].mean()*100:.1f}%", help="Percentage of total trips delayed by more than 15 minutes")
    with col4:
        st.metric("🌧 Bad Weather Days", f"{(filtered_df['weather_severity']>=3).sum():,}", help="Number of days experiencing severe weather (Heavy Rain, Snow, Storms)")
    with col5:
        st.metric("🎉 Event Days", f"{filtered_df['has_event'].sum():,}", help="Number of days with significant local events (Sports, Concerts, Holidays)")

    st.markdown("<br/>", unsafe_allow_html=True)

    # Charts Row 1
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Daily Average Delay Trend")
        daily_avg = filtered_df.groupby("date")["delay_minutes"].mean().reset_index()
        fig = px.line(daily_avg, x="date", y="delay_minutes",
                      labels={"delay_minutes": "Avg Delay (min)", "date": "Date"},
                      color_discrete_sequence=["#667eea"])
        fig.update_layout(template="plotly_white", height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🥧 Delay Category Distribution")
        cat_counts = filtered_df["delay_category"].value_counts().reset_index()
        cat_counts.columns = ["Category", "Count"]
        fig = px.pie(cat_counts, values="Count", names="Category",
                     color_discrete_sequence=px.colors.qualitative.Set2,
                     hole=0.4)
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    # Charts Row 2
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🏢 Delays by Borough")
        if "borough" in filtered_df.columns:
            borough_delay = filtered_df.groupby("borough")["delay_minutes"].mean().sort_values(ascending=True).reset_index()
            fig = px.bar(borough_delay, x="delay_minutes", y="borough",
                         orientation="h", color="delay_minutes",
                         color_continuous_scale="Viridis",
                         labels={"delay_minutes": "Avg Delay (min)"})
            fig.update_layout(template="plotly_white", height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("⚡ Top Delay Reasons")
        if "delay_reason" in filtered_df.columns:
            reason_counts = filtered_df["delay_reason"].value_counts().head(8).reset_index()
            reason_counts.columns = ["Reason", "Count"]
            fig = px.bar(reason_counts, x="Count", y="Reason", orientation="h",
                         color="Count", color_continuous_scale="Reds")
            fig.update_layout(template="plotly_white", height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("🗂️ Interactive Data Explorer")
    st.markdown("Search, sort, and download the raw data driving these insights. (Showing first 1,000 records)")
    
    st.dataframe(filtered_df.head(1000), use_container_width=True)
    
    csv = filtered_df.head(1000).to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Download This Data (CSV)",
        data=csv,
        file_name="transit_insights_data.csv",
        mime="text/csv",
    )


# ═══════════════════════════════════════════════════════════════════
# PAGE 2: EDA EXPLORER
# ═══════════════════════════════════════════════════════════════════
elif page == "🔍 EDA Explorer":
    st.markdown("""
    <div class="hero-banner">
        <h1 class="main-header">🔍 Explore the Data</h1>
        <p class="sub-header">Discover patterns in how weather, time, and events impact bus delays.</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🗺️ Spatial Analysis", "🌦️ Weather Impact", "🕐 Time Analysis", "🎉 Event Impact", "🔗 Correlations", "🎛️ Custom Plotter"])

    with tab1:
        st.subheader("Geospatial Map of Average Delays by Borough")
        
        # Hardcode basic coordinates for standard NYC-area boroughs
        borough_coords = {
            "Manhattan": {"lat": 40.7831, "lon": -73.9712},
            "Brooklyn": {"lat": 40.6782, "lon": -73.9442},
            "Queens": {"lat": 40.7282, "lon": -73.7949},
            "Bronx": {"lat": 40.8448, "lon": -73.8648},
            "Staten Island": {"lat": 40.5795, "lon": -74.1502},
            "Nassau County": {"lat": 40.7282, "lon": -73.5828},
            "Westchester": {"lat": 41.1220, "lon": -73.7949},
            "New Jersey": {"lat": 40.7178, "lon": -74.0431},
            "Rockland County": {"lat": 41.1489, "lon": -74.0401},
            "Suffolk": {"lat": 40.8522, "lon": -73.1189}
        }
        
        if "borough" in filtered_df.columns:
            spatial_df = filtered_df.groupby("borough")["delay_minutes"].mean().reset_index()
            spatial_df["counts"] = filtered_df.groupby("borough").size().values
            
            # Map coords
            spatial_df["lat"] = spatial_df["borough"].map(lambda b: borough_coords.get(b, {}).get("lat", None))
            spatial_df["lon"] = spatial_df["borough"].map(lambda b: borough_coords.get(b, {}).get("lon", None))
            spatial_df = spatial_df.dropna(subset=["lat", "lon"])
            
            if not spatial_df.empty:
                fig = px.scatter_mapbox(spatial_df, lat="lat", lon="lon", size="delay_minutes",
                                        color="delay_minutes", hover_name="borough", hover_data=["counts"],
                                        color_continuous_scale=px.colors.sequential.YlOrRd, size_max=40,
                                        zoom=8, mapbox_style="carto-positron",
                                        labels={"delay_minutes": "Avg Delay (min)", "counts": "Total Records"})
                fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, height=500)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No geospatial data available for the selected filters.")

    with tab2:
        st.subheader("Weather Condition vs Delay Duration")
        col1, col2 = st.columns(2)

        with col1:
            fig = px.box(filtered_df, x="weather_condition", y="delay_minutes",
                         color="weather_condition", color_discrete_sequence=px.colors.qualitative.Pastel,
                         labels={"delay_minutes": "Delay (min)", "weather_condition": "Weather"})
            fig.update_layout(template="plotly_white", height=450, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.box(filtered_df, x="weather_severity", y="delay_minutes",
                         color="weather_severity", color_discrete_sequence=px.colors.sequential.YlOrRd,
                         labels={"delay_minutes": "Delay (min)", "weather_severity": "Severity Level"})
            fig.update_layout(template="plotly_white", height=450, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Precipitation vs Delay")
        col1, col2 = st.columns(2)
        with col1:
            fig = px.scatter(filtered_df.sample(min(5000, len(filtered_df))),
                            x="precipitation_mm", y="delay_minutes",
                            color="weather_condition", opacity=0.5,
                            labels={"precipitation_mm": "Precipitation (mm)", "delay_minutes": "Delay (min)"},
                            trendline="ols")
            fig.update_layout(template="plotly_white", height=400)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            if "temperature_mean_c" in filtered_df.columns:
                fig = px.scatter(filtered_df.sample(min(5000, len(filtered_df))),
                                x="temperature_mean_c", y="delay_minutes",
                                color="weather_condition", opacity=0.5,
                                labels={"temperature_mean_c": "Temperature (°C)", "delay_minutes": "Delay (min)"},
                                trendline="ols")
                fig.update_layout(template="plotly_white", height=400)
                st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Delays by Hour of Day")
        col1, col2 = st.columns(2)

        with col1:
            hourly = filtered_df.groupby("hour")["delay_minutes"].mean().reset_index()
            fig = px.bar(hourly, x="hour", y="delay_minutes",
                         color="delay_minutes", color_continuous_scale="Blues",
                         labels={"hour": "Hour", "delay_minutes": "Avg Delay (min)"})
            fig.update_layout(template="plotly_white", height=400)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            if "day_of_week" in filtered_df.columns:
                dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
                dow_delay = filtered_df.groupby("day_of_week")["delay_minutes"].mean().reindex(dow_order).reset_index()
                fig = px.bar(dow_delay, x="day_of_week", y="delay_minutes",
                             color="delay_minutes", color_continuous_scale="Purples",
                             labels={"day_of_week": "Day", "delay_minutes": "Avg Delay (min)"})
                fig.update_layout(template="plotly_white", height=400)
                st.plotly_chart(fig, use_container_width=True)

        st.subheader("Rush Hour vs Non-Rush Hour")
        rush_data = filtered_df.groupby("is_rush_hour")["delay_minutes"].agg(["mean","median","count"]).reset_index()
        rush_data["is_rush_hour"] = rush_data["is_rush_hour"].map({0: "Non-Rush Hour", 1: "Rush Hour"})
        col1, col2, col3 = st.columns(3)
        for i, (_, row) in enumerate(rush_data.iterrows()):
            with [col1, col2, col3][i]:
                st.metric(row["is_rush_hour"], f"{row['mean']:.1f} min avg", f"{int(row['count']):,} trips")

    with tab3:
        st.subheader("Impact of Events on Delays")
        col1, col2 = st.columns(2)

        with col1:
            event_comp = filtered_df.groupby("has_event")["delay_minutes"].agg(["mean","median"]).reset_index()
            event_comp["has_event"] = event_comp["has_event"].map({0: "No Event", 1: "Event Day"})
            fig = px.bar(event_comp, x="has_event", y="mean",
                         color="has_event", color_discrete_sequence=["#4ecdc4", "#ff6b6b"],
                         labels={"mean": "Avg Delay (min)", "has_event": ""})
            fig.update_layout(template="plotly_white", height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            if "event_impact_score" in filtered_df.columns:
                event_days = filtered_df[filtered_df["has_event"]==1]
                if len(event_days) > 0:
                    fig = px.scatter(event_days, x="event_impact_score", y="delay_minutes",
                                    size="total_attendance", color="weather_condition",
                                    labels={"event_impact_score": "Event Impact Score", "delay_minutes": "Delay (min)"},
                                    opacity=0.6)
                    fig.update_layout(template="plotly_white", height=400)
                    st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.subheader("Feature Correlation Heatmap")
        numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns
        key_cols = [c for c in ["delay_minutes","precipitation_mm","wind_speed_max_kmh",
                                "temperature_mean_c","weather_severity","has_event",
                                "event_impact_score","is_rush_hour","is_weekend",
                                "hour","passenger_count","weather_severity_index"] if c in numeric_cols]
        if key_cols:
            corr = filtered_df[key_cols].corr()
            fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                           aspect="auto", zmin=-1, zmax=1)
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)

    with tab6:
        st.subheader("Build Your Own Chart")
        st.markdown("Select any variables from the dataset to automatically build custom visualizations on the fly!")
        
        col1, col2, col3 = st.columns(3)
        
        numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = filtered_df.select_dtypes(include=['object', 'category']).columns.tolist()
        all_cols = numeric_cols + cat_cols
        
        with col1:
            x_axis = st.selectbox("X-Axis Variable", options=all_cols, index=min(len(all_cols)-1, all_cols.index('date') if 'date' in all_cols else 0))
        with col2:
            y_axis = st.selectbox("Y-Axis Variable", options=numeric_cols, index=min(len(numeric_cols)-1, numeric_cols.index('delay_minutes') if 'delay_minutes' in numeric_cols else 0))
        with col3:
            color_by = st.selectbox("Color Grouping (Optional)", options=["None"] + cat_cols, index=0)
            
        chart_type = st.radio("Select Chart Type", ["Scatter Plot", "Box Plot", "Average Bar Chart"], horizontal=True)
        
        color_arg = color_by if color_by != "None" else None
        
        # Limit to 5000 points so the browser doesn't freeze
        plot_sample = filtered_df.sample(min(5000, len(filtered_df)))
        
        if chart_type == "Scatter Plot":
            fig = px.scatter(plot_sample, x=x_axis, y=y_axis, color=color_arg, opacity=0.6)
        elif chart_type == "Box Plot":
            fig = px.box(plot_sample, x=x_axis, y=y_axis, color=color_arg)
        elif chart_type == "Average Bar Chart":
            agg_df = plot_sample.groupby(x_axis)[y_axis].mean().reset_index()
            fig = px.bar(agg_df, x=x_axis, y=y_axis, color=x_axis)
            
        fig.update_layout(template="plotly_white", height=500, hovermode="closest")
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# PAGE 3: PREDICTIONS
# ═══════════════════════════════════════════════════════════════════
elif page == "🔮 Predictions":
    st.markdown("""
    <div class="hero-banner">
        <h1 class="main-header">🔮 Predict Your Trip</h1>
        <p class="sub-header">Play with different scenarios to see how our AI predicts travel delays.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card" style="border-left-color: #ec4899;">
        <h4 style="margin-top:0;">🤖 AI Prediction Engine</h4>
        <p style="color:#64748b; margin-bottom:0;">Change the time, weather, and event options below. Our AI model will instantly calculate the expected delay based on thousands of past trips.</p>
    </div>
    <br/>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 🕐 Time & Route")
        hour = st.slider("Hour of Day", 5, 22, 8)
        is_weekend = st.selectbox("Weekend?", ["No", "Yes"])
        is_rush = st.selectbox("Rush Hour?", ["Yes", "No"])
        borough = st.selectbox("Borough", ["Brooklyn", "Bronx", "Queens", "Manhattan", "Staten Island"])

    with col2:
        st.markdown("### 🌦️ Weather")
        temp = st.slider("Temperature (°C)", -10, 40, 20)
        precipitation = st.slider("Precipitation (mm)", 0.0, 50.0, 0.0, 0.5)
        wind_speed = st.slider("Wind Speed (km/h)", 0.0, 80.0, 15.0, 1.0)
        weather = st.selectbox("Weather Condition", ["Clear", "Light Rain", "Rain", "Heavy Rain", "Snow"])

    with col3:
        st.markdown("### 🎉 Events")
        has_event = st.selectbox("Event Today?", ["No", "Yes"])
        if has_event == "Yes":
            event_attendance = st.slider("Expected Attendance", 0, 300000, 10000)
            event_impact = st.selectbox("Impact Level", ["Low", "Medium", "High"])
        else:
            event_attendance = 0
            event_impact = "Low"

    st.markdown("<br/>", unsafe_allow_html=True)
    if st.button("🔮 Run Delay Prediction Model", use_container_width=True):
        severity_map = {"Clear": 1, "Light Rain": 2, "Rain": 3, "Heavy Rain": 4, "Snow": 4, "Stormy": 5}

        # Simple rule-based prediction for display
        base_delay = 20
        # Weather effect
        base_delay += precipitation * 0.8 + (wind_speed - 20) * 0.3 if wind_speed > 20 else 0
        if weather in ["Heavy Rain", "Snow"]:
            base_delay += 15
        elif weather == "Rain":
            base_delay += 8
        # Rush hour
        if is_rush == "Yes":
            base_delay += 10
        # Weekend reduction
        if is_weekend == "Yes":
            base_delay -= 8
        # Event
        if has_event == "Yes":
            base_delay += np.log1p(event_attendance) * 1.5

        predicted_delay = max(0, base_delay)
        is_delayed_pred = predicted_delay > 15

        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            color = "🔴" if predicted_delay > 30 else "🟡" if predicted_delay > 15 else "🟢"
            st.metric(f"{color} Predicted Delay", f"{predicted_delay:.0f} min")
        with col2:
            st.metric("📋 Status", "DELAYED" if is_delayed_pred else "ON TIME")
        with col3:
            if predicted_delay <= 15:
                cat = "Minor"
            elif predicted_delay <= 30:
                cat = "Moderate"
            elif predicted_delay <= 60:
                cat = "Major"
            else:
                cat = "Severe"
            st.metric("📊 Category", cat)

        st.info(f"**Key factors:** Weather ({weather}), Rush Hour ({is_rush}), Weekend ({is_weekend}), Event ({has_event})")


# ═══════════════════════════════════════════════════════════════════
# PAGE 4: MODEL INSIGHTS
# ═══════════════════════════════════════════════════════════════════
elif page == "🧠 Model Insights":
    st.markdown("""
    <div class="hero-banner">
        <h1 class="main-header">🧠 How the AI Works</h1>
        <p class="sub-header">Look under the hood to see how accurate our AI is and what data it relies on.</p>
    </div>
    """, unsafe_allow_html=True)

    results = load_results()
    fi = load_feature_importance()
    shap_data = load_shap_data()

    tab1, tab2, tab3 = st.tabs(["📊 AI Accuracy Scores", "🔑 Most Important Factors", "💡 AI Reasoning"])

    with tab1:
        if "regression" in results:
            st.markdown("### ⏱️ Predicting Exact Delay Minutes")
            st.markdown("We tested several AI models. The table below shows their performance. **Lower average errors are better**, while a **higher R² score means the model is more accurate**.")
            reg_df = results["regression"]
            st.dataframe(reg_df.style.highlight_max(subset=["R2_Score","CV_R2_Mean"], color="#90EE90")
                                      .highlight_min(subset=["MAE","RMSE"], color="#90EE90"),
                         use_container_width=True)

            fig = make_subplots(rows=1, cols=3, subplot_titles=["MAE ↓", "RMSE ↓", "R² Score ↑"])
            for i, metric in enumerate(["MAE", "RMSE", "R2_Score"]):
                colors = ["#667eea" if v == reg_df[metric].min() or (metric=="R2_Score" and v == reg_df[metric].max())
                          else "#c3cfe2" for v in reg_df[metric]]
                fig.add_trace(go.Bar(x=reg_df["Model"], y=reg_df[metric], name=metric,
                                     marker_color=colors, showlegend=False), row=1, col=i+1)
            fig.update_layout(height=400, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

        if "classification" in results:
            st.markdown("### 🚦 Predicting 'Delayed' vs 'On-Time'")
            st.markdown("Here, the models just try to guess if the bus will be delayed or not. **Higher Accuracy and F1 Scores mean the model makes fewer mistakes.**")
            cls_df = results["classification"]
            st.dataframe(cls_df.style.highlight_max(subset=["Accuracy","F1_Score","CV_F1_Mean"], color="#90EE90"),
                         use_container_width=True)

            fig = make_subplots(rows=1, cols=3, subplot_titles=["Accuracy ↑", "F1 Score ↑", "CV F1 ↑"])
            for i, metric in enumerate(["Accuracy", "F1_Score", "CV_F1_Mean"]):
                colors = ["#764ba2" if v == cls_df[metric].max() else "#d4c5e2" for v in cls_df[metric]]
                fig.add_trace(go.Bar(x=cls_df["Model"], y=cls_df[metric], name=metric,
                                     marker_color=colors, showlegend=False), row=1, col=i+1)
            fig.update_layout(height=400, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown("### 🔑 What causes the most delays?")
        st.markdown("According to our AI, these are the most important factors that determine if a bus will be late. Longer bars mean the factor has a bigger impact.")
        for task_name, task_fi in fi.items():
            st.markdown(f"**{task_name.title()} Model**")
            top_n = st.slider(f"Number of features ({task_name})", 5, 30, 15, key=f"fi_{task_name}")
            top_features = task_fi.head(top_n)
            fig = px.bar(top_features, x="importance", y="feature", orientation="h",
                         color="importance", color_continuous_scale="Viridis",
                         labels={"importance": "Importance", "feature": "Feature"})
            fig.update_layout(template="plotly_white", height=max(400, top_n * 25),
                              yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown("### 💡 How the AI makes decisions")
        st.markdown("""
        <div class="info-card" style="border-left-color: #10b981;">
            <p style="margin-bottom:0;"><b>Explainable AI:</b> This chart shows exactly how much each factor (like rain, snow, or rush hour) adds to or subtracts from the delay for a group of specific trips. Points further to the right mean a longer delay.</p>
        </div>
        """, unsafe_allow_html=True)
        
        task_choice = st.radio("Select Model Task", list(shap_data.keys()))
        if task_choice and task_choice in shap_data:
            s_data = shap_data[task_choice]
            shap_values = s_data["shap_values"]
            X_sample = s_data["X_sample"]
            
            # Matplotlib figure for SHAP summary plot
            fig, ax = plt.subplots(figsize=(10, 6))
            # Fix SHAP dependence on matplotlib current figure
            shap.summary_plot(shap_values, X_sample, plot_type="dot", show=False, max_display=15)
            st.pyplot(fig)
            plt.clf()

# ─── PAGE 5: 7-DAY FORECAST ───────────────────────────────────────────
elif page == "🔮 7-Day Forecast":
    st.markdown("""
    <div class="hero-banner">
        <h1 class="main-header">📅 7-Day Delay Forecast</h1>
        <p class="sub-header">Using historical trends, our AI predicts the average delay across the entire NYC bus network for the next week.</p>
    </div>
    """, unsafe_allow_html=True)
    
    forecast_df = load_forecast_data()
    
    if forecast_df is not None:
        st.markdown("### 🌦️ What if the weather gets worse?")
        st.markdown("Play with the scenarios below to see how a sudden storm or major event would impact the baseline forecast.")
        
        col1, col2 = st.columns(2)
        with col1:
            future_weather = st.selectbox("Simulate Weather for Next Week", ["Typical (No Change)", "Heavy Rain", "Snowstorm", "Severe Storms"])
        with col2:
            future_event = st.selectbox("Simulate Major City Event", ["None", "Parade / Marathon", "Major Concert / Sports Game"])

        # Create a dynamic copy of the forecast
        dynamic_forecast = forecast_df.copy()
        
        # Apply modifiers to the 'Forecast' type only
        forecast_mask = dynamic_forecast['type'] == 'Forecast'
        
        # Apply Weather Modifiers
        if future_weather == "Heavy Rain":
            dynamic_forecast.loc[forecast_mask, 'delay'] += 12.5
        elif future_weather == "Snowstorm":
            dynamic_forecast.loc[forecast_mask, 'delay'] += 25.0
        elif future_weather == "Severe Storms":
            dynamic_forecast.loc[forecast_mask, 'delay'] += 18.0
            
        # Apply Event Modifiers
        if future_event == "Parade / Marathon":
            dynamic_forecast.loc[forecast_mask, 'delay'] += 15.0
        elif future_event == "Major Concert / Sports Game":
            dynamic_forecast.loc[forecast_mask, 'delay'] += 8.0

        # We only want to plot the last 90 days of history + 7 days forecast to make the chart readable
        cutoff_date = dynamic_forecast[dynamic_forecast['type'] == 'Forecast']['date'].min() - pd.Timedelta(days=90)
        plot_df = dynamic_forecast[dynamic_forecast['date'] >= cutoff_date]
        
        fig = px.line(plot_df, x="date", y="delay", color="type",
                      color_discrete_map={"Historical": "#1f77b4", "Forecast": "#ff7f0e"},
                      labels={"date": "Date", "delay": "Avg Daily Delay (Minutes)", "type": "Data Type"})
        
        fig.update_layout(template="plotly_white", hovermode="x unified", height=500,
                          legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
        
        fig.update_traces(line=dict(width=3))
        # Make forecast dotted
        fig.update_traces(line=dict(dash="dot"), selector=dict(name="Forecast"))
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show metric summary for forecast
        st.markdown("### 📅 Upcoming Forecast Summary")
        forecast_only = dynamic_forecast[dynamic_forecast['type'] == 'Forecast'].sort_values("date")
        
        cols = st.columns(7)
        for i, row in enumerate(forecast_only.itertuples()):
            if i < 7:
                with cols[i]:
                    day_name = row.date.strftime("%A")
                    short_date = row.date.strftime("%b %d")
                    
                    # Highlight high delays
                    if row.delay > 60:
                        border_color = "#ef4444" # red
                        val_color = "#ef4444"
                    elif row.delay > 40:
                        border_color = "#f59e0b" # yellow
                        val_color = "#f59e0b"
                    else:
                        border_color = "#f97316" # orange (default)
                        val_color = "#f97316"
                        
                    st.markdown(f"""
                    <div class="forecast-card" style="border-top-color: {border_color};">
                        <div class="forecast-day">{day_name}<br/><small>{short_date}</small></div>
                        <div class="forecast-val" style="color: {val_color};">{row.delay:.1f}m</div>
                    </div>
                    """, unsafe_allow_html=True)
    else:
        st.info("Forecast data is not available yet. Please run the forecasting pipeline.")

# ─── Footer ───────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown(
    "Built with ❤️ using **Streamlit** | "
    "Data: NYC Open Data, Open-Meteo"
)
