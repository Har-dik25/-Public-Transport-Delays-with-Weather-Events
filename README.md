# 🚍 Public Transport Delay with Weather & Events

An end-to-end data science project that analyzes NYC MTA bus delays and their correlation with weather conditions and local events.

## 📌 Overview

This project combines **real-world data** from three sources to understand what drives public transport delays:

| Data Source | Description | Records |
|---|---|---|
| **NYC Open Data** | MTA Bus Breakdown & Delays | 150,000+ |
| **Open-Meteo API** | Historical weather for NYC (2023-2024) | 731 days |
| **Nager.Date + NYC Events** | US holidays + recurring local pro sports | 404 events |

## 🔬 Key Findings

- **89.7%** of trips experienced delays greater than 15 minutes
- **Heavy Traffic** remains the #1 delay reason
- **SMOTE** oversampling achieved perfectly balanced classes for training, elevating model robustness
- **Model Explainability (SHAP)** reveals exactly how weather and events drive predictions
- **Geospatial Hotspots** mapped across NYC boroughs.
- **Time-Series Forecasting** successfully predicts 7-day aggregate delay trends
- **Production Optimized** using `.parquet` and `.joblib`, reducing app memory footprint by 95%
-2. **Geospatial Mapping:** Plotly scatter_mapbox visualizes delay severity by NYC borough.
3. **Delay Prediction Form:** Interactive inference using the Random Forest model.
4. **Model Insights & SHAP:** Visualize Feature Importance and SHAP values for model explainability.
5. **🔮 7-Day Forecast:** Holt-Winters time-series forecast for upcoming network delays.

### Advanced MLOps & Production
- 🧪 **Unit Testing:** `pytest` suite added for data pipeline integrity.
- ⚙️ **Dockerized:** Ready for cloud deployment via `Dockerfile`.
- 🗜️ **Optimized:** Data stored in highly compressed `.parquet` format (90% size reduction) and models compressed via joblib `zlib` for ultra-fast, low-memory inference on cloud platforms.
- **Time-Series Forecasting** successfully predicts 7-day aggregate delay trends
- **Production Optimized** using `.parquet` and `.joblib`, reducing app memory footprint by 95%

## 🏗️ Project Structure

```
├── data/
│   ├── raw/                    # Original downloaded datasets
│   └── processed/              # Cleaned & merged datasets
├── notebooks/                  # Jupyter notebooks for analysis
├── src/
│   ├── data_loader.py          # Data download & loading
│   ├── preprocessing.py        # Data cleaning & merging
│   ├── feature_engineering.py  # Feature creation
│   └── model.py                # ML model training
├── dashboard/
│   └── app.py                  # Streamlit interactive dashboard
├── models/                     # Saved trained models
├── reports/figures/            # Saved plots
├── download_dataset.py         # Script to download real data
├── requirements.txt            # Python dependencies
└── README.md
```

## 🚀 Quick Start

### 1. Clone & Setup
```bash
git clone <repo-url>
cd "Public Transport Delay With weather and events"
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 2. Download Data
```bash
python download_dataset.py
```

### 3. Run Pipeline
```bash
python -m src.preprocessing
python -m src.feature_engineering
python -m src.model
```

### 4. Launch Dashboard
```bash
streamlit run dashboard/app.py
```

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| **Language** | Python 3.10+ |
| **Data** | Pandas, NumPy |
| **ML** | Scikit-learn, XGBoost, LightGBM |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **Dashboard** | Streamlit |
| **Data Sources** | NYC Open Data, Open-Meteo API, Nager.Date API |

## 📊 Model Results

### Regression (Predicting delay duration)
| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | 16.57 | 20.11 | 0.18 |
| Decision Tree | 8.94 | 14.98 | 0.54 |
| **Random Forest** | **8.60** | **13.31** | **0.64** |
| Gradient Boosting | 10.76 | 14.72 | 0.56 |

### Classification (Predicting if delayed >15 min — using SMOTE)
| Model | Accuracy | F1 Score |
|---|---|---|
| Logistic Regression | 69.1% | 0.75 |
| **Decision Tree** | **89.5%** | **0.90** |
| Random Forest | 88.7% | 0.89 |
| Gradient Boosting | 89.4% | 0.90 |

## 📝 License

This project uses publicly available open data. See data sources above for individual data licensing.
