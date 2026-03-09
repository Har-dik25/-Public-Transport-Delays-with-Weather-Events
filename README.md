<div align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/Plotly-239120?style=for-the-badge&logo=plotly&logoColor=white" />
  <img src="https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white" />
</div>

<h1 align="center">🚍 TransitAI: Public Transport Delay Analyzer</h1>

<p align="center">
  <b>An end-to-end Machine Learning project analyzing and predicting NYC MTA bus delays using historical weather and city event data.</b>
</p>

---

## 📖 What the project does

**TransitAI** is a comprehensive data science portfolio project built to tackle a real-world problem: predicting public transportation delays before they happen.

The project ingests simulated records of NYC MTA bus trips and enriches them with historical weather data (from Open-Meteo) and major city events (concerts, marathons). Using supervised Machine Learning models (Random Forest, Gradient Boosting), the system learns the complex correlations between heavy rain, rush hour, and major events to accurately predict whether a future bus trip will be delayed, and by exactly how many minutes.

Finally, the project exposes these insights through a highly interactive, beautifully designed **Streamlit Dashboard** that allows users to explore the raw data, build custom charts, and run "What-If" predictive scenarios.

## 🚀 Why the project is useful

1. **For Commuters:** Provides actionable insights into how severe weather or city-wide events might compound their travel time.
2. **For Transit Authorities:** Demonstrates how explainable AI (using SHAP values) can be used to isolate the root causes of systemic transport unreliability.
3. **For Data Science Portfolios:** Showcases an applicant's ability to build an entire end-to-end MLOps pipeline. It features automated tests (`pytest`), efficient data storage (`.parquet`), code modularity, explainable AI (`SHAP`), and robust visualizations.

## 🛠️ How to get started

### 1. Prerequisites
Ensure you have **Python 3.10+** installed on your system. 

### 2. Clone the Repository
```bash
git clone https://github.com/Har-dik25/-Public-Transport-Delays-with-Weather-Events.git
cd -Public-Transport-Delays-with-Weather-Events
```

### 3. Install Dependencies
It is recommended to use a virtual environment.
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Run the Data Pipeline (Optional)
The pre-processed data and trained ML models are already included in the repository, but you can regenerate them from scratch using the included Makefile:
```bash
# 1. Download raw data sets and merge them
make data

# 2. Create ML features (Rush hour, Severity Index, etc.)
make features

# 3. Train the AI Models & generate explanations (takes ~1-2 mins)
make train
```

### 5. Launch the Intelligence Dashboard
Fire up the Streamlit UI to explore the interactive dashboard:
```bash
python -m streamlit run dashboard/app.py
```
> The dashboard will open automatically in your browser at `http://localhost:8501`.

---

## 🧠 Key Features of the Dashboard

* **📊 Overview Metrics:** High-level KPIs showing total records, average delays, and percentage of trips delayed by bad weather.
* **🎛️ Interactive Data Explorer:** Sort, filter, and download the raw `.parquet` dataset powering the ML engine natively inside the app.
* **🔍 Custom Plotter:** Build your own interactive Scatter Plots, Box Plots, and Bar Charts on the fly by selecting X, Y, and Color variables.
* **🔮 Trip Predictor:** A "What-If" engine where you can dial in specific weather temperatures, rainfall amounts, and event attendance parameters to generate an instant delay prediction.
* **💡 Model Insights (SHAP):** An Explainable AI visualization layer that empirically proves *why* the model made its decision, isolating the exact minute penalty added by "Heavy Rain" vs "Rush Hour".
* **📅 7-Day Forecast:** An interactive preview of the week ahead, capable of injecting hypothetical "Snowstorms" to see how the delay graph spikes dynamically.

## 🤝 Getting Help & Contributing 
If you encounter any bugs, have questions, or want to suggest an improvement, please open a new [Issue](https://github.com/Har-dik25/-Public-Transport-Delays-with-Weather-Events/issues) on this repository!

Pull Requests are always welcome.

## 👨‍💻 Maintainers
This project was built and is currently maintained by **[Har-dik25](https://github.com/Har-dik25)**.
