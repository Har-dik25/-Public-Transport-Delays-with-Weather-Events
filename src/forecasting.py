import pandas as pd
import numpy as np
import os
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import joblib

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

def run_forecasting():
    print("\n" + "="*60)
    print("📈 TIME SERIES FORECASTING (7-Day)")
    print("="*60)
    
    # 1. Load Data
    filepath = os.path.join(PROCESSED_DIR, "merged_dataset.csv")
    df = pd.read_csv(filepath, parse_dates=["date"])
    
    # 2. Aggregate Daily Delays
    daily_delays = df.groupby(df['date'].dt.date)['delay_minutes'].mean().reset_index()
    daily_delays['date'] = pd.to_datetime(daily_delays['date'])
    daily_delays = daily_delays.sort_values('date').set_index('date')
    
    # Ensure continuous date frequency
    idx = pd.date_range(daily_delays.index.min(), daily_delays.index.max(), freq='D')
    daily_delays = daily_delays.reindex(idx)
    # Forward fill any missing days
    daily_delays['delay_minutes'] = daily_delays['delay_minutes'].ffill().bfill()
    
    print(f"✅ Aggregated {len(daily_delays)} days of historical delay data.")
    
    # 3. Train Holt-Winters Exponential Smoothing Model
    # Since transport delays usually have weekly seasonality (7 days)
    model = ExponentialSmoothing(
        daily_delays['delay_minutes'], 
        seasonal_periods=7, 
        trend='add', 
        seasonal='add', 
        initialization_method="estimated"
    )
    fitted_model = model.fit()
    print("✅ ExponentialSmoothing Model Trained.")
    
    # 4. Forecast next 7 days
    forecast_steps = 7
    forecast = fitted_model.forecast(forecast_steps)
    
    # 5. Prepare Output DataFrame
    historical_df = daily_delays.reset_index().rename(columns={'index': 'date', 'delay_minutes': 'delay'})
    historical_df['type'] = 'Historical'
    
    forecast_df = pd.DataFrame({
        'date': forecast.index,
        'delay': forecast.values,
        'type': 'Forecast'
    })
    
    final_df = pd.concat([historical_df, forecast_df], ignore_index=True)
    
    # Save the output
    out_path = os.path.join(PROCESSED_DIR, "delay_forecast.csv")
    final_df.to_csv(out_path, index=False)
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "forecasting_model.pkl")
    joblib.dump(fitted_model, model_path)
    
    print(f"💾 Forecast saved -> {out_path}")
    print(f"💾 Model saved -> {model_path}")
    print("============================================================\n")

if __name__ == "__main__":
    run_forecasting()
