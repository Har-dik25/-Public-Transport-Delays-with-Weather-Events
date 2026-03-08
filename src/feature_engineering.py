"""
feature_engineering.py — Feature Creation for Real NYC Data
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")


def add_time_features(df):
    print("⏰ Adding time features...")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)

    def get_season(m):
        if m in [12,1,2]: return "Winter"
        elif m in [3,4,5]: return "Spring"
        elif m in [6,7,8]: return "Summer"
        else: return "Autumn"
    df["season"] = df["month"].apply(get_season)

    def get_time_period(h):
        if 5<=h<9: return "Early_Morning"
        elif 9<=h<12: return "Morning"
        elif 12<=h<14: return "Midday"
        elif 14<=h<17: return "Afternoon"
        elif 17<=h<20: return "Evening"
        else: return "Night"
    df["time_period"] = df["hour"].apply(get_time_period)

    # Cyclical encoding
    df["hour_sin"] = np.sin(2*np.pi*df["hour"]/24)
    df["hour_cos"] = np.cos(2*np.pi*df["hour"]/24)
    df["month_sin"] = np.sin(2*np.pi*df["month"]/12)
    df["month_cos"] = np.cos(2*np.pi*df["month"]/12)
    df["day_of_week_num"] = df["date"].dt.dayofweek
    print(f"   ✅ Added time features")
    return df


def add_weather_features(df):
    print("🌡️  Adding weather features...")
    df = df.copy()

    # Weather severity index
    precip = df["precipitation_mm"].fillna(0)
    wind = df["wind_speed_max_kmh"].fillna(0)
    snow = df["snowfall_cm"].fillna(0) if "snowfall_cm" in df.columns else 0

    df["weather_severity_index"] = (
        df["weather_severity"]*2 +
        np.log1p(precip)*1.5 +
        np.clip(wind-20, 0, 60)/10 +
        snow*0.5
    ).round(2)

    # Temperature range
    if "temperature_max_c" in df.columns and "temperature_min_c" in df.columns:
        df["temp_range_c"] = df["temperature_max_c"] - df["temperature_min_c"]

    # Temperature category
    temp_col = "temperature_mean_c" if "temperature_mean_c" in df.columns else "temperature_max_c"
    df["temp_category"] = pd.cut(df[temp_col], bins=[-30,0,10,20,30,50],
                                  labels=["Freezing","Cold","Mild","Warm","Hot"])

    # Rain intensity
    df["rain_category"] = pd.cut(precip, bins=[-1,0,2,10,25,200],
                                  labels=["No_Rain","Drizzle","Light_Rain","Moderate_Rain","Heavy_Rain"])

    # Is extreme weather
    df["is_extreme_weather"] = ((df["weather_severity"]>=4) |
                                 (precip>20) | (wind>50)).astype(int)

    # Wind chill / feels like difference
    if "apparent_temp_min_c" in df.columns and "temperature_min_c" in df.columns:
        df["wind_chill_effect"] = df["temperature_min_c"] - df["apparent_temp_min_c"]

    # Is precipitation day
    df["is_rainy_day"] = (precip > 0.5).astype(int)
    df["is_snowy_day"] = (snow > 0.1).astype(int) if isinstance(snow, pd.Series) else 0

    print(f"   ✅ Added weather features")
    return df


def add_event_features(df):
    print("🎪 Adding event features...")
    df = df.copy()

    if "max_impact_score" in df.columns:
        df["event_impact_score"] = (
            df["max_impact_score"]*2 +
            np.log1p(df["total_attendance"]) +
            df["event_count"]*0.5
        ).round(2)
        df.loc[df["has_event"]==0, "event_impact_score"] = 0
    else:
        df["event_impact_score"] = 0

    if "max_attendance" in df.columns:
        df["has_large_event"] = (df["max_attendance"]>10000).astype(int)
    else:
        df["has_large_event"] = 0

    if "event_count" in df.columns:
        df["has_multiple_events"] = (df["event_count"]>1).astype(int)
    else:
        df["has_multiple_events"] = 0

    print(f"   ✅ Added event features")
    return df


def add_lag_features(df):
    print("📈 Adding lag & rolling features...")
    df = df.copy()
    df = df.sort_values(["date","hour"]).reset_index(drop=True)

    # Daily average delay
    daily = df.groupby("date")["delay_minutes"].mean().reset_index()
    daily.columns = ["date","daily_avg_delay"]
    daily = daily.sort_values("date")

    # Lag features
    daily["prev_day_avg_delay"] = daily["daily_avg_delay"].shift(1)
    daily["rolling_3d_avg_delay"] = daily["daily_avg_delay"].rolling(3, min_periods=1).mean().round(2)
    daily["rolling_7d_avg_delay"] = daily["daily_avg_delay"].rolling(7, min_periods=1).mean().round(2)
    daily["delay_trend_3d"] = (daily["daily_avg_delay"] - daily["rolling_3d_avg_delay"]).round(2)

    df = df.merge(daily[["date","prev_day_avg_delay","rolling_3d_avg_delay",
                          "rolling_7d_avg_delay","delay_trend_3d"]], on="date", how="left")

    mean_delay = df["delay_minutes"].mean()
    for col in ["prev_day_avg_delay","rolling_3d_avg_delay","rolling_7d_avg_delay","delay_trend_3d"]:
        df[col] = df[col].fillna(mean_delay if "trend" not in col else 0)

    print(f"   ✅ Added lag & rolling features")
    return df


def encode_features(df):
    print("🔢 Encoding categorical features...")
    df = df.copy()

    cats = ["incident_type","weather_condition","season","time_period",
            "delay_category","rain_category","borough","delay_reason","run_type"]
    existing = [c for c in cats if c in df.columns]

    # Limit high-cardinality columns
    for col in ["borough","delay_reason","run_type"]:
        if col in df.columns:
            top = df[col].value_counts().nlargest(8).index
            df[col] = df[col].where(df[col].isin(top), "Other")

    df_enc = pd.get_dummies(df, columns=existing, drop_first=False, dtype=int)

    # Label encode route_id if present
    if "route_id" in df_enc.columns:
        le = LabelEncoder()
        df_enc["route_id_encoded"] = le.fit_transform(df_enc["route_id"].astype(str))

    print(f"   ✅ Encoded → {len(df_enc.columns)} total columns")
    return df_enc


def run_feature_engineering_pipeline():
    print("="*65)
    print("⚙️  FEATURE ENGINEERING PIPELINE (Real Data)")
    print("="*65)

    filepath = os.path.join(PROCESSED_DIR, "merged_dataset.csv")
    df = pd.read_csv(filepath)
    print(f"📂 Loaded: {len(df)} records, {len(df.columns)} columns\n")

    df = add_time_features(df)
    df = add_weather_features(df)
    df = add_event_features(df)
    df = add_lag_features(df)

    feat_path = os.path.join(PROCESSED_DIR, "featured_dataset.csv")
    df.to_csv(feat_path, index=False)
    print(f"\n💾 Featured dataset → {feat_path}")

    df_enc = encode_features(df)
    enc_path = os.path.join(PROCESSED_DIR, "encoded_dataset.csv")
    df_enc.to_csv(enc_path, index=False)
    print(f"💾 Encoded dataset → {enc_path}")

    orig_cols = len(pd.read_csv(filepath).columns)
    print(f"\n📊 Summary: {orig_cols} → {len(df.columns)} → {len(df_enc.columns)} columns")
    return df, df_enc


if __name__ == "__main__":
    run_feature_engineering_pipeline()
