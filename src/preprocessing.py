"""
preprocessing.py — Data Cleaning & Merging for REAL NYC Data
=============================================================
Handles:
  - Cleaning NYC MTA bus delay data
  - Cleaning Open-Meteo weather data
  - Cleaning events/holidays data
  - Merging all three on date
"""

import pandas as pd
import numpy as np
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")


def clean_transport_data(df):
    """Clean NYC MTA Bus Breakdown & Delay data."""
    print("🧹 Cleaning transport data...")
    df = df.copy()

    # Parse dates
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["occurred_on"] = pd.to_datetime(df["occurred_on"], errors="coerce")

    # Drop rows with no date
    df = df.dropna(subset=["date"])

    # Extract just the date part for merging
    df["date_only"] = df["date"].dt.date
    df["date_only"] = pd.to_datetime(df["date_only"])

    # Extract hour from occurred_on
    df["hour"] = df["occurred_on"].dt.hour.fillna(8).astype(int)

    # Convert delay_description to numeric minutes (midpoint of range)
    delay_map = {
        "0-15 Min": 7.5,
        "16-30 Min": 23.0,
        "31-45 Min": 38.0,
        "46-60 Min": 53.0,
        "61-90 Min": 75.5,
    }
    df["delay_minutes"] = df["delay_description"].map(delay_map)
    # For breakdowns without delay description, use median
    df["delay_minutes"] = df["delay_minutes"].fillna(df["delay_minutes"].median())

    # Binary: is_delayed (more than 15 min)
    df["is_delayed"] = (df["delay_minutes"] > 15).astype(int)

    # Delay category
    def categorize_delay(mins):
        if mins <= 15: return "Minor"
        elif mins <= 30: return "Moderate"
        elif mins <= 60: return "Major"
        else: return "Severe"
    df["delay_category"] = df["delay_minutes"].apply(categorize_delay)

    # Day of week features
    df["day_of_week"] = df["date_only"].dt.day_name()
    df["is_weekend"] = df["date_only"].dt.dayofweek.isin([5, 6]).astype(int)
    df["is_rush_hour"] = df["hour"].apply(lambda h: 1 if (7 <= h <= 9) or (16 <= h <= 18) else 0)

    # Clean borough
    df["borough"] = df["borough"].fillna("Unknown").str.strip()

    # Clean delay_reason
    df["delay_reason"] = df["delay_reason"].fillna("Unknown").str.strip()

    # Clean incident_type
    df["incident_type"] = df["incident_type"].fillna("Unknown").str.strip()

    # Clean run_type
    df["run_type"] = df["run_type"].fillna("Unknown").str.strip()

    # Passenger count
    df["passenger_count"] = pd.to_numeric(df["passenger_count"], errors="coerce").fillna(0).astype(int)

    # Remove duplicates
    initial = len(df)
    df = df.drop_duplicates(subset=["busbreakdown_id"])
    removed = initial - len(df)
    if removed > 0:
        print(f"   Removed {removed} duplicate rows")

    # Select and reorder key columns
    keep_cols = [
        "date_only", "hour", "busbreakdown_id", "route_id", "run_type",
        "borough", "bus_company", "delay_reason", "delay_description",
        "delay_minutes", "is_delayed", "delay_category",
        "incident_type", "passenger_count", "day_of_week",
        "is_weekend", "is_rush_hour", "school_age_or_prek"
    ]
    existing = [c for c in keep_cols if c in df.columns]
    df = df[existing]
    df = df.rename(columns={"date_only": "date"})

    print(f"   ✅ Transport cleaned: {len(df)} records, {len(df.columns)} columns")
    print(f"   📅 Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"   📊 Delay distribution: {dict(df['delay_category'].value_counts())}")
    return df


def clean_weather_data(df):
    """Clean Open-Meteo weather data."""
    print("🧹 Cleaning weather data...")
    df = df.copy()

    df["date"] = pd.to_datetime(df["date"])

    # Fill missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].ffill().bfill()

    # Replace any remaining NaN
    df = df.fillna(0)

    # Validate ranges
    df["precipitation_mm"] = df["precipitation_mm"].clip(lower=0)
    df["wind_speed_max_kmh"] = df["wind_speed_max_kmh"].clip(lower=0)

    print(f"   ✅ Weather cleaned: {len(df)} records")
    return df


def clean_events_data(df):
    """Clean events / holiday data."""
    print("🧹 Cleaning events data...")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["impact_level"] = df["impact_level"].str.strip().str.title()
    df["event_type"] = df["event_type"].str.strip()
    df["expected_attendance"] = pd.to_numeric(df["expected_attendance"], errors="coerce").fillna(0).astype(int)
    print(f"   ✅ Events cleaned: {len(df)} events")
    return df


def aggregate_daily_events(events_df):
    """Aggregate events to daily level."""
    print("📊 Aggregating events to daily level...")

    impact_map = {"Low": 1, "Medium": 2, "High": 3}
    events_df = events_df.copy()
    events_df["impact_score"] = events_df["impact_level"].map(impact_map).fillna(0)

    daily = events_df.groupby("date").agg(
        event_count=("event_name", "count"),
        total_attendance=("expected_attendance", "sum"),
        max_attendance=("expected_attendance", "max"),
        max_impact_score=("impact_score", "max"),
        event_types=("event_type", lambda x: ", ".join(x.unique())),
        has_high_impact=("impact_score", lambda x: int((x >= 3).any())),
    ).reset_index()
    daily["has_event"] = 1

    print(f"   ✅ {len(daily)} days with events")
    return daily


def merge_datasets(transport_df, weather_df, events_daily_df):
    """Merge all three datasets on date."""
    print("🔗 Merging datasets...")

    # Merge transport + weather on date
    merged = transport_df.merge(weather_df, on="date", how="left")
    print(f"   After transport+weather: {len(merged)} records")

    # Merge with events
    merged = merged.merge(events_daily_df, on="date", how="left")
    print(f"   After adding events: {len(merged)} records")

    # Fill NaN for days without events
    event_cols = {
        "has_event": 0, "event_count": 0, "total_attendance": 0,
        "max_attendance": 0, "max_impact_score": 0, "has_high_impact": 0,
    }
    for col, fill_val in event_cols.items():
        if col in merged.columns:
            merged[col] = merged[col].fillna(fill_val).astype(int)
    if "event_types" in merged.columns:
        merged["event_types"] = merged["event_types"].fillna("None")

    print(f"   ✅ Final merged: {len(merged)} records, {len(merged.columns)} columns")
    return merged


def run_preprocessing_pipeline():
    """Full pipeline: load → clean → merge → save."""
    print("=" * 65)
    print("🔧 PREPROCESSING PIPELINE (Real Data)")
    print("=" * 65)

    # Load raw
    transport = pd.read_csv(os.path.join(RAW_DIR, "transport_delays.csv"))
    weather = pd.read_csv(os.path.join(RAW_DIR, "weather_data.csv"))
    events = pd.read_csv(os.path.join(RAW_DIR, "events_data.csv"))
    print(f"📂 Loaded: Transport={len(transport)}, Weather={len(weather)}, Events={len(events)}\n")

    # Clean
    transport_clean = clean_transport_data(transport)
    weather_clean = clean_weather_data(weather)
    events_clean = clean_events_data(events)
    print()

    # Aggregate events daily
    events_daily = aggregate_daily_events(events_clean)
    print()

    # Merge
    merged = merge_datasets(transport_clean, weather_clean, events_daily)

    # Save
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    filepath = os.path.join(PROCESSED_DIR, "merged_dataset.csv")
    merged.to_csv(filepath, index=False)
    print(f"\n💾 Saved → {filepath}")

    # Stats
    print(f"\n📊 Dataset Summary:")
    print(f"   Shape: {merged.shape}")
    print(f"   Date range: {merged['date'].min()} to {merged['date'].max()}")
    print(f"   Avg delay: {merged['delay_minutes'].mean():.1f} min")
    print(f"   Delayed >15min: {merged['is_delayed'].mean()*100:.1f}%")
    print(f"   Days with events: {(merged['has_event']==1).sum()}/{len(merged)} records")

    return merged


if __name__ == "__main__":
    run_preprocessing_pipeline()
