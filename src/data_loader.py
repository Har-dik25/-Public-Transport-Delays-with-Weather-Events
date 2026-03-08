"""
data_loader.py — Data Loading & Synthetic Data Generation
=========================================================
Generates realistic synthetic datasets for:
  1. Public Transport Delays
  2. Weather Conditions
  3. Local Events
  
The data includes realistic correlations:
  - Bad weather → longer delays
  - Large events → more congestion & delays
  - Rush hours → more delays
  - Weekends → fewer delays
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import random

# ─── Seed for reproducibility ────────────────────────────────────────────────
np.random.seed(42)
random.seed(42)

# ─── Configuration ───────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

# Date range: 2 years of data
START_DATE = datetime(2023, 1, 1)
END_DATE = datetime(2024, 12, 31)

# Routes configuration
ROUTES = {
    "R001": {"name": "Downtown Express", "type": "Bus"},
    "R002": {"name": "Airport Shuttle", "type": "Bus"},
    "R003": {"name": "University Line", "type": "Bus"},
    "R004": {"name": "Harbor Route", "type": "Bus"},
    "R005": {"name": "Suburban Connect", "type": "Bus"},
    "R006": {"name": "Metro Blue Line", "type": "Metro"},
    "R007": {"name": "Metro Red Line", "type": "Metro"},
    "R008": {"name": "Metro Green Line", "type": "Metro"},
    "R009": {"name": "Metro Orange Line", "type": "Metro"},
    "R010": {"name": "Metro Purple Line", "type": "Metro"},
    "R011": {"name": "Central Railway", "type": "Train"},
    "R012": {"name": "Western Express", "type": "Train"},
    "R013": {"name": "Eastern Corridor", "type": "Train"},
    "R014": {"name": "Northern Line", "type": "Train"},
    "R015": {"name": "Southern Express", "type": "Train"},
    "R016": {"name": "Cross-City Bus", "type": "Bus"},
    "R017": {"name": "Night Owl Bus", "type": "Bus"},
    "R018": {"name": "Metro Gold Line", "type": "Metro"},
    "R019": {"name": "Intercity Rail", "type": "Train"},
    "R020": {"name": "Coastal Express", "type": "Train"},
}

# Departure time slots (hourly)
DEPARTURE_HOURS = list(range(5, 24))  # 5 AM to 11 PM

# Event configuration
EVENT_TYPES = ["Concert", "Sports", "Festival", "Protest", "Parade", "Conference"]
VENUES = [
    "City Stadium", "Convention Center", "Central Park", "Town Hall Plaza",
    "Riverside Arena", "Exhibition Grounds", "Cultural Center", "Sports Complex",
    "University Auditorium", "Waterfront Pavilion"
]
EVENT_NAMES = {
    "Concert": ["Summer Music Fest", "Rock Night Live", "Jazz Evening", "Symphony Gala",
                 "Pop Stars Tour", "Indie Showcase", "Country Music Fair", "Classical Night"],
    "Sports": ["City Derby Match", "National League Game", "Marathon", "Cricket Finals",
               "Football Championship", "Tennis Open", "Basketball Tournament", "Athletics Meet"],
    "Festival": ["Food & Wine Festival", "Diwali Celebrations", "Spring Carnival",
                 "Lantern Festival", "Cultural Fair", "Art Exhibition", "Tech Expo",
                 "Heritage Day Festival"],
    "Protest": ["Climate March", "Workers Rally", "Student Protest", "Peace March",
                "Rights Demonstration"],
    "Parade": ["Independence Day Parade", "Victory Parade", "Flower Parade",
               "Holiday Parade", "Pride Parade"],
    "Conference": ["Tech Summit", "Business Forum", "Medical Conference",
                   "Education Symposium", "Startup Week"]
}


def _get_season(month):
    """Return season based on month."""
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Autumn"


def _get_weather_for_date(date):
    """
    Generate realistic weather conditions based on season/month.
    Returns a dictionary of weather attributes.
    """
    month = date.month
    season = _get_season(month)

    # Base temperature by season (Celsius)
    temp_base = {"Winter": 5, "Spring": 18, "Summer": 32, "Autumn": 22}
    temperature = temp_base[season] + np.random.normal(0, 4)
    temperature = round(max(-5, min(45, temperature)), 1)

    # Humidity (higher in monsoon-like months)
    if month in [7, 8, 9]:  # Monsoon-like
        humidity = np.random.uniform(70, 98)
    elif season == "Winter":
        humidity = np.random.uniform(40, 75)
    else:
        humidity = np.random.uniform(30, 80)
    humidity = round(humidity, 1)

    # Rainfall
    rain_prob = {"Winter": 0.15, "Spring": 0.20, "Summer": 0.35, "Autumn": 0.25}
    if np.random.random() < rain_prob[season]:
        if month in [7, 8]:  # Heavy rain months
            rainfall = np.random.exponential(15)
        else:
            rainfall = np.random.exponential(5)
    else:
        rainfall = 0.0
    rainfall = round(max(0, rainfall), 1)

    # Wind speed
    wind_speed = np.random.gamma(2, 5)
    if season == "Winter":
        wind_speed *= 1.3
    wind_speed = round(max(0, min(80, wind_speed)), 1)

    # Visibility
    if rainfall > 20:
        visibility = np.random.uniform(0.5, 3)
    elif rainfall > 5:
        visibility = np.random.uniform(2, 6)
    elif season == "Winter" and np.random.random() < 0.2:  # Fog
        visibility = np.random.uniform(0.2, 2)
    else:
        visibility = np.random.uniform(5, 15)
    visibility = round(visibility, 1)

    # Weather condition
    if rainfall > 20:
        condition = "Storm" if wind_speed > 30 else "Heavy Rain"
        severity = 5 if condition == "Storm" else 4
    elif rainfall > 5:
        condition = "Rain"
        severity = 3
    elif rainfall > 0:
        condition = "Light Rain"
        severity = 2
    elif visibility < 2:
        condition = "Fog"
        severity = 3
    elif temperature < 0:
        condition = "Snow"
        severity = 4
    elif humidity > 85:
        condition = "Cloudy"
        severity = 1
    else:
        condition = "Clear"
        severity = 1

    return {
        "temperature_c": temperature,
        "humidity_pct": humidity,
        "rainfall_mm": rainfall,
        "wind_speed_kmh": wind_speed,
        "visibility_km": visibility,
        "weather_condition": condition,
        "weather_severity": severity
    }


def generate_weather_data():
    """Generate daily weather data for the entire date range."""
    print("🌦️  Generating weather data...")
    dates = pd.date_range(START_DATE, END_DATE, freq="D")
    records = []

    for date in dates:
        weather = _get_weather_for_date(date)
        weather["date"] = date.strftime("%Y-%m-%d")
        records.append(weather)

    df = pd.DataFrame(records)
    # Reorder columns
    df = df[["date", "temperature_c", "humidity_pct", "rainfall_mm",
             "wind_speed_kmh", "visibility_km", "weather_condition", "weather_severity"]]

    filepath = os.path.join(RAW_DIR, "weather_data.csv")
    df.to_csv(filepath, index=False)
    print(f"   ✅ Weather data saved: {len(df)} records → {filepath}")
    return df


def generate_events_data():
    """Generate local events data (2-5 events per week on average)."""
    print("🎉  Generating events data...")
    dates = pd.date_range(START_DATE, END_DATE, freq="D")
    records = []

    for date in dates:
        # More events on weekends and holidays
        is_weekend = date.dayofweek >= 5
        num_events_prob = 0.5 if is_weekend else 0.25

        if np.random.random() < num_events_prob:
            # 1-3 events on this day
            num_events = np.random.choice([1, 2, 3], p=[0.6, 0.3, 0.1])
            for _ in range(num_events):
                event_type = np.random.choice(EVENT_TYPES, p=[0.25, 0.30, 0.15, 0.10, 0.10, 0.10])
                event_name = np.random.choice(EVENT_NAMES[event_type])

                # Attendance based on event type
                attendance_ranges = {
                    "Concert": (500, 25000),
                    "Sports": (2000, 50000),
                    "Festival": (1000, 30000),
                    "Protest": (200, 10000),
                    "Parade": (500, 15000),
                    "Conference": (100, 5000)
                }
                low, high = attendance_ranges[event_type]
                expected_attendance = int(np.random.uniform(low, high))

                # Impact level based on attendance
                if expected_attendance > 15000:
                    impact_level = "High"
                elif expected_attendance > 5000:
                    impact_level = "Medium"
                else:
                    impact_level = "Low"

                # Event timing
                start_hour = np.random.choice(range(8, 20))
                duration = np.random.choice([2, 3, 4, 5, 6])
                end_hour = min(start_hour + duration, 23)

                records.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "event_name": event_name,
                    "event_type": event_type,
                    "expected_attendance": expected_attendance,
                    "impact_level": impact_level,
                    "venue": np.random.choice(VENUES),
                    "start_time": f"{start_hour:02d}:00",
                    "end_time": f"{end_hour:02d}:00"
                })

    df = pd.DataFrame(records)
    filepath = os.path.join(RAW_DIR, "events_data.csv")
    df.to_csv(filepath, index=False)
    print(f"   ✅ Events data saved: {len(df)} records → {filepath}")
    return df


def generate_transport_delays(weather_df, events_df):
    """
    Generate transport delay data with realistic correlations to weather & events.
    
    Correlations built in:
      - Bad weather (rain, storm, fog) → higher delays
      - Large events → more congestion delays
      - Rush hours (7-9 AM, 5-7 PM) → higher delays
      - Weekends → fewer delays
      - Night hours → fewer delays
      - Metro is more reliable than Bus; Train is somewhere in between
    """
    print("🚍  Generating transport delay data...")
    dates = pd.date_range(START_DATE, END_DATE, freq="D")
    records = []

    # Create weather lookup
    weather_lookup = {}
    for _, row in weather_df.iterrows():
        weather_lookup[row["date"]] = row

    # Create events lookup (aggregate daily impact)
    events_daily = {}
    for _, row in events_df.iterrows():
        date = row["date"]
        if date not in events_daily:
            events_daily[date] = {"max_impact": 0, "total_attendance": 0, "event_count": 0,
                                   "event_hours": set()}
        impact_score = {"Low": 1, "Medium": 2, "High": 3}[row["impact_level"]]
        events_daily[date]["max_impact"] = max(events_daily[date]["max_impact"], impact_score)
        events_daily[date]["total_attendance"] += row["expected_attendance"]
        events_daily[date]["event_count"] += 1
        start_h = int(row["start_time"].split(":")[0])
        end_h = int(row["end_time"].split(":")[0])
        for h in range(start_h, end_h + 1):
            events_daily[date]["event_hours"].add(h)

    for date in dates:
        date_str = date.strftime("%Y-%m-%d")
        day_of_week = date.strftime("%A")
        is_weekend = 1 if date.dayofweek >= 5 else 0

        # Get weather for this date
        weather = weather_lookup.get(date_str, {})
        weather_severity = weather.get("weather_severity", 1) if isinstance(weather, dict) else 1
        rainfall = weather.get("rainfall_mm", 0) if isinstance(weather, dict) else 0
        visibility = weather.get("visibility_km", 10) if isinstance(weather, dict) else 10

        # Get events for this date
        event_info = events_daily.get(date_str, {"max_impact": 0, "total_attendance": 0,
                                                   "event_count": 0, "event_hours": set()})

        # Generate records for each route at multiple departure times
        # Not every route runs every hour — sample a subset
        for route_id, route_info in ROUTES.items():
            # Each route has 6-12 departures per day
            num_departures = np.random.randint(6, 13)
            departure_hours = sorted(np.random.choice(DEPARTURE_HOURS, size=num_departures, replace=False))

            for hour in departure_hours:
                minute = np.random.choice([0, 15, 30, 45])
                scheduled_time = f"{hour:02d}:{minute:02d}"

                # ─── Calculate delay with realistic correlations ─────
                base_delay = np.random.exponential(2)  # Base: most trips are ~on time

                # Rush hour effect (+3-8 min)
                is_rush = 1 if (7 <= hour <= 9) or (17 <= hour <= 19) else 0
                if is_rush:
                    base_delay += np.random.uniform(3, 8)

                # Weekend reduction (-2 min)
                if is_weekend:
                    base_delay -= 2

                # Weather effect
                if weather_severity >= 4:  # Storm / Heavy Rain
                    base_delay += np.random.uniform(8, 20)
                elif weather_severity == 3:  # Rain / Fog
                    base_delay += np.random.uniform(3, 10)
                elif weather_severity == 2:  # Light Rain
                    base_delay += np.random.uniform(1, 4)

                # Low visibility
                if visibility < 2:
                    base_delay += np.random.uniform(5, 12)
                elif visibility < 5:
                    base_delay += np.random.uniform(1, 5)

                # Event effect (stronger during event hours)
                if hour in event_info["event_hours"]:
                    impact_multiplier = event_info["max_impact"]  # 1, 2, or 3
                    base_delay += np.random.uniform(2, 6) * impact_multiplier
                elif event_info["event_count"] > 0:
                    # Some spillover even outside event hours
                    base_delay += np.random.uniform(0, 2)

                # Transport type reliability
                type_factor = {"Metro": 0.6, "Train": 0.8, "Bus": 1.0}
                base_delay *= type_factor[route_info["type"]]

                # Add some randomness & ensure non-negative
                delay = max(0, base_delay + np.random.normal(0, 1.5))
                delay = round(delay, 1)

                # Sometimes early (-ve delay), ~10% chance
                if np.random.random() < 0.10 and delay < 3:
                    delay = round(-np.random.uniform(0.5, 3), 1)

                # Passenger count (higher during rush, events)
                pax_base = np.random.randint(20, 150)
                if is_rush:
                    pax_base = int(pax_base * 1.5)
                if hour in event_info["event_hours"]:
                    pax_base = int(pax_base * 1.3)
                if is_weekend:
                    pax_base = int(pax_base * 0.7)

                # Delay categorization
                if delay <= 2:
                    delay_cat = "None"
                    is_delayed = 0
                elif delay <= 10:
                    delay_cat = "Minor"
                    is_delayed = 1
                elif delay <= 25:
                    delay_cat = "Moderate"
                    is_delayed = 1
                else:
                    delay_cat = "Major"
                    is_delayed = 1

                # Actual departure
                sched_dt = datetime.combine(date.date(), datetime.strptime(scheduled_time, "%H:%M").time())
                actual_dt = sched_dt + timedelta(minutes=delay)

                records.append({
                    "date": date_str,
                    "route_id": route_id,
                    "route_name": route_info["name"],
                    "transport_type": route_info["type"],
                    "scheduled_departure": sched_dt.strftime("%Y-%m-%d %H:%M"),
                    "actual_departure": actual_dt.strftime("%Y-%m-%d %H:%M"),
                    "delay_minutes": delay,
                    "is_delayed": is_delayed,
                    "delay_category": delay_cat,
                    "passenger_count": pax_base,
                    "day_of_week": day_of_week,
                    "is_weekend": is_weekend,
                    "is_rush_hour": is_rush,
                    "hour": hour
                })

    df = pd.DataFrame(records)
    filepath = os.path.join(RAW_DIR, "transport_delays.csv")
    df.to_csv(filepath, index=False)
    print(f"   ✅ Transport delays saved: {len(df)} records → {filepath}")
    return df


def load_raw_data():
    """Load all raw datasets from the data/raw directory."""
    transport = pd.read_csv(os.path.join(RAW_DIR, "transport_delays.csv"))
    weather = pd.read_csv(os.path.join(RAW_DIR, "weather_data.csv"))
    events = pd.read_csv(os.path.join(RAW_DIR, "events_data.csv"))
    print(f"📂 Loaded: Transport={len(transport)}, Weather={len(weather)}, Events={len(events)}")
    return transport, weather, events


def load_processed_data():
    """Load the merged/processed dataset."""
    filepath = os.path.join(PROCESSED_DIR, "merged_dataset.csv")
    df = pd.read_csv(filepath)
    print(f"📂 Loaded merged dataset: {len(df)} records")
    return df


# ─── Main: Generate all data ────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("🚀 PUBLIC TRANSPORT DELAY — DATA GENERATION")
    print("=" * 60)
    print(f"📅 Date range: {START_DATE.date()} to {END_DATE.date()}")
    print(f"🛤️  Routes: {len(ROUTES)}")
    print()

    # Ensure directories exist
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Generate in order (transport depends on weather & events)
    weather_df = generate_weather_data()
    events_df = generate_events_data()
    transport_df = generate_transport_delays(weather_df, events_df)

    print()
    print("=" * 60)
    print("✅ ALL DATA GENERATED SUCCESSFULLY!")
    print("=" * 60)
    print(f"\n📊 Summary:")
    print(f"   Weather records:   {len(weather_df):>8,}")
    print(f"   Event records:     {len(events_df):>8,}")
    print(f"   Transport records: {len(transport_df):>8,}")
    print(f"\n📁 Files saved to: {RAW_DIR}")
