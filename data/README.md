# 📊 Data Dictionary

## Overview
This project uses three primary datasets that are merged for analysis.
All datasets are synthetically generated to mimic realistic patterns.

---

## 1. Transport Delays (`transport_delays.csv`)

| Column | Type | Description |
|---|---|---|
| `date` | datetime | Date of the transport record |
| `time` | time | Scheduled departure time |
| `route_id` | string | Unique route identifier (e.g., R001–R020) |
| `route_name` | string | Human-readable route name |
| `transport_type` | string | Type: Bus, Metro, Train |
| `scheduled_departure` | datetime | Planned departure time |
| `actual_departure` | datetime | Actual departure time |
| `delay_minutes` | float | Delay in minutes (0 = on time, negative = early) |
| `is_delayed` | binary | 1 if delay > 5 minutes, else 0 |
| `delay_category` | string | None / Minor (5-15 min) / Moderate (15-30 min) / Major (30+ min) |
| `passenger_count` | int | Estimated passenger count |
| `day_of_week` | string | Monday through Sunday |
| `is_weekend` | binary | 1 if Saturday/Sunday |
| `is_rush_hour` | binary | 1 if 7-9 AM or 5-7 PM |

## 2. Weather Data (`weather_data.csv`)

| Column | Type | Description |
|---|---|---|
| `date` | datetime | Date of weather record |
| `temperature_c` | float | Temperature in Celsius |
| `humidity_pct` | float | Humidity percentage (0-100) |
| `rainfall_mm` | float | Rainfall in millimeters |
| `wind_speed_kmh` | float | Wind speed in km/h |
| `visibility_km` | float | Visibility in kilometers |
| `weather_condition` | string | Clear, Cloudy, Rain, Heavy Rain, Fog, Storm, Snow |
| `weather_severity` | int | 1 (mild) to 5 (severe) |

## 3. Events Data (`events_data.csv`)

| Column | Type | Description |
|---|---|---|
| `date` | datetime | Date of the event |
| `event_name` | string | Name of the event |
| `event_type` | string | Concert, Sports, Festival, Protest, Parade, Conference |
| `expected_attendance` | int | Estimated number of attendees |
| `impact_level` | string | Low / Medium / High |
| `venue` | string | Event venue location |
| `start_time` | time | Event start time |
| `end_time` | time | Event end time |

## 4. Merged Dataset (`merged_dataset.csv`)
Combined dataset with all transport, weather, and event features merged on `date`.

---

## Data Generation
All data is synthetically generated using Python scripts with realistic correlations:
- Higher rainfall → more delays
- Lower visibility → more delays  
- Large events → more delays during event hours
- Rush hours → more delays
- Weekends → fewer delays
