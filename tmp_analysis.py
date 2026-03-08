import pandas as pd
import numpy as np

transport = pd.read_csv('data/raw/transport_delays.csv')
weather = pd.read_csv('data/raw/weather_data.csv')
events = pd.read_csv('data/raw/events_data.csv')
merged = pd.read_csv('data/processed/merged_dataset.csv', low_memory=False)

print('='*65)
print('DATA SUFFICIENCY ANALYSIS')
print('='*65)

print('\n1. VOLUME')
print(f'   Transport records: {len(transport):,}')
print(f'   Weather records:   {len(weather):,}')
print(f'   Events records:    {len(events):,}')
print(f'   Merged records:    {len(merged):,}')

print('\n2. DATE COVERAGE')
merged['date'] = pd.to_datetime(merged['date'])
date_min = merged['date'].min().date()
date_max = merged['date'].max().date()
print(f'   Range: {date_min} to {date_max}')
print(f'   Total days covered: {merged["date"].nunique()}')
months = merged['date'].dt.to_period('M').nunique()
print(f'   Months covered: {months}')

print('\n3. TARGET VARIABLE')
print(f'   Delay mean:    {merged["delay_minutes"].mean():.1f} min')
print(f'   Delay median:  {merged["delay_minutes"].median():.1f} min')
print(f'   Delay std:     {merged["delay_minutes"].std():.1f} min')
delayed_count = (merged['is_delayed']==1).sum()
not_delayed = (merged['is_delayed']==0).sum()
print(f'   Delayed >15min:  {delayed_count:,} ({merged["is_delayed"].mean()*100:.1f}%)')
print(f'   On-time (<15m):  {not_delayed:,} ({(1-merged["is_delayed"].mean())*100:.1f}%)')

print('\n4. FEATURE COVERAGE')
print(f'   Unique boroughs: {merged["borough"].nunique()}')
print(f'   Unique routes:   {merged["route_id"].nunique()}')
print(f'   Weather types:   {merged["weather_condition"].nunique()}')
print(f'   Delay reasons:   {merged["delay_reason"].nunique()}')

print('\n5. CLASS IMBALANCE')
ratio = delayed_count / max(not_delayed, 1)
print(f'   Delayed:     {delayed_count:,} (89%)')
print(f'   Not delayed: {not_delayed:,} (11%)')
print(f'   Imbalance:   {ratio:.1f}:1')

print('\n6. WEATHER VARIETY')
print(merged['weather_condition'].value_counts().to_string())

print('\n7. EVENTS COVERAGE')
with_events = (merged['has_event']==1).sum()
no_events = (merged['has_event']==0).sum()
print(f'   Records on event days:    {with_events:,}')
print(f'   Records on non-event days: {no_events:,}')
print(f'   Event day ratio:          {merged["has_event"].mean()*100:.1f}%')

print('\n8. SEASONAL COVERAGE')
if 'season' not in merged.columns:
    merged['month'] = merged['date'].dt.month
    merged['season'] = merged['month'].map(lambda m: 'Winter' if m in [12,1,2] else 'Spring' if m in [3,4,5] else 'Summer' if m in [6,7,8] else 'Fall')
print(merged.get('season', merged['date'].dt.month).value_counts().to_string())

print('\n9. MODEL PERFORMANCE')
reg = pd.read_csv('data/processed/regression_results.csv')
cls = pd.read_csv('data/processed/classification_results.csv')
print(f'   Best R2 (regression):     {reg["R2_Score"].max():.4f}')
print(f'   Best F1 (classification): {cls["F1_Score"].max():.4f}')

print('\n' + '='*65)
print('VERDICT')
print('='*65)
