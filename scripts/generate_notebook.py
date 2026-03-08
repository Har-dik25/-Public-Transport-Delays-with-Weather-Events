import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

# 1. Introduction
nb.cells.append(nbf.v4.new_markdown_cell("""# 🚍 Exploratory Data Analysis (EDA) — Public Transport Delays

This notebook explores a large-scale real-world dataset combining:
1. **NYC MTA Bus Delays (150,000+ records)**
2. **Open-Meteo Historical Weather Data (731 days)**
3. **Nager Public Holidays & Local Events**

**Goal:** Understand the statistical relationship between weather severity, large-scale events, and public transport delays.
"""))

# 2. Setup
nb.cells.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set aesthetic style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# Load the merged dataset
df = pd.read_csv('../data/processed/merged_dataset.csv', parse_dates=['date'])
print(f"Dataset Shape: {df.shape}")
df.head()
"""))

# 3. Univariate Analysis
nb.cells.append(nbf.v4.new_markdown_cell("""## 1. Univariate Analysis: Delay Distribution
How are bus delays distributed? Are most delays minor or severe?
"""))
nb.cells.append(nbf.v4.new_code_cell("""# Plotting Delay Distribution (Capping at 120 minutes for visualization)
plt.figure(figsize=(12, 6))
sns.histplot(df['delay_minutes'], bins=50, kde=True, color='coral')
plt.title('Distribution of Delay Durations (Minutes)', fontsize=14)
plt.xlabel('Delay Minutes')
plt.ylabel('Frequency')
plt.xlim(0, 120) 
plt.show()

print(f"Median Delay: {df['delay_minutes'].median()} minutes")
print(f"Mean Delay: {df['delay_minutes'].mean():.2f} minutes")
"""))

# 4. Bivariate Analysis
nb.cells.append(nbf.v4.new_markdown_cell("""## 2. Bivariate Analysis: Weather Impact
Does severe weather actually cause longer delays? Let's verify this visually and statistically.
"""))
nb.cells.append(nbf.v4.new_code_cell("""# Delay by Weather Severity (1 to 5)
plt.figure(figsize=(10, 6))
sns.boxplot(x='weather_severity', y='delay_minutes', data=df, palette='viridis', showfliers=False)
plt.title('Transport Delay by Weather Severity Category', fontsize=14)
plt.xlabel('Weather Severity (1=Clear, 5=Severe Storm/Snow)')
plt.ylabel('Delay (Minutes) [Outliers Excluded]')
plt.show()
"""))

# 5. Statistical Testing (T-Test)
nb.cells.append(nbf.v4.new_markdown_cell("""### 🔬 Statistical Hypothesis Testing (T-Test)
**Hypothesis:** Transport delays on "Severe Weather" days (Severity >= 3) are significantly longer than delays on "Clear/Light" days (Severity < 3).
"""))
nb.cells.append(nbf.v4.new_code_cell("""severe_weather = df[df['weather_severity'] >= 3]['delay_minutes'].dropna()
normal_weather = df[df['weather_severity'] < 3]['delay_minutes'].dropna()

t_stat, p_val = stats.ttest_ind(severe_weather, normal_weather, equal_var=False)

print(f"Average delay (Severe Weather): {severe_weather.mean():.2f} min")
print(f"Average delay (Normal Weather): {normal_weather.mean():.2f} min")
print(f"T-Statistic: {t_stat:.4f}")
print(f"P-Value: {p_val:.4e}")

if p_val < 0.05:
    print("\\n✅ Conclusion: Reject the null hypothesis. Severe weather causes a statistically significant increase in delay duration.")
else:
    print("\\n❌ Conclusion: Fail to reject the null hypothesis. No significant difference found.")
"""))

# 6. Time Based Trends
nb.cells.append(nbf.v4.new_markdown_cell("""## 3. Time-Based Trends
When do delays peak during the week and day?
"""))
nb.cells.append(nbf.v4.new_code_cell("""# Extract Hour from Scheduled Time if available, else group by Day of Week
df['day_of_week_num'] = df['date'].dt.dayofweek
days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

daily_delays = df.groupby('day_of_week_num')['delay_minutes'].mean()

plt.figure(figsize=(10, 5))
sns.barplot(x=days, y=daily_delays.values, palette='magma')
plt.title('Average Delay by Day of the Week')
plt.ylabel('Average Delay (Minutes)')
plt.show()
"""))

# 7. Correlation Heatmap
nb.cells.append(nbf.v4.new_markdown_cell("""## 4. Feature Correlation Analysis
What features have the strongest correlation with delays?
"""))
nb.cells.append(nbf.v4.new_code_cell("""# Select key numeric features
cols = ['delay_minutes', 'temperature_max_c', 'precipitation_mm', 'snowfall_cm', 
        'wind_speed_max_kmh', 'weather_severity', 'expected_attendance']

corr = df[cols].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", vmin=-0.2, vmax=0.6)
plt.title('Correlation Matrix of Key Features', fontsize=14)
plt.show()
"""))

os.makedirs('../notebooks', exist_ok=True)
nbf.write(nb, '../notebooks/01_exploratory_data_analysis.ipynb')
print("✅ Successfully generated ../notebooks/01_exploratory_data_analysis.ipynb")
