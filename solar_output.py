import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load historical solar and weather data
data = pd.read_csv("solar_mn_data.csv")
data['date'] = pd.to_datetime(data['date'])
data['month'] = data['date'].dt.month

# Only use certain months
months_to_use = [1, 2, 8, 9, 10, 11, 12]
month_names = {1: 'January', 2: 'February', 8: 'August',
               9: 'September', 10: 'October', 11: 'November', 12: 'December'}
data = data[data['month'].isin(months_to_use)]

# Features and target
features = ['irradiance', 'temp', 'cloud_cover', 'month']
target = 'output_kWh'
X = data[features]
y = data[target]

# Split and train
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Degradation + prediction
degradation_rate = 0.995
future_years = range(2025, 2046)

# Group by month and calculate average values for each feature
monthly_avg = data.groupby('month')[features].mean().reset_index(drop=True)

monthly_outputs = {}
average_outputs = {}

for year in future_years:
    future_data = monthly_avg.copy()
    prediction = model.predict(
        future_data[features]) * (degradation_rate ** (year - 2025))
    monthly_outputs[year] = np.round(prediction, 2)
    average_outputs[year] = round(np.mean(prediction), 2)

# Reorganize data for line plot
monthly_lines = {month: [] for month in months_to_use}
for year in future_years:
    predictions = monthly_outputs[year]
    for i, month in enumerate(months_to_use):
        monthly_lines[month].append(predictions[i])

# Plotting
fig, ax = plt.subplots(figsize=(10, 5))

# Line plot: one line per month
for month in months_to_use:
    ax.plot(future_years, monthly_lines[month],
            label=month_names[month], marker='o', linewidth=1.5)

# Labels and appearance
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Total Energy Output (kWh)', fontsize=12)
ax.set_title('Monthly Solar Output by Year (2025–2045)', fontsize=14)
ax.set_xticks(list(future_years))

# Legend
ax.legend(title='Month', fontsize='small',
          title_fontsize='small', loc='upper right')

# Table under plot
columns = [month_names[m] for m in months_to_use] + ['Yearly Avg']
table_data = [list(monthly_outputs[year]) + [average_outputs[year]]
              for year in future_years]

table = plt.table(cellText=table_data,
                  colLabels=columns,
                  rowLabels=future_years,
                  cellLoc='center',
                  rowLoc='center',
                  colColours=["#f1f1f1"] * len(columns),
                  bbox=[0.05, -0.75, 0.9, 0.6])

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.6)

plt.subplots_adjust(bottom=0.4)
plt.tight_layout()
plt.show()

plt.show()


fig, ax = plt.subplots(figsize=(10, 5))

# Calculate total output for each year (sum of all months)
yearly_totals = {year: sum(monthly_outputs[year]) for year in future_years}

years = list(yearly_totals.keys())
total_energy = list(yearly_totals.values())

bars = ax.bar(years, total_energy, color='mediumseagreen')

# Add energy values inside the bars
for bar, energy in zip(bars, total_energy):
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        height / 2,
        f'{int(energy)}',
        ha='center',
        va='center',
        color='white',
        fontsize=9,
        fontweight='bold'
    )

ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Total Energy Output (kWh)', fontsize=12)
ax.set_title(
    'Total Yearly Solar Energy Output (2025–2045) (Months 1-2, 8-12)', fontsize=14)
ax.set_xticks(years)
ax.set_xticklabels(years, rotation=45)

plt.tight_layout()
plt.show()
