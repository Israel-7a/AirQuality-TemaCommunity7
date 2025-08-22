import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio

# === Setup Paths ===
data_path = "openaq_location_4948390_measurments.csv"
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

# === Load Dataset ===
df = pd.read_csv(data_path)
df['datetimeLocal'] = pd.to_datetime(df['datetimeLocal'])
parameters = df['parameter'].unique()

# === Drop Unnecessary Columns ===
df = df.drop(['location_id', 'location_name', 'timezone', 'owner_name', 'provider', 
              'country_iso', 'isMobile', 'isMonitor'], axis=1, errors='ignore')

# === Pivot Data for Analysis ===
df_pivot = df.pivot(index='datetimeLocal', columns='parameter', values='value').reset_index()

# Estimate PM2.5 if missing
if 'pm25' not in df_pivot.columns and 'pm1' in df_pivot.columns:
    df_pivot['pm25'] = df_pivot['pm1'] / 0.7

# === Save Cleaned Data ===
df_pivot.to_csv(os.path.join(output_dir, 'updated_air_quality.csv'), index=False)

# === Create Plotly Interactive Figures ===
figs = []

# 1. Time-series plot for each parameter
for param in parameters:
    fig = px.line(df[df["parameter"] == param],
                  x="datetimeLocal", y="value",
                  title=f"{param} Over Time (Tema, Ghana)")
    figs.append(fig)

# 2. Hourly mean trends for each parameter
df['hour'] = df['datetimeLocal'].dt.hour
for param in parameters:
    hourly_means = df[df["parameter"] == param].groupby('hour')['value'].mean().reset_index()
    fig = px.line(hourly_means, x="hour", y="value", title=f"Hourly Mean Trend for {param}")
    figs.append(fig)

# 3. Distribution plots
for param in parameters:
    fig = px.histogram(df[df["parameter"] == param],
                       x="value", nbins=30,
                       title=f"Distribution of {param}", marginal="box")
    figs.append(fig)

# === Combine All Plotly Figures into index.html ===
html_content = ""
for fig in figs:
    html_content += pio.to_html(fig, full_html=False, include_plotlyjs='cdn')

with open("index.html", "w", encoding="utf-8") as f:
    f.write(html_content)

print("Interactive dashboard saved as index.html!")
