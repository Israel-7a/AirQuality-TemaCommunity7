# extract_visuals.py
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from matplotlib.dates import AutoDateLocator, DateFormatter

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

# === 1. Time-Series Plots ===
fig, axes = plt.subplots(len(parameters), 1, figsize=(12, 4 * len(parameters)), sharex=True)
for i, param in enumerate(parameters):
    param_df = df[df['parameter'] == param]
    axes[i].plot(param_df['datetimeLocal'], param_df['value'], label=param, color='blue', alpha=0.7)
    axes[i].set_title(f'{param} Over Time (Tema, Ghana)')
    axes[i].set_ylabel(f'{param} ({param_df["unit"].iloc[0]})')
    axes[i].legend()
    axes[i].grid(True)
axes[-1].set_xlabel('Date')
for ax in axes:
    ax.xaxis.set_major_locator(AutoDateLocator())
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
    ax.tick_params(axis='x', rotation=45, labelsize=10, labelright=True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'time_series_separate_plots.png'), dpi=300)
plt.close()

# === 2. Hourly Boxplots ===
df['hour'] = df['datetimeLocal'].dt.hour
fig, axes = plt.subplots(len(parameters), 1, figsize=(12, 4 * len(parameters)), sharex=True)
for i, param in enumerate(parameters):
    param_df = df[df['parameter'] == param]
    sns.boxplot(x='hour', y='value', data=param_df, ax=axes[i])
    axes[i].set_title(f'Hourly Trends for {param}')
    axes[i].set_ylabel(f'{param} ({param_df["unit"].iloc[0]})')
    axes[i].grid(True)
axes[-1].set_xlabel('Hour of Day')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'hourly_trends_boxplots.png'), dpi=300)
plt.close()

# === 3. Hourly Mean Line Plots ===
fig, axes = plt.subplots(len(parameters), 1, figsize=(12, 4 * len(parameters)), sharex=True)
for i, param in enumerate(parameters):
    param_df = df[df['parameter'] == param]
    hourly_means = param_df.groupby('hour')['value'].mean()
    axes[i].plot(hourly_means.index, hourly_means, marker='o', color='blue')
    axes[i].set_title(f'Hourly Mean Trends for {param}')
    axes[i].set_ylabel(f'{param} ({param_df["unit"].iloc[0]})')
    axes[i].grid(True)
axes[-1].set_xlabel('Hour of Day')
axes[-1].set_xticks(range(24))
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'hourly_mean_trends.png'), dpi=300)
plt.close()

# === 4. Distribution Plots ===
for param in parameters:
    plt.figure(figsize=(8, 5))
    sns.histplot(df[df['parameter'] == param]['value'], kde=True)
    plt.title(f'Distribution of {param}')
    plt.xlabel(param)
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(output_dir, f'distribution_{param}.png'))
    plt.close()

# === 5. Correlation Pairplot (if PM2.5 exists) ===
if 'pm25' in df_pivot.columns and 'pm1' in df_pivot.columns and 'um003' in df_pivot.columns:
    sns.pairplot(df_pivot[['pm25', 'pm1', 'um003']], diag_kind='kde')
    plt.suptitle('Pairplot of PM2.5, PM1, and UM003', y=1.02)
    plt.savefig(os.path.join(output_dir, 'pairplot.png'))
    plt.close()

print(f"All visuals saved in '{output_dir}' folder!")
