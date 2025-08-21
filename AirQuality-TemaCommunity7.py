# tema_air_quality.py
# Integrated Air Quality Analysis & RandomForest Model for Tema Community

import os
import json
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from matplotlib.dates import AutoDateLocator, DateFormatter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ------------------------
# Config / output folder
# ------------------------
CSV = "openaq_location_4948390_measurments.csv"
OUT = "outputs"
os.makedirs(OUT, exist_ok=True)

# ------------------------
# 1) Load data
# ------------------------
try:
    df = pd.read_csv(CSV)
except FileNotFoundError:
    raise SystemExit(f"File not found: {CSV} — put your CSV in the same folder or change the CSV path.")

print("Initial rows, cols:", df.shape)
print("Columns:", df.columns.tolist())

# ------------------------
# 2) Normalize datetime & minimal cleaning
# ------------------------
if "datetimeLocal" not in df.columns:
    raise SystemExit("Expected column 'datetimeLocal' not found.")

df["datetimeLocal"] = pd.to_datetime(df["datetimeLocal"], errors="coerce")
df = df.dropna(subset=["datetimeLocal"])

drop_cols = ["location_id", "location_name", "timezone", "owner_name", "provider",
             "country_iso", "isMobile", "isMonitor"]
df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

# ------------------------
# 3) Pivot long -> wide
# ------------------------
df_pivot = df.pivot(index="datetimeLocal", columns="parameter", values="value").sort_index().reset_index()
print("Pivoted shape:", df_pivot.shape)

if "pm25" not in df_pivot.columns and "pm1" in df_pivot.columns:
    print("pm25 missing — estimating pm25 = pm1 / 0.7")
    df_pivot["pm25"] = df_pivot["pm1"] / 0.7

if "pm25" not in df_pivot.columns:
    raise SystemExit("pm25 not found and cannot be estimated. Need pm25 or pm1 in the data.")

# ------------------------
# 4) Add weather columns (mock if missing)
# ------------------------
if "temperature" not in df_pivot.columns:
    df_pivot["temperature"] = np.random.uniform(20, 30, size=len(df_pivot))
if "relativehumidity" not in df_pivot.columns:
    df_pivot["relativehumidity"] = np.random.uniform(60, 95, size=len(df_pivot))

# Clip extreme pm25 to 99th percentile
p99 = df_pivot["pm25"].quantile(0.99)
df_pivot["pm25"] = df_pivot["pm25"].clip(upper=p99)

# Save cleaned pivoted CSV
clean_csv = os.path.join(OUT, "cleaned_air_quality_wide.csv")
df_pivot.to_csv(clean_csv, index=False)
print("Saved cleaned pivoted file:", clean_csv)

# ------------------------
# 5) EDA (static plots)
# ------------------------
# (a) Time series
plt.figure(figsize=(12,4))
plt.plot(df_pivot["datetimeLocal"], df_pivot["pm25"], lw=0.7)
plt.title("PM2.5 Over Time")
plt.xlabel("Date")
plt.ylabel("PM2.5 (µg/m³)")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "pm25_timeseries.png"), dpi=200)
plt.show()

# (b) Distribution
plt.figure(figsize=(8,4))
sns.histplot(df_pivot["pm25"], kde=True, bins=30)
plt.title("Distribution of PM2.5")
plt.xlabel("PM2.5 (µg/m³)")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "pm25_distribution.png"), dpi=200)
plt.show()

# (c) Hourly boxplot
df_pivot["hour"] = df_pivot["datetimeLocal"].dt.hour
plt.figure(figsize=(12,5))
sns.boxplot(x="hour", y="pm25", data=df_pivot)
plt.title("PM2.5 by Hour of Day")
plt.xlabel("Hour")
plt.ylabel("PM2.5 (µg/m³)")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "pm25_by_hour_boxplot.png"), dpi=200)
plt.show()

# (d) Day of week barplot
df_pivot["day_name"] = df_pivot["datetimeLocal"].dt.day_name()
order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
plt.figure(figsize=(10,5))
sns.barplot(x="day_name", y="pm25", data=df_pivot, ci="sd", order=order, palette="viridis")
plt.title("Average PM2.5 by Day of Week (with SD)")
plt.xlabel("Day of Week")
plt.ylabel("PM2.5 (µg/m³)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "pm25_by_dayofweek.png"), dpi=200)
plt.show()

# ------------------------
# 6) Interactive Plotly visualizations
# ------------------------
# (a) Time series
fig = px.line(df_pivot, x="datetimeLocal", y="pm25", title="Interactive PM2.5 Over Time",
              labels={"datetimeLocal":"Date", "pm25":"PM2.5 (µg/m³)"})
fig.update_layout(template="plotly_white", height=450)
html_ts = os.path.join(OUT, "interactive_pm25_timeseries.html")
fig.write_html(html_ts)
print("Saved interactive time series:", html_ts)
fig.show()

# (b) Hourly average
hourly = df_pivot.groupby("hour", as_index=False)["pm25"].mean()
fig2 = px.bar(hourly, x="hour", y="pm25", title="Average PM2.5 by Hour",
              labels={"pm25":"PM2.5 (µg/m³)", "hour":"Hour of Day"})
fig2.update_layout(template="plotly_white", height=400)
html_hour = os.path.join(OUT, "interactive_hourly_avg.html")
fig2.write_html(html_hour)
print("Saved interactive hourly plot:", html_hour)
fig2.show()

# ------------------------
# 7) Feature engineering for modeling
# ------------------------
df_model = df_pivot.copy().sort_values("datetimeLocal").reset_index(drop=True)
df_model["day_of_week"] = df_model["datetimeLocal"].dt.dayofweek
df_model["pm25_lag1"] = df_model["pm25"].shift(1)
df_model["pm25_lag2"] = df_model["pm25"].shift(2)
df_model["pm25_roll3"] = df_model["pm25"].rolling(window=3, min_periods=1).mean()
df_model["temp_humidity"] = df_model["temperature"] * df_model["relativehumidity"]
df_model = df_model.dropna().reset_index(drop=True)
print("Rows available for modeling:", len(df_model))

# ------------------------
# 8) Train/Test split
# ------------------------
candidate_features = ["pm1", "um003", "temperature", "relativehumidity", "hour",
                      "day_of_week", "temp_humidity", "pm25_lag1", "pm25_lag2", "pm25_roll3"]
features = [f for f in candidate_features if f in df_model.columns]
X = df_model[features]
y = df_model["pm25"]

split_idx = int(len(df_model) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
print("Training rows:", len(X_train), "Test rows:", len(X_test))

# ------------------------
# 9) Train RandomForest
# ------------------------
rf = RandomForestRegressor(n_estimators=200, max_depth=12, min_samples_split=4,
                           min_samples_leaf=2, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)
print(f"Test RMSE: {rmse:.3f} µg/m³")
print(f"Test R²: {r2:.3f}")

# Save model and metrics
joblib.dump(rf, os.path.join(OUT, "rf_baseline.pkl"))
metrics = {"rmse": float(rmse), "r2": float(r2), "n_train": len(X_train),
           "n_test": len(X_test), "features": features}
with open(os.path.join(OUT, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)
print("Saved model and metrics to outputs/")

# ------------------------
# 10) Save predictions + actual vs predicted plot
# ------------------------
pred_df = pd.DataFrame({
    "datetimeLocal": df_model["datetimeLocal"].iloc[split_idx:].values,
    "actual_pm25": y_test.values,
    "predicted_pm25": y_pred
})
pred_df.to_csv(os.path.join(OUT, "predictions.csv"), index=False)

plt.figure(figsize=(12,4))
plt.plot(pred_df["datetimeLocal"], pred_df["actual_pm25"], label="Actual", lw=1)
plt.plot(pred_df["datetimeLocal"], pred_df["predicted_pm25"], label="Predicted (RF)", lw=1)
plt.legend()
plt.title("Actual vs Predicted PM2.5 (Test Set)")
plt.xlabel("Date")
plt.ylabel("PM2.5 (µg/m³)")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "actual_vs_predicted.png"), dpi=200)
plt.show()

# ------------------------
# 11) Feature importance
# ------------------------
fi = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=True)
plt.figure(figsize=(8, max(3, 0.35*len(fi))))
fi.plot(kind="barh")
plt.title("RandomForest Feature Importances")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "feature_importance.png"), dpi=200)
plt.show()

print("\nDone. All outputs are in the 'outputs' folder.")
print(os.listdir(OUT))
