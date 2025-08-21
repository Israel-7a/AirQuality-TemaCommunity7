# Air Quality Analysis in Tema Community 7, Ghana

## Project Overview
This project analyzes air quality measurements in Tema Community 7, Ghana, focusing primarily on PM2.5 levels along with other environmental parameters such as PM1, temperature, and relative humidity. The goal is to understand temporal trends, correlations, and build a predictive model for PM2.5 using Random Forest regression.

---

## Dataset
- Source: [OpenAQ Measurements for Tema](openaq_location_4948390_measurements.csv)
- Parameters included: `pm1`, `pm25`, `relativehumidity`, `temperature`, `um003`
- Data spans multiple timestamps with hourly granularity.

---

## Project Features
1. **Data Cleaning & Preprocessing**
   - Handling missing values and duplicates
   - Pivoting data from long to wide format
   - Clipping outliers (99th percentile for PM2.5)
   - Adding mock weather data where necessary

2. **Exploratory Data Analysis (EDA)**
   - Time series plots of PM2.5
   - Hourly and daily trends
   - Distribution plots
   - Interactive visualizations using Plotly

3. **Feature Engineering**
   - Lag features (`pm25_lag1`, `pm25_lag2`)
   - Rolling average (`pm25_roll3`)
   - Combined features (`temperature × relative humidity`)

4. **Modeling**
   - Random Forest Regressor for PM2.5 prediction
   - Time-aware train/test split (first 80% train, last 20% test)
   - Performance metrics:
     - RMSE: ~1.337 µg/m³
     - R²: 0.980

5. **Outputs**
   - Cleaned CSV files
   - Static plots (`png`) and interactive plots (`html`)
   - Model saved as `rf_baseline.pkl`
   - Predictions saved as `predictions.csv`
   - Feature importance visualization

---

## Folder Structure
# Air Quality Analysis in Tema, Ghana

## Project Overview
This project analyzes air quality measurements in Tema Comunity 7, Ghana, focusing primarily on PM2.5 levels along with other environmental parameters such as PM1, temperature, and relative humidity. The goal is to understand temporal trends, correlations, and build a predictive model for PM2.5 using Random Forest regression.

---

## Dataset
- Source: [OpenAQ Measurements for Tema](openaq_location_4948390_measurements.csv)
- Parameters included: `pm1`, `pm25`, `relativehumidity`, `temperature`, `um003`
- Data spans multiple timestamps with hourly granularity.

---

## Project Features
1. **Data Cleaning & Preprocessing**
   - Handling missing values and duplicates
   - Pivoting data from long to wide format
   - Clipping outliers (99th percentile for PM2.5)
   - Adding mock weather data where necessary

2. **Exploratory Data Analysis (EDA)**
   - Time series plots of PM2.5
   - Hourly and daily trends
   - Distribution plots
   - Interactive visualizations using Plotly

3. **Feature Engineering**
   - Lag features (`pm25_lag1`, `pm25_lag2`)
   - Rolling average (`pm25_roll3`)
   - Combined features (`temperature × relative humidity`)

4. **Modeling**
   - Random Forest Regressor for PM2.5 prediction
   - Time-aware train/test split (first 80% train, last 20% test)
   - Performance metrics:
     - RMSE: ~1.337 µg/m³
     - R²: 0.980

5. **Outputs**
   - Cleaned CSV files
   - Static plots (`png`) and interactive plots (`html`)
   - Model saved as `rf_baseline.pkl`
   - Predictions saved as `predictions.csv`
   - Feature importance visualization

---
## Folder Structure
AirQuality-TemaCommunity7/
├── AirQuality-TemaCommunity7.py # Main Python script
├── AirQuality-TemaCommunity7.ipnyb 
├── outputs/ # Generated files
│ ├── cleaned_air_quality_wide.csv
│ ├── pm25_timeseries.png
│ ├── interactive_pm25_timeseries.html
│ └── ... (other outputs)
├── README.md # Project overview
└── requirements.txt (optional) # Python dependencies


---

## How to Run
```bash
# Clone the repository
git clone <your-repo-url>
cd project-folder

# Ensure dependencies are installed
pip install -r requirements.txt  # Optional, if you have a requirements file

# Run the full pipeline
python AirQuality-TemaCommunity7.py

Key Observations

PM2.5 shows clear daily and hourly patterns in Tema Community 7.

Temperature and humidity contribute to PM2.5 variations.

Random Forest model predicts PM2.5 with high accuracy (R² ≈ 0.98).

Interactive plots provide an easy way to explore trends over time.

Technologies Used

Python 3.x

Pandas, NumPy, Seaborn, Matplotlib

Plotly (interactive visualization)

Scikit-learn (Random Forest regression)


---

