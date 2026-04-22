# Idaho Wildfire and Air Quality Risk Forecast

This project explores short-term air quality risk in Idaho by combining PM2.5 air quality data, wildfire activity, and weather conditions into one end-to-end data science workflow. It includes data retrieval, cleaning, feature engineering, forecasting, and an interactive dashboard for exploring results.

## Project Goal

The goal of this project was to build a more useful short-term air quality forecast by combining environmental and wildfire-related signals instead of relying on air quality data alone. This project was designed to show the full workflow of a data science project, from collecting raw data to presenting results in a dashboard.

## Data Sources

- EPA AQS air quality data
- PurpleAir sensor data
- Open-Meteo weather data
- NIFC wildfire data

## Tools Used

- Python
- pandas and NumPy
- scikit-learn
- Streamlit
- Plotly
- Geospatial data workflows
- Public environmental APIs

## What This Project Includes

- Retrieval scripts for multiple public environmental data sources
- Data cleaning and merging for a modeling-ready dataset
- Feature engineering and forecasting for next-day PM2.5 risk
- A Streamlit dashboard for exploring forecasts and environmental conditions

## Main Workflow

1. Retrieve air quality, wildfire, and weather data
2. Clean and combine the datasets
3. Build forecasting features and train predictive models
4. Evaluate results
5. Display outputs in an interactive dashboard

## Repository Structure

- `AQS_Retrieval_Code.py` — pulls EPA AQS air quality data
- `PurpleAir_Retrieval_Code.py` — pulls PurpleAir sensor data
- `Open_Meteo_Retrieval_Code.py` — pulls weather data
- `NIFC_Retrieval_Code.py` — pulls wildfire data
- `Data_Manipulation.py` — combines and prepares the data
- `Forecast_Modeling.py` — builds and evaluates the forecasting model
- `Streamlit_app.py` — launches the dashboard

## Run Order

Run the scripts in this order:

1. `AQS_Retrieval_Code.py`
2. `PurpleAir_Retrieval_Code.py`
3. `Open_Meteo_Retrieval_Code.py`
4. `NIFC_Retrieval_Code.py`
5. `Data_Manipulation.py`
6. `Forecast_Modeling.py`
7. `Streamlit_app.py`

## Environment Variables

This project requires API credentials for AQS and PurpleAir.

Set the following environment variables in PowerShell before running the scripts:

```powershell
$env:AQS_EMAIL=""
$env:AQS_KEY=""
$env:PURPLEAIR_API_KEY=""
