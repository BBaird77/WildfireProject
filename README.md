# Wildfire and Air Quality in Idaho Project

This project combines environmental data retrieval, data wrangling, forecasting, and dashboard development to analyze wildfire smoke and PM2.5 risk in Idaho.

# Tools Used
- Python
- pandas and NumPy
- scikit-learn
- Plotly
- Streamlit
- Geospatial data workflows
- External environmental APIs

# Project Workflow
1. Retrieve air quality, wildfire, and weather data
2. Clean and combine datasets into a modeling-ready table
3. Build forecasting features and train predictive models
4. Publish results in an interactive dashboard

# Goal
This project demonstrates my experience building an end-to-end data science workflow: collecting raw data, preparing it for analysis, modeling trends, and presenting findings in a clear, interactive format.

# Additional Info
This requires two API keys, being AQS and PurpleAir.
Here's the environmental variables that need to be set.

$env:AQS_EMAIL=""
$env:AQS_KEY="" 
$env:PURPLEAIR_API_KEY=""

The run order is:
1. AQS_Retrieval_Code.py
2. PurpleAir_Retrieval_Code.py
3. Open_Meteo_Retrieval_Code.py
4. NIFC_Retrieval_Code.py
5. Data_Manipulation.py
6. Forecast_Modeling.py
7. Streamlit.app

The NIFC raw data file is a .geojson. It's very large, and makes the file run very slowly. Give it some time, it does pull the raw data eventually.
