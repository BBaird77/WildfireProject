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
