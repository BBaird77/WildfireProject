### PurpleAir Daily Data Retrieval for Idaho

import os
import time
import datetime as dt
import urllib.request
from io import StringIO
from pathlib import Path
import geopandas as gpd
import pandas as pd
import requests


# Important variables
USCensus_States = "https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_state_20m.zip"
start_date = dt.date(2021, 1, 1)
end_date = dt.date(2026, 3, 31)
max_age = 0
max_sensors = 0
sleep_seconds = 1.1

Path("outputs/raw").mkdir(parents=True, exist_ok=True)
Path("outputs/raw/purpleair_history_cache").mkdir(parents=True, exist_ok=True)
Path("outputs/raw/.cache").mkdir(parents=True, exist_ok=True)
Path("outputs/data").mkdir(parents=True, exist_ok=True)

api_key = os.getenv("PURPLEAIR_API_KEY", "").strip()
if not api_key:
    raise ValueError("You forgot PURPLEAIR_API_KEY. Add all the API Keys in at once so this doesn't happen.")


# Load Idaho boundary once so it can be reused later
Census_path = Path("outputs/raw/.cache/cb_2023_us_state_20m.zip")
if not Census_path.exists():
    print("Downloading the Census shapefile for faster use on the other files.")
    urllib.request.urlretrieve(USCensus_States, Census_path)

states = gpd.read_file(f"zip://{Census_path}")
idaho = states[states["STUSPS"] == "ID"].to_crs("EPSG:4326")
Idaho_Border = idaho.geometry.iloc[0]


# Get PurpleAir sensor list for Idaho
print("Requesting the PurpleAir sensor list for Idaho.")

PurpAir_Sensorlist = requests.get(
    "https://api.purpleair.com/v1/sensors",
    headers={"X-API-Key": api_key},
    params={ # Have to use a bounding box, because it doesn't support the polygon. Polygon's way more useful later.
        "location_type": 0,
        "max_age": max_age,
        "nwlat": 49.0,
        "nwlng": -117.3,
        "selat": 42.0,
        "selng": -111.0,
        "fields": "sensor_index,name,latitude,longitude,last_seen,confidence,position_rating",
    },
    timeout=60,
)
PurpAir_Sensorlist.raise_for_status()

Sensor_json = PurpAir_Sensorlist.json()
IdahoSensors = pd.DataFrame(Sensor_json["data"], columns=Sensor_json["fields"])

for col in ["sensor_index", "latitude", "longitude", "last_seen", "confidence", "position_rating"]:
    if col in IdahoSensors.columns:
        IdahoSensors[col] = pd.to_numeric(IdahoSensors[col], errors="coerce")

IdahoSensors = IdahoSensors.dropna(subset=["sensor_index", "latitude", "longitude"])

sensor_points = gpd.GeoSeries(
    gpd.points_from_xy(IdahoSensors["longitude"], IdahoSensors["latitude"]),
    crs="EPSG:4326",
)
IdahoSensors = IdahoSensors.loc[sensor_points.within(Idaho_Border)]

if "last_seen" in IdahoSensors.columns:
    IdahoSensors = IdahoSensors.sort_values("last_seen", ascending=False)

IdahoSensors = IdahoSensors.reset_index(drop=True)

if max_sensors > 0:
    IdahoSensors = IdahoSensors.head(max_sensors)

IdahoSensors_file = Path("outputs/raw/purpleair_idaho_sensors.csv")
IdahoSensors.to_csv(IdahoSensors_file, index=False)

print("Saved list of Idaho sensors:", IdahoSensors_file)
print("Sensors found:", len(IdahoSensors))


# Pull daily history in chunks so reruns can use cache
Additive_timeframes = []
end_exclusive = end_date + dt.timedelta(days=1)

for i, row in IdahoSensors.iterrows():
    sensor_index = int(row["sensor_index"])
    current = start_date

    while current < end_exclusive:
        next_date = min(current + dt.timedelta(days=730), end_exclusive)
        cache_file = Path("outputs/raw/purpleair_history_cache") / (
            f"sensor_{sensor_index}_{current}_{next_date - dt.timedelta(days=1)}.csv"
        )

        if cache_file.exists():
            history_chunk = pd.read_csv(cache_file)
        else:
            history_response = requests.get(
                f"https://api.purpleair.com/v1/sensors/{sensor_index}/history/csv",
                headers={"X-API-Key": api_key},
                params={
                    "start_timestamp": int(
                        dt.datetime(
                            current.year,
                            current.month,
                            current.day,
                            tzinfo=dt.timezone.utc
                        ).timestamp()
                    ),
                    "end_timestamp": int(
                        dt.datetime(
                            next_date.year,
                            next_date.month,
                            next_date.day,
                            tzinfo=dt.timezone.utc
                        ).timestamp()
                    ),
                    "average": 1440,
                    "fields": "pm2.5_atm_a,pm2.5_atm_b,humidity",
                },
                timeout=120,
            )
            history_response.raise_for_status()

            text = history_response.text.strip()
            history_chunk = pd.read_csv(StringIO(text)) if text else pd.DataFrame()
            history_chunk.to_csv(cache_file, index=False)
            time.sleep(sleep_seconds)

        if not history_chunk.empty:
            SensorDailyData = history_chunk

            if "time_stamp" in SensorDailyData.columns:
                SensorDailyData["time_stamp"] = pd.to_numeric(SensorDailyData["time_stamp"], errors="coerce")
                SensorDailyData["date_utc"] = pd.to_datetime(
                    SensorDailyData["time_stamp"],
                    unit="s",
                    utc=True
                ).dt.date
            else:
                SensorDailyData["date_utc"] = pd.NaT

            for col in ["pm2.5_atm_a", "pm2.5_atm_b", "pm2.5_atm", "humidity"]:
                if col in SensorDailyData.columns:
                    SensorDailyData[col] = pd.to_numeric(SensorDailyData[col], errors="coerce")

            # Average channel A and B when both exist
            if "pm2.5_atm_a" in SensorDailyData.columns and "pm2.5_atm_b" in SensorDailyData.columns:
                SensorDailyData["pm25_atm"] = SensorDailyData[["pm2.5_atm_a", "pm2.5_atm_b"]].mean(axis=1)
            elif "pm2.5_atm_a" in SensorDailyData.columns:
                SensorDailyData["pm25_atm"] = SensorDailyData["pm2.5_atm_a"]
            elif "pm2.5_atm_b" in SensorDailyData.columns:
                SensorDailyData["pm25_atm"] = SensorDailyData["pm2.5_atm_b"]
            elif "pm2.5_atm" in SensorDailyData.columns:
                SensorDailyData["pm25_atm"] = SensorDailyData["pm2.5_atm"]
            else:
                SensorDailyData["pm25_atm"] = pd.NA

            keep_cols = ["date_utc", "pm25_atm"]
            if "humidity" in SensorDailyData.columns:
                keep_cols.append("humidity")

            SensorDailyData = SensorDailyData[keep_cols].dropna(subset=["date_utc"])
            SensorDailyData["sensor_index"] = sensor_index
            SensorDailyData["name"] = str(row.get("name", "")).strip()
            SensorDailyData["latitude"] = float(row["latitude"])
            SensorDailyData["longitude"] = float(row["longitude"])

            Additive_timeframes.append(SensorDailyData)

        current = next_date

    if (i + 1) % 25 == 0 or (i + 1) == len(IdahoSensors):
        print(f"Processed {i + 1}/{len(IdahoSensors)} PurpleAir sensors so far.")

PurpleAir_raw = pd.concat(Additive_timeframes, ignore_index=True)
raw_data_file = Path(f"outputs/raw/purpleair_daily_{start_date}_{end_date}.csv")
PurpleAir_raw.to_csv(raw_data_file, index=False)

# Cleaning Data
PurpleAirFullData = PurpleAir_raw
PurpleAirFullData["date"] = pd.to_datetime(PurpleAirFullData["date_utc"]).dt.date
PurpleAirFullData["pm25"] = pd.to_numeric(PurpleAirFullData["pm25_atm"])
PurpleAirFullData["sensor_index"] = pd.to_numeric(PurpleAirFullData["sensor_index"])
PurpleAirFullData["latitude"] = pd.to_numeric(PurpleAirFullData["latitude"])
PurpleAirFullData["longitude"] = pd.to_numeric(PurpleAirFullData["longitude"])

PurpleAirFullData["name"] = PurpleAirFullData["name"].fillna("PurpleAir").astype(str).str.strip()
PurpleAirFullData.loc[PurpleAirFullData["name"] == "", "name"] = "PurpleAir"

PurpleAirFullData = PurpleAirFullData.dropna(
    subset=["date", "pm25", "sensor_index", "latitude", "longitude"]
)

purpleair_points = gpd.GeoSeries(
    gpd.points_from_xy(PurpleAirFullData["longitude"], PurpleAirFullData["latitude"]),
    crs="EPSG:4326",
)
PurpleAirFullData = PurpleAirFullData.loc[purpleair_points.within(Idaho_Border)]

PurpleAirFullData["sensor_key"] = "PA|" + PurpleAirFullData["sensor_index"].astype(int).astype(str)
PurpleAirFullData["source"] = "PurpleAir"

PurpleAirFullData = PurpleAirFullData.groupby(["sensor_key", "date"], as_index=False).agg(
    name=("name", "first"),
    source=("source", "first"),
    latitude=("latitude", "mean"),
    longitude=("longitude", "mean"),
    pm25=("pm25", "mean"),
)

PurpleAirFullData = PurpleAirFullData[
    ["sensor_key", "name", "source", "latitude", "longitude", "date", "pm25"]
]

PurpleAirFullData = PurpleAirFullData.sort_values(["sensor_key", "date"]).reset_index(drop=True)

handoff_file = Path("outputs/data/PurpleAirFullData_handoff.csv")
PurpleAirFullData.to_csv(handoff_file, index=False)

print("Saved PurpleAir daily history:", raw_data_file)
print("Saved PurpleAir handoff:", handoff_file)
print("PurpleAir Data: complete. Most irritating API is now done.")
print("Never touch this unless I need to expand the timeframe again.")

# NEVER DELETE THIS DATA. FRESH DATA PULLS COST MONEY.