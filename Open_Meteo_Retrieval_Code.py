### Open-Meteo Wind Data Retrieval Code

import datetime as dt
from pathlib import Path
import numpy as np
import pandas as pd
import requests

#Important Variables
start_date = dt.date(2021, 1, 1)
end_date = dt.date(2026, 3, 31)
OpenMeteoArchive = "https://archive-api.open-meteo.com/v1/archive"

Path("outputs/raw").mkdir(parents=True, exist_ok=True)
Path("outputs/data").mkdir(parents=True, exist_ok=True)
raw_weather = Path(f"outputs/raw/idaho_OMData_{start_date}_{end_date}.csv")
OMData_file = Path("outputs/data/OMData_handoff.csv")


# Four Decently large cities that cover most of Idaho. 
idaho_wind_keylocations = [
    {"name": "Boise", "latitude": 43.6150, "longitude": -116.2023}, #Southwest
    {"name": "Coeur d'Alene", "latitude": 47.6777, "longitude": -116.7805}, #North
    {"name": "Idaho Falls", "latitude": 43.4917, "longitude": -112.0333}, #Southeast
    {"name": "Twin Falls", "latitude": 42.5629, "longitude": -114.4609}, #South
]



# Retrieval Section
if raw_weather.exists():
    OMData = pd.read_csv(raw_weather)
    print("Using cached Open-Meteo file:", raw_weather)
else:
    point_daily_parts = []

    for point in idaho_wind_keylocations:
        params = {
            "latitude": point["latitude"],
            "longitude": point["longitude"],
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "hourly": "wind_speed_10m,wind_direction_10m",
            "wind_speed_unit": "kmh",
            "timezone": "America/Boise",
        }

        response = requests.get(OpenMeteoArchive, params=params, timeout=90)
        response.raise_for_status()

        weather = response.json().get("hourly", {})
        if not weather:
            continue

        point_hourly = pd.DataFrame({
            "time": pd.to_datetime(weather["time"]),
            "wind_from_deg": pd.to_numeric(weather["wind_direction_10m"], errors="coerce"),
            "wind_speed_kmh": pd.to_numeric(weather["wind_speed_10m"], errors="coerce"),
        })

        point_hourly["date"] = point_hourly["time"].dt.date
        point_hourly = point_hourly.dropna(subset=["date", "wind_from_deg", "wind_speed_kmh"])

        if point_hourly.empty:
            continue

        radians = np.radians(point_hourly["wind_from_deg"])
        point_hourly["x"] = np.cos(radians) * point_hourly["wind_speed_kmh"]
        point_hourly["y"] = np.sin(radians) * point_hourly["wind_speed_kmh"]

        point_daily = point_hourly.groupby("date", as_index=False).agg({
            "x": "mean",
            "y": "mean",
            "wind_speed_kmh": "mean"
        })

        point_daily["wind_from_deg"] = (
            np.degrees(np.arctan2(point_daily["y"], point_daily["x"])) + 360
        ) % 360

        point_daily_parts.append(point_daily[["date", "wind_from_deg", "wind_speed_kmh"]])

    if point_daily_parts:
        OMData = pd.concat(point_daily_parts, ignore_index=True)

        radians = np.radians(OMData["wind_from_deg"])
        OMData["x"] = np.cos(radians) * OMData["wind_speed_kmh"]
        OMData["y"] = np.sin(radians) * OMData["wind_speed_kmh"]

        OMData = OMData.groupby("date", as_index=False).agg({
            "x": "mean",
            "y": "mean",
            "wind_speed_kmh": "mean"
        })

        OMData["wind_from_deg"] = (
            np.degrees(np.arctan2(OMData["y"], OMData["x"])) + 360
        ) % 360

        OMData = OMData[["date", "wind_from_deg", "wind_speed_kmh"]]
        OMData.to_csv(raw_weather, index=False)
        print("Saved raw Open-Meteo file:", raw_weather)
    else:
        OMData = pd.DataFrame(columns=["date", "wind_from_deg", "wind_speed_kmh"])



# Cleaning Data
OMData["date"] = pd.to_datetime(OMData["date"]).dt.date
OMData["wind_from_deg"] = pd.to_numeric(OMData["wind_from_deg"])
OMData["wind_speed_kmh"] = pd.to_numeric(OMData["wind_speed_kmh"])

OMData = OMData.dropna(subset=["date", "wind_from_deg", "wind_speed_kmh"])
OMData = OMData[(OMData["date"] >= start_date) & (OMData["date"] <= end_date)]
OMData = OMData.sort_values("date")
OMData = OMData.drop_duplicates("date", keep="last")
OMData = OMData.reset_index(drop=True)

OMData.to_csv(OMData_file, index=False)
print("Saved Open-Meteo handoff:", OMData_file)