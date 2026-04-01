### AQS PM2.5 Data Retrieval for Idaho

import os
import time
import random
import datetime as dt
from pathlib import Path
import pandas as pd
import requests

#Important Variables
aqs_base = "https://aqs.epa.gov/data/api"
idaho_fips = "16"
pm25_param = "88101"
start_date = dt.date(2021, 1, 1)
end_date = dt.date(2026, 3, 31)
min_city_days = 120
aqs_email = os.getenv("AQS_EMAIL", "").strip()
aqs_key = os.getenv("AQS_KEY", "").strip()

Path(r"outputs/raw").mkdir(parents=True, exist_ok=True)
Path(r"outputs/data").mkdir(parents=True, exist_ok=True)
AQSDaily_cache = Path(r"outputs/raw/aqs_pm25_idaho_" + f"{start_date}_{end_date}.csv")
AQSmonitor_cache = Path(r"outputs/raw/aqs_pm25_idaho_monitors_" + f"{start_date.year}_{end_date.year}.csv")

if not aqs_email or not aqs_key:
    raise ValueError("You forgot to input your AQS_EMAIL or AQS_KEY, you dolt. Fix that and try it again.")



# Data Retrieval Loop
AQS_pulls = [
    {
        "name": "daily",
        "cache_file": AQSDaily_cache,
        "endpoint": "dailyData/byState",
        "date_mode": "bounded",
        "add_year": False,
        "fail_msg": "AQS daily data for {year} failed. Try restarting VSCode. Worked before.",
    },
    {
        "name": "monitors",
        "cache_file": AQSmonitor_cache,
        "endpoint": "monitors/byState",
        "date_mode": "full_year",
        "add_year": True,
        "fail_msg": "AQS monitor data for {year} failed. How did the first one work and this didn't?",
    },
]

loaded_data = {}
for dataset in AQS_pulls:
    cache_file = dataset["cache_file"]

    if cache_file.exists():
        loaded_data[dataset["name"]] = pd.read_csv(cache_file)
        print(f"Using cached AQS {dataset['name']} file:", cache_file)
        continue

    yearly_parts = []

    for year in range(start_date.year, end_date.year + 1):
        if dataset["date_mode"] == "bounded":
            year_start = max(start_date, dt.date(year, 1, 1))
            year_end = min(end_date, dt.date(year, 12, 31))
        else:
            year_start = dt.date(year, 1, 1)
            year_end = dt.date(year, 12, 31)

        params = {
            "email": aqs_email,
            "key": aqs_key,
            "param": pm25_param,
            "Sdate": year_start.strftime("%Y%m%d"),
            "Edate": year_end.strftime("%Y%m%d"),
            "state": idaho_fips,
        }

        result = None

        for attempt in range(5):
            try:
                response = requests.get(
                    f"{aqs_base}/{dataset['endpoint']}",
                    params=params,
                    timeout=90,
                )
                response.raise_for_status()
                result = response.json()
                break
            except Exception:
                wait_time = min((1.7 ** attempt) + random.random() * 0.2, 12)
                time.sleep(wait_time)

        if result is None:
            raise RuntimeError(dataset["fail_msg"].format(year=year))

        year_df = pd.DataFrame(result.get("Data", []))
        if not year_df.empty:
            if dataset["add_year"]:
                year_df["year"] = year
            yearly_parts.append(year_df)

    loaded_data[dataset["name"]] = (
        pd.concat(yearly_parts, ignore_index=True) if yearly_parts else pd.DataFrame()
    )
    loaded_data[dataset["name"]].to_csv(cache_file, index=False)
    print(f"Saved AQS {dataset['name']} cache:", cache_file)



AQSData_raw = loaded_data["daily"]
AQSmonitors_raw = loaded_data["monitors"]
if AQSData_raw.empty:
    raise RuntimeError("No data showed up in the main file.")
if AQSmonitors_raw.empty:
    raise RuntimeError("No sensors here. Need those for potentially identifying at some point.")



# Handoff Section
AQSFullData = AQSData_raw[["state_code", "county_code", "site_number", "date_local", "arithmetic_mean"]].copy()
AQSFullData["date"] = pd.to_datetime(AQSFullData["date_local"]).dt.date
AQSFullData["pm25"] = pd.to_numeric(AQSFullData["arithmetic_mean"])
AQSFullData = AQSFullData.dropna(subset=["date", "pm25"])
AQSFullData = AQSFullData[(AQSFullData["date"] >= start_date) & (AQSFullData["date"] <= end_date)]



# Making an id to combine these.
frames = {
    "AQSFullData": AQSFullData,
    "monitors": AQSmonitors_raw.copy(),
}
for name, df in frames.items():
    state = pd.to_numeric(df["state_code"], errors="coerce").astype("Int64").astype(str).str.zfill(2)
    county = pd.to_numeric(df["county_code"], errors="coerce").astype("Int64").astype(str).str.zfill(3)
    site = pd.to_numeric(df["site_number"], errors="coerce").astype("Int64").astype(str).str.zfill(4)

    df["site_id"] = state + "-" + county + "-" + site
frames["AQSFullData"] = (
    frames["AQSFullData"]
    .groupby(["site_id", "date"], as_index=False)["pm25"]
    .mean()
)
AQSFullData = frames["AQSFullData"]
monitors = frames["monitors"]



# Cleaning this up a bit here.
# monitors["latitude"] = pd.to_numeric(monitors["latitude"], errors="coerce")
# monitors["longitude"] = pd.to_numeric(monitors["longitude"], errors="coerce")
monitors["name"] = monitors["city_name"].fillna("AQS site").astype(str).str.strip()
monitors["city_name"] = monitors["city_name"].fillna("UNKNOWN").astype(str).str.strip()
monitors.loc[monitors["name"] == "", "name"] = "AQS site"
monitors.loc[monitors["city_name"] == "", "city_name"] = "UNKNOWN"
monitors = (
    monitors.sort_values("year")
    .dropna(subset=["latitude", "longitude"])
    .drop_duplicates("site_id", keep="last")
    [["site_id", "latitude", "longitude", "name", "city_name"]]
)



# Cleaning Data
AQSFullData = AQSFullData.merge(monitors, on="site_id", how="left")
AQSFullData = AQSFullData.dropna(subset=["latitude", "longitude"])
AQSFullData["sensor_key"] = "AQS|" + AQSFullData["site_id"].astype(str)
AQSFullData["source"] = "AQS"
AQSFullData = AQSFullData[["sensor_key", "site_id", "name", "city_name", "source", "latitude", "longitude", "date", "pm25"]]
AQSFullData = AQSFullData.sort_values(["sensor_key", "date"]).reset_index(drop=True)
AQSFullData_file = Path(r"outputs/data/AQSFullData_handoff.csv")
AQSFullData.to_csv(AQSFullData_file, index=False)

# Wanted these earlier. Useful for identifying, but not really needed anymore.
#print("Unique AQS site Idataset:", AQSFullData["site_id"].nunique())
#print("Top monitor names:", AQSFullData["name"].value_counts().head(15).to_dict())
print("Saved AQS handoff:", AQSFullData_file)



# Keeping this city file too. Originally was planning on trying to predict by city. Something to work on later.
AQSCity_Data = (
    AQSFullData.groupby(["city_name", "date"], as_index=False)
    .agg(pm25=("pm25", "mean"), n_rows=("pm25", "size"))
    .sort_values(["city_name", "date"])
)

coverage = AQSCity_Data.groupby("city_name")["date"].nunique()
keep_cities = coverage[coverage >= min_city_days].index
AQSCity_Data = AQSCity_Data[AQSCity_Data["city_name"].isin(keep_cities)]
AQSCity_Data_file = Path(r"outputs/data/AQSCity_Data_pm25.csv")
AQSCity_Data.to_csv(AQSCity_Data_file, index=False)

#print("Saved city-level file just in case:", AQSCity_Data_file)
print("AQS Data Retrieval: Completed")
