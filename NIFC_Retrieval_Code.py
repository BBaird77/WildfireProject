### NIFC Wildfire Data Retrieval Code

import datetime as dt
import json
import urllib.request
from pathlib import Path
import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from shapely import make_valid


# Important variables
start_date = dt.date(2021, 1, 1)
end_date = dt.date(2026, 3, 31)
USCensus_States = "https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_state_20m.zip"
NIFC_url = (
    "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/"
    "services/WFIGS_Interagency_Perimeters/FeatureServer/0/query"
)

# These seem like the most relevant "western" states andthey seem the most likely to affect Idaho just based on distance.
WesternStates = ["WA", "OR", "CA", "ID", "MT", "WY", "NV", "UT", "CO", "AZ", "NM"]

Path("outputs/raw").mkdir(parents=True, exist_ok=True)
Path("outputs/raw/.cache").mkdir(parents=True, exist_ok=True)
Path("outputs/data").mkdir(parents=True, exist_ok=True)


# Read state boundaries
States_zip = Path("outputs/raw/.cache/cb_2023_us_state_20m.zip")
if not States_zip.exists():
    urllib.request.urlretrieve(USCensus_States, States_zip)

states = gpd.read_file(f"zip://{States_zip}").to_crs("EPSG:4326")
west_polygon = states.loc[states["STUSPS"].isin(WesternStates), "geometry"].union_all()

NIFCData_raw = Path(f"outputs/raw/NIFCWildfire_perimeters_{start_date}_{end_date}.geojson")
NIFCData_file = Path("outputs/data/NIFCData_handoff.csv")


# NIFC Data Retrieval
if NIFCData_raw.exists():
    with open(NIFCData_raw, "r", encoding="utf-8") as file:
        raw_geojson = json.load(file)
    print("Using cached NIFC pull:", NIFCData_raw)
else:
    next_day = end_date + dt.timedelta(days=1)
    bounds = west_polygon.bounds

    where = (
        "attr_IncidentTypeCategory = 'WF' AND "
        "("
        f"poly_DateCurrent >= DATE '{start_date:%Y-%m-%d}' AND poly_DateCurrent < DATE '{next_day:%Y-%m-%d}' "
        "OR "
        f"poly_PolygonDateTime >= DATE '{start_date:%Y-%m-%d}' AND poly_PolygonDateTime < DATE '{next_day:%Y-%m-%d}'"
        ")"
    )

    all_features = []
    offset = 0

    while True:
        response = requests.get(
            NIFC_url,
            params={
                "where": where,
                "outFields": "*",
                "returnGeometry": "true",
                "outSR": 4326,
                "f": "geojson",
                "resultRecordCount": 2000,
                "resultOffset": offset,
                "geometry": f"{bounds[0]},{bounds[1]},{bounds[2]},{bounds[3]}",
                "geometryType": "esriGeometryEnvelope",
                "inSR": 4326,
                "spatialRel": "esriSpatialRelIntersects",
            },
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=90,
        )
        response.raise_for_status()
        page = response.json()
        features = page.get("features", [])

        if not features:
            break
        all_features.extend(features)

        if len(features) < 2000:
            break
        offset += 2000

    raw_geojson = {"type": "FeatureCollection", "features": all_features}

    with open(NIFCData_raw, "w", encoding="utf-8") as file:
        json.dump(raw_geojson, file)
    print("Saved raw NIFC pull:", NIFCData_raw)

NIFCRaw_features = raw_geojson.get("features", [])
#print("Raw features returned:", len(NIFCRaw_features))


Wildfires = gpd.GeoDataFrame.from_features(NIFCRaw_features, crs="EPSG:4326")

Wildfire_date = pd.Series(pd.NaT, index=Wildfires.index)
for col in ["poly_DateCurrent", "poly_PolygonDateTime", "attr_WildfireDiscoveryDateTime"]:
    if col in Wildfires.columns:
        if pd.api.types.is_numeric_dtype(Wildfires[col]):
            parsed = pd.to_datetime(Wildfires[col], unit="ms", errors="coerce")
        else:
            parsed = pd.to_datetime(Wildfires[col], errors="coerce")
            parsed_ms = pd.to_datetime(
                pd.to_numeric(Wildfires[col], errors="coerce"),
                unit="ms",
                errors="coerce",
            )
            parsed = parsed.fillna(parsed_ms)
        Wildfire_date = Wildfire_date.fillna(parsed)
Wildfires["date"] = Wildfire_date.dt.date


Wildfires["Wildfire_id"] = Wildfires.index.astype(str)
if "poly_IRWINID" in Wildfires.columns:
    Wildfires["Wildfire_id"] = Wildfires["poly_IRWINID"].astype(str)
elif "attr_IrwinID" in Wildfires.columns:
    Wildfires["Wildfire_id"] = Wildfires["attr_IrwinID"].astype(str)
elif "poly_IncidentName" in Wildfires.columns:
    Wildfires["Wildfire_id"] = Wildfires["poly_IncidentName"].astype(str)
elif "attr_IncidentName" in Wildfires.columns:
    Wildfires["Wildfire_id"] = Wildfires["attr_IncidentName"].astype(str)


Wildfires["Wildfire_acres"] = np.nan
for col in ["poly_GISAcres", "poly_Acres_AutoCalc", "attr_IncidentSize", "attr_FinalAcres"]:
    if col in Wildfires.columns:
        Wildfires["Wildfire_acres"] = Wildfires["Wildfire_acres"].fillna(
            pd.to_numeric(Wildfires[col], errors="coerce")
        )

Wildfires = Wildfires.dropna(subset=["date", "geometry"])
#print("Rows after dropping missing date/geometry:", len(Wildfires))
Wildfires = Wildfires[(Wildfires["date"] >= start_date) & (Wildfires["date"] <= end_date)]
#print("Rows after date filter:", len(Wildfires))
Wildfires = Wildfires[Wildfires.geometry.notna() & ~Wildfires.geometry.is_empty]
#print("Rows after dropping empty geometry:", len(Wildfires))
Wildfires["geometry"] = Wildfires["geometry"].apply(make_valid)
Wildfires = Wildfires[Wildfires.geometry.notna() & ~Wildfires.geometry.is_empty]
#print("Rows after make_valid:", len(Wildfires))
rep_points = Wildfires.geometry.representative_point()
Wildfires = Wildfires.loc[rep_points.within(west_polygon)]
#print("Rows after western states filter:", len(Wildfires))


rep_points = Wildfires.geometry.representative_point()
Wildfires["Wildfire_lon"] = rep_points.x
Wildfires["Wildfire_lat"] = rep_points.y
Wildfires["Wildfire_acres"] = pd.to_numeric(Wildfires["Wildfire_acres"], errors="coerce").fillna(0)

# Cleaning Data
NIFCData_handoff = Wildfires[["date", "Wildfire_id", "Wildfire_acres", "Wildfire_lat", "Wildfire_lon"]]
NIFCData_handoff = NIFCData_handoff.dropna(subset=["date", "Wildfire_id", "Wildfire_lat", "Wildfire_lon"])
NIFCData_handoff = NIFCData_handoff.sort_values(["date", "Wildfire_id"])
NIFCData_handoff = NIFCData_handoff.drop_duplicates(subset=["Wildfire_id", "date"], keep="last")
NIFCData_handoff = NIFCData_handoff.reset_index(drop=True)

#print("Final handoff rows:", len(NIFCData_handoff))
NIFCData_handoff.to_csv(NIFCData_file, index=False)
print("Saved NIFC handoff:", NIFCData_file)
print("NIFC Wildfire Data finished. Arcgis Data in python is a pain. Hopefully it really helps the model.")