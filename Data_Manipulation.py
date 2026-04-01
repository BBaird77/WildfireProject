### Combining and Wrangling Data

import datetime as dt
import urllib.request
from pathlib import Path
import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from shapely.geometry import Point
from sklearn.neighbors import BallTree

# Important variables
start_date = dt.date(2021, 1, 1)
end_date = dt.date(2026, 3, 31)
StandardRisk_pm25 = 35.0
grid_km = 20.0
k_neighbors = 10
idw_power = 2.0
n_bands = 18
USCensus_States = "https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_state_20m.zip"

Path("outputs/raw").mkdir(parents=True, exist_ok=True)
Path("outputs/raw/.cache").mkdir(parents=True, exist_ok=True)
Path("outputs/reports").mkdir(parents=True, exist_ok=True)


# Load Idaho boundary and file paths
States_zip = Path("outputs/raw/.cache/cb_2023_us_state_20m.zip")
if not States_zip.exists():
    urllib.request.urlretrieve(USCensus_States, States_zip)
states = gpd.read_file(f"zip://{States_zip}").to_crs("EPSG:4326")
Idaho_Border = states.loc[states["STUSPS"] == "ID", "geometry"].union_all()
Idaho_Centroid = Idaho_Border.centroid
AQSFullData_file = Path("outputs/data/AQSFullData_handoff.csv")
PurpleAirFullData_file = Path("outputs/data/PurpleAirFullData_handoff.csv")
OMData_file = Path("outputs/data/OMData_handoff.csv")
NIFCData_file = Path("outputs/data/NIFCData_handoff.csv")


# Load datasets
AQSData = pd.read_csv(AQSFullData_file)
AQSData["date"] = pd.to_datetime(AQSData["date"]).dt.date
PurpAirData = pd.read_csv(PurpleAirFullData_file)
PurpAirData["date"] = pd.to_datetime(PurpAirData["date"]).dt.date
wind_df = pd.read_csv(OMData_file)
wind_df["date"] = pd.to_datetime(wind_df["date"]).dt.date
WildfireData = pd.read_csv(NIFCData_file)
WildfireData["date"] = pd.to_datetime(WildfireData["date"]).dt.date


# Combine sensor data
sensor_frames = [frame for frame in [AQSData, PurpAirData] if not frame.empty]

if sensor_frames:
    SensorDailyData = pd.concat(sensor_frames, ignore_index=True)
else:
    SensorDailyData = pd.DataFrame(columns=["sensor_key", "latitude", "longitude", "date", "pm25"])
SensorDailyData = SensorDailyData[
    (SensorDailyData["date"] >= start_date) & (SensorDailyData["date"] <= end_date)
]


# One row per sensor site
sensor_sites = (
    SensorDailyData.groupby("sensor_key", as_index=False)
    .agg(latitude=("latitude", "mean"), longitude=("longitude", "mean"))
    .sort_values("sensor_key")
    .reset_index(drop=True)
)
sensor_order = sensor_sites["sensor_key"].tolist()
daily_wide = SensorDailyData.pivot_table(
    index="date",
    columns="sensor_key",
    values="pm25",
    aggfunc="mean"
)
daily_wide = daily_wide.reindex(columns=sensor_order)
all_dates = sorted(day for day in daily_wide.index if isinstance(day, dt.date))


# Build Idaho grid once and reuse it
minx, miny, maxx, maxy = Idaho_Border.bounds
lat_mid = (miny + maxy) / 2
step_lat = grid_km / 111.0
step_lon = grid_km / (111.0 * max(np.cos(np.radians(lat_mid)), 1e-6))
lats = np.arange(miny, maxy + step_lat, step_lat)
lons = np.arange(minx, maxx + step_lon, step_lon)
grid_points = []
for lat in lats:
    for lon in lons:
        if Idaho_Border.contains(Point(lon, lat)):
            grid_points.append((lat, lon))
idaho_grid = pd.DataFrame(grid_points, columns=["latitude", "longitude"])


# Neighbor lookup does not change by day
sensor_xy = np.radians(sensor_sites[["latitude", "longitude"]].to_numpy())
grid_xy = np.radians(idaho_grid[["latitude", "longitude"]].to_numpy())

neighbor_tree = BallTree(sensor_xy, metric="haversine")
neighbor_dist, neighbor_idx = neighbor_tree.query(grid_xy, k=min(k_neighbors, len(sensor_sites)))
dist_km = neighbor_dist * 6371.0


# Build daily PM2.5 surfaces and statewide metrics
pm25_surfaces = []
metric_rows = []

for day in all_dates:
    values = daily_wide.loc[day].to_numpy(dtype=float)
    neighbor_vals = values[neighbor_idx]

    weights = 1.0 / (np.power(dist_km, idw_power) + 1e-6)

    bad = np.isnan(neighbor_vals)
    weights = np.where(bad, 0.0, weights)
    neighbor_vals = np.where(bad, 0.0, neighbor_vals)

    weight_totals = weights.sum(axis=1)
    pm_est = (weights * neighbor_vals).sum(axis=1) / np.where(weight_totals == 0, np.nan, weight_totals)

    pm25_surfaces.append(pm_est)
    metric_rows.append(
        {
            "date": day,
            "risk_pct": 100 * float(np.nanmean(pm_est >= StandardRisk_pm25)),
            "mean_pm25": float(np.nanmean(pm_est)),
            "median_pm25": float(np.nanmedian(pm_est)),
            "p90_pm25": float(np.nanpercentile(pm_est, 90)),
        }
    )

Statewide_pm25 = pd.DataFrame(metric_rows)


Wildfire_daily = pd.DataFrame(
    columns=["date", "Wildfire_count", "Wildfire_acres", "transport_ready_acres", "raw_Wildfire_influence", "smoke_transport_score",
             "smoke_transport_score_5day", "wind_from_deg", "wind_speed_kmh",
    ]
)


if not WildfireData.empty and not wind_df.empty:
    Wildfires = WildfireData.merge(wind_df, on="date", how="left")
    Wildfires = Wildfires.dropna(subset=["wind_from_deg", "wind_speed_kmh"])

    if not Wildfires.empty:
        la24H = np.radians(Idaho_Centroid.y)
        lon1 = np.radians(Idaho_Centroid.x)
        lat2 = np.radians(Wildfires["Wildfire_lat"].to_numpy())
        lon2 = np.radians(Wildfires["Wildfire_lon"].to_numpy())

        DiffLat = lat2 - la24H
        DiffLon = lon2 - lon1

        a = np.sin(DiffLat / 2) ** 2 + np.cos(la24H) * np.cos(lat2) * np.sin(DiffLon / 2) ** 2
        Wildfires["distance_km"] = 6371.0 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

        x = np.sin(DiffLon) * np.cos(lat2)
        y = np.cos(la24H) * np.sin(lat2) - np.sin(la24H) * np.cos(lat2) * np.cos(DiffLon)
        Wildfires["bearing_deg"] = (np.degrees(np.arctan2(x, y)) + 360) % 360

        Wildfires["angle_diff_deg"] = np.abs(
            (Wildfires["bearing_deg"] - Wildfires["wind_from_deg"] + 180) % 360 - 180
        )
        Wildfires["direction_weight"] = np.exp(-((Wildfires["angle_diff_deg"] / 35.0) ** 2))
        Wildfires["distance_weight"] = np.exp(-(Wildfires["distance_km"] / 350.0))
        Wildfires["speed_weight"] = np.clip(Wildfires["wind_speed_kmh"] / 15.0, 0.5, 3.0)

        Wildfires["transport_ready_acres"] = (
            Wildfires["Wildfire_acres"] * Wildfires["direction_weight"] * Wildfires["distance_weight"]
        )
        Wildfires["raw_Wildfire_influence"] = Wildfires["transport_ready_acres"] * Wildfires["speed_weight"]

        Wildfire_daily = Wildfires.groupby("date", as_index=False).agg(
            Wildfire_count=("Wildfire_id", "nunique"),
            Wildfire_acres=("Wildfire_acres", "sum"),
            transport_ready_acres=("transport_ready_acres", "sum"),
            raw_Wildfire_influence=("raw_Wildfire_influence", "sum"),
            wind_from_deg=("wind_from_deg", "first"),
            wind_speed_kmh=("wind_speed_kmh", "first"),
        )

        Wildfire_daily = Wildfire_daily.sort_values("date").reset_index(drop=True)

        raw = Wildfire_daily["raw_Wildfire_influence"].fillna(0)
        Wildfire_daily["smoke_transport_score"] = (
            0.15 * raw
            + 0.35 * raw.shift(1, fill_value=0)
            + 0.30 * raw.shift(2, fill_value=0)
            + 0.20 * raw.shift(3, fill_value=0)
        )
        Wildfire_daily["smoke_transport_score_5day"] = (
            Wildfire_daily["smoke_transport_score"].rolling(5, min_periods=1).mean()
        )


# Main handoff for modeling and dashboard
Model_Ready_df = pd.DataFrame({"date": all_dates})
Model_Ready_df = Model_Ready_df.merge(Statewide_pm25, on="date", how="left")
Model_Ready_df = Model_Ready_df.merge(Wildfire_daily, on="date", how="left")

Model_Ready_df = Model_Ready_df.fillna(
    {
        "Wildfire_count": 0,
        "Wildfire_acres": 0,
        "transport_ready_acres": 0,
        "raw_Wildfire_influence": 0,
        "smoke_transport_score": 0,
        "smoke_transport_score_5day": 0,
    }
)

csv_path = Path("outputs/reports/statewide_pm25_and_smoke_transport_all_years.csv")
Model_Ready_df.to_csv(csv_path, index=False)
print("Saved statewide modeling table:", csv_path)


# Yearly line charts
for value_col, file_name, title, yaxis_title in [
    (
        "risk_pct",
        "statewide_risk_by_year.html",
        f"Idaho PM2.5 Risk by Year | threshold={StandardRisk_pm25} µg/m³",
        "Percent of Idaho grid at risk",
    ),
    (
        "mean_pm25",
        "statewide_mean_pm25_by_year.html",
        "Idaho Mean PM2.5 by Year",
        "Mean PM2.5",
    ),
    (
        "smoke_transport_score_5day",
        "smoke_transport_score_by_year.html",
        "Smoke Transport Score by Year | 5-day mean",
        "Smoke transport score",
    ),
]:
    plot_df = Model_Ready_df
    plot_df["date"] = pd.to_datetime(plot_df["date"], errors="coerce")
    plot_df = plot_df.dropna(subset=["date", value_col])

    plot_df["year"] = plot_df["date"].dt.year.astype(str)
    plot_df["plot_date"] = pd.to_datetime(
        "2000-" + plot_df["date"].dt.strftime("%m-%d"),
        errors="coerce"
    )
    plot_df = plot_df.dropna(subset=["plot_date"])

    chart_plot = go.Figure()

    for year in sorted(plot_df["year"].unique()):
        part = plot_df[plot_df["year"] == year].sort_values("plot_date")
        chart_plot.add_trace(
            go.Scatter(
                x=part["plot_date"],
                y=part[value_col],
                mode="lines",
                name=year,
                hovertemplate=(
                    f"Year: {year}"
                    "<br>Date: %{x|%b %d}"
                    "<br>Value: %{y:.1f}<extra></extra>"
                ),
            )
        )

    chart_plot.update_layout(
        title=title,
        xaxis_title="Month",
        yaxis_title=yaxis_title,
        hovermode="x unified",
        margin=dict(l=40, r=20, t=60, b=40),
    )
    chart_plot.update_xaxes(tickformat="%b")

    chart_path = Path("outputs/reports") / file_name
    chart_plot.write_html(chart_path, include_plotlyjs="cdn")
    print("Saved:", chart_path)

surface_matrix = np.vstack(pm25_surfaces)
band_ids = pd.cut(idaho_grid["latitude"], bins=n_bands, labels=False, include_lowest=True)

y_vals = []
z_rows = []
for band in sorted(pd.Series(band_ids).dropna().astype(int).unique()):
    mask = (band_ids == band).to_numpy()
    if mask.sum() == 0:
        continue

    y_vals.append(round(float(idaho_grid.loc[mask, "latitude"].mean()), 3))
    z_rows.append(np.nanmean(surface_matrix[:, mask], axis=1))
z = np.vstack(z_rows)

heatmap_plot = go.Figure(
    data=go.Heatmap(
        x=pd.to_datetime(all_dates),
        y=y_vals,
        z=z,
        colorscale="YlOrRd",
        colorbar=dict(title="PM2.5"),
        hovertemplate=(
            "Date: %{x|%Y-%m-%d}"
            "<br>Lat band: %{y}"
            "<br>PM2.5: %{z:.1f}<extra></extra>"
        ),
    )
)


# Average PM2.5 map
avg_map_file = Path("outputs/reports/average_pm25_map.html")
avg_pm25 = np.nanmean(surface_matrix, axis=0)

display_lat_count = 260
display_lon_count = int(display_lat_count * (maxx - minx) / (maxy - miny))

display_lats = np.linspace(miny, maxy, display_lat_count)
display_lons = np.linspace(minx, maxx, display_lon_count)

lon_mesh, lat_mesh = np.meshgrid(display_lons, display_lats)

source_xy = np.radians(idaho_grid[["latitude", "longitude"]].to_numpy())
target_xy = np.radians(np.column_stack([lat_mesh.ravel(), lon_mesh.ravel()]))

display_tree = BallTree(source_xy, metric="haversine")
display_dist, display_idx = display_tree.query(target_xy, k=min(12, len(idaho_grid)))
display_dist_km = display_dist * 6371.0

display_neighbor_vals = avg_pm25[display_idx]
display_weights = 1.0 / (np.power(display_dist_km, 2.0) + 1e-6)

display_bad = np.isnan(display_neighbor_vals)
display_weights = np.where(display_bad, 0.0, display_weights)
display_neighbor_vals = np.where(display_bad, 0.0, display_neighbor_vals)

display_weight_totals = display_weights.sum(axis=1)
display_values = (display_weights * display_neighbor_vals).sum(axis=1) / np.where(
    display_weight_totals == 0,
    np.nan,
    display_weight_totals,
)
display_z = display_values.reshape(len(display_lats), len(display_lons))

inside_mask = np.array(
    [Idaho_Border.contains(Point(lon, lat)) for lat, lon in zip(lat_mesh.ravel(), lon_mesh.ravel())]
).reshape(len(display_lats), len(display_lons))
display_z[~inside_mask] = np.nan

display_zmax = float(np.nanpercentile(display_z, 97))

Avg_pm25_map = go.Figure()
Avg_pm25_map.add_trace(
    go.Heatmap(
        x=display_lons,
        y=display_lats,
        z=display_z,
        zsmooth="best",
        colorscale="YlOrRd",
        zmin=0,
        zmax=display_zmax,
        colorbar=dict(title="Avg PM2.5"),
        hovertemplate=(
            "Lon: %{x:.3f}"
            "<br>Lat: %{y:.3f}"
            "<br>Avg PM2.5: %{z:.1f}<extra></extra>"
        ),
    )
)

boundary = gpd.GeoSeries([Idaho_Border], crs="EPSG:4326").boundary.iloc[0]

if boundary.geom_type == "MultiLineString":
    for line in boundary.geoms:
        x, y = line.xy
        Avg_pm25_map.add_trace(
            go.Scatter(
                x=list(x),
                y=list(y),
                mode="lines",
                line=dict(width=3, color="black"),
                hoverinfo="skip",
                showlegend=False,
            )
        )
else:
    x, y = boundary.xy
    Avg_pm25_map.add_trace(
        go.Scatter(
            x=list(x),
            y=list(y),
            mode="lines",
            line=dict(width=3, color="black"),
            hoverinfo="skip",
            showlegend=False,
        )
    )

Avg_pm25_map.update_layout(
    title=f"Average PM2.5 Across Idaho Over {start_date.year}-{end_date.year}",
    template="simple_white",
    margin=dict(l=20, r=20, t=60, b=20),
    xaxis=dict(visible=False),
    yaxis=dict(visible=False, scaleanchor="x", scaleratio=1),
)

Avg_pm25_map.write_html(avg_map_file, include_plotlyjs="cdn")
print("Saved:", avg_map_file)


# Latitude heatmap
heatmap_plot.update_layout(
    title="Idaho Daily PM2.5 Heatmap by Latitude Band",
    xaxis_title="Date",
    yaxis_title="Latitude band (south to north)",
    margin=dict(l=60, r=20, t=60, b=40),
)
heatmap_path = Path("outputs/reports/pm25_latitude_heatmap.html")
heatmap_plot.write_html(heatmap_path, include_plotlyjs="cdn")
print("Saved:", heatmap_path)
print("Data wrangling finished. Statewide table and the simple charts are ready.")
