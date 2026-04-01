# Streamlit Dashboard

from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Idaho PM2.5 Forecast Dashboard", layout="wide")
st.title("Idaho PM2.5 Forecast Risk Dashboard")
st.caption("Statewide PM2.5 trends, smoke context, and short forecasts")

pipeline_file = Path("outputs/reports/statewide_pm25_and_smoke_transport_all_years.csv")
metrics_file = Path("outputs/reports/modeling/metrics_summary.csv")
latest_file = Path("outputs/reports/modeling/latest_forecasts.csv")
pred_24H_file = Path("outputs/reports/modeling/test_predictions_24H.csv")
pred_72H_file = Path("outputs/reports/modeling/test_predictions_72H.csv")
avg_map_file = Path("outputs/reports/average_pm25_map.html")
heatmap_file = Path("outputs/reports/pm25_latitude_heatmap.html")
pipeline_df = pd.read_csv(pipeline_file)
metrics_df = pd.read_csv(metrics_file)
latest_df = pd.read_csv(latest_file)
pred_24H_df = pd.read_csv(pred_24H_file) if pred_24H_file.exists() else pd.DataFrame()
pred_72H_df = pd.read_csv(pred_72H_file) if pred_72H_file.exists() else pd.DataFrame()

# Clean main pipeline data
pipeline_df["date"] = pd.to_datetime(pipeline_df["date"])
pipeline_df["mean_pm25"] = pd.to_numeric(pipeline_df["mean_pm25"])
pipeline_df["smoke_transport_score_5day"] = pd.to_numeric(pipeline_df["smoke_transport_score_5day"])


# Clean latest forecast data
for col in ["source_date", "forecast_date"]:
    if col in latest_df.columns:
        latest_df[col] = pd.to_datetime(latest_df[col], errors="coerce")

for col in ["current_mean_pm25", "pred_mean_pm25", "pred_delta_pm25"]:
    if col in latest_df.columns:
        latest_df[col] = pd.to_numeric(latest_df[col], errors="coerce")

for col in ["model", "Forecast"]:
    if col in latest_df.columns:
        latest_df[col] = latest_df[col].astype(str)

pipeline_actuals = pipeline_df.dropna(subset=["date", "mean_pm25"]).sort_values("date").copy()

if pipeline_actuals.empty:
    st.error("The statewide pipeline file loaded, but there were no usable PM2.5 rows.")
    st.stop()

latest_actual = pipeline_actuals.iloc[-1]
current_pm25 = float(latest_actual["mean_pm25"])
current_date = latest_actual["date"]

if current_pm25 < 12:
    current_risk = "Good"
elif current_pm25 < 35.5:
    current_risk = "Moderate"
elif current_pm25 < 55.5:
    current_risk = "Unhealthy (Sensitive Groups)"
elif current_pm25 < 91.5:
    current_risk = "Unhealthy (All Groups)"
else:
    current_risk = "Hazardous"

# Pick best model for each Forecast
best_models = {"24H": None, "72H": None}

if {"Forecast", "model", "mae"}.issubset(metrics_df.columns):
    for Forecast in ["24H", "72H"]:
        Forecast_metrics = metrics_df[metrics_df["Forecast"] == Forecast]
        Forecast_metrics = Forecast_metrics.dropna(subset=["mae"]).sort_values("mae")
        if not Forecast_metrics.empty:
            best_models[Forecast] = str(Forecast_metrics.iloc[0]["model"])

if {"Forecast", "model"}.issubset(latest_df.columns):
    for Forecast in ["24H", "72H"]:
        if best_models[Forecast] is None:
            Forecast_models = latest_df[latest_df["Forecast"] == Forecast]["model"].dropna().tolist()
            if Forecast_models:
                best_models[Forecast] = str(Forecast_models[0])

latest_best = {"24H": pd.DataFrame(), "72H": pd.DataFrame()}

if {"Forecast", "model"}.issubset(latest_df.columns):
    for Forecast in ["24H", "72H"]:
        if best_models[Forecast] is not None:
            latest_best[Forecast] = latest_df[
                (latest_df["Forecast"] == Forecast) & (latest_df["model"] == best_models[Forecast])
            ].copy().sort_values("source_date").tail(1)

tomorrow_pm25 = None
tomorrow_delta = None
day3_pm25 = None
day3_delta = None

if not latest_best["24H"].empty:
    tomorrow_pm25 = float(latest_best["24H"]["pred_mean_pm25"].iloc[0])
    tomorrow_delta = float(latest_best["24H"]["pred_delta_pm25"].iloc[0])

if not latest_best["72H"].empty:
    day3_pm25 = float(latest_best["72H"]["pred_mean_pm25"].iloc[0])
    day3_delta = float(latest_best["72H"]["pred_delta_pm25"].iloc[0])

Name24H = best_models["24H"].replace("_", " ") if best_models["24H"] is not None else "N/A"
Name72H = best_models["72H"].replace("_", " ") if best_models["72H"] is not None else "N/A"

col1, col2, col3, col4 = st.columns(4, gap = None)
col1.metric("Current PM2.5", f"{current_pm25:.1f}")
col2.metric(
    "Tomorrow",
    f"{tomorrow_pm25:.1f}" if tomorrow_pm25 is not None else "N/A",
    f"{tomorrow_delta:+.1f}" if tomorrow_delta is not None else None,
)
col3.metric(
    "72 hours",
    f"{day3_pm25:.1f}" if day3_pm25 is not None else "N/A",
    f"{day3_delta:+.1f}" if day3_delta is not None else None,
)
col4.metric("Risk level", current_risk)

st.markdown(
    f"As of **{current_date.date()}**, Idaho's statewide mean PM2.5 is **{current_pm25:.1f}**. "
    f"The current best model for **24 hours** is **{Name24H}**, "
    f"and the current best model for **72 hours** is **{Name72H}**."
)

# Recent actuals and current forecast
recent_actuals = pipeline_actuals.tail(60).copy()

Forecast_Plot = go.Figure()
Forecast_Plot.add_trace(
    go.Scatter(
        x=recent_actuals["date"],
        y=recent_actuals["mean_pm25"],
        mode="lines",
        name="Actual PM2.5",
    )
)

if not latest_best["24H"].empty:
    Forecast_Plot.add_trace(
        go.Scatter(
            x=[
                latest_best["24H"]["source_date"].iloc[0],
                latest_best["24H"]["forecast_date"].iloc[0],
            ],
            y=[
                latest_best["24H"]["current_mean_pm25"].iloc[0],
                latest_best["24H"]["pred_mean_pm25"].iloc[0],
            ],
            mode="lines+markers",
            name=f"24-hour forecast ({Name24H})",
        )
    )

if not latest_best["72H"].empty:
    Forecast_Plot.add_trace(
        go.Scatter(
            x=[
                latest_best["72H"]["source_date"].iloc[0],
                latest_best["72H"]["forecast_date"].iloc[0],
            ],
            y=[
                latest_best["72H"]["current_mean_pm25"].iloc[0],
                latest_best["72H"]["pred_mean_pm25"].iloc[0],
            ],
            mode="lines+markers",
            name=f"72-hour forecast ({Name72H})",
        )
    )

Forecast_Plot.update_layout(
    title="Recent PM2.5 and Current Forecast",
    xaxis_title="Date",
    yaxis_title="Mean PM2.5",
    hovermode="x unified",
)
st.plotly_chart(Forecast_Plot, use_container_width=True)

left_col, right_col = st.columns([2, 1])

with left_col:
    st.subheader("Smoke transport context")

    context = pipeline_actuals.tail(60).copy()
    context["smoke_transport_score_5day"] = pd.to_numeric(context["smoke_transport_score_5day"])

    Context_plot = go.Figure()
    Context_plot.add_trace(
        go.Scatter(
            x=context["date"],
            y=context["smoke_transport_score_5day"],
            mode="lines",
            name="Smoke transport",
        )
    )
    Context_plot.add_trace(
        go.Scatter(
            x=context["date"],
            y=context["mean_pm25"],
            mode="lines",
            name="Mean PM2.5",
            yaxis="y2",
        )
    )
    Context_plot.update_layout(
        title="Smoke Transport vs PM2.5",
        xaxis_title="Date",
        yaxis=dict(title="Smoke transport score"),
        yaxis2=dict(title="Mean PM2.5", overlaying="y", side="right"),
        hovermode="x unified",
    )
    st.plotly_chart(Context_plot, use_container_width=True)

with right_col:
    st.subheader("Model errors")

    show_cols = ["Forecast", "model", "mae", "rmse", "smape", "r2", "delta_mae"]
    existing_show_cols = [col for col in show_cols if col in metrics_df.columns]

    if existing_show_cols:
        metrics_view = metrics_df[existing_show_cols].copy()

        if "model" in metrics_view.columns:
            metrics_view["model"] = metrics_view["model"].str.replace("_", " ")

        number_cols = [
            col for col in ["mae", "rmse", "smape", "r2", "delta_mae"]
            if col in metrics_view.columns
        ]
        metrics_view[number_cols] = metrics_view[number_cols].round(3)

        st.dataframe(
            metrics_view.sort_values(["Forecast", "mae"]),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.write("The metrics file loaded, but it did not have the columns I expected.")

st.subheader("Latest saved forecast rows")

latest_show_cols = [
    "Forecast",
    "model",
    "source_date",
    "forecast_date",
    "current_mean_pm25",
    "pred_mean_pm25",
    "pred_delta_pm25",
]
existing_latest_cols = [col for col in latest_show_cols if col in latest_df.columns]

if existing_latest_cols:
    latest_view = latest_df[existing_latest_cols].copy()

    if "model" in latest_view.columns:
        latest_view["model"] = latest_view["model"].str.replace("_", " ")

    number_cols = [
        col for col in ["current_mean_pm25", "pred_mean_pm25", "pred_delta_pm25"]
        if col in latest_view.columns
    ]
    latest_view[number_cols] = latest_view[number_cols].round(3)

    st.dataframe(
        latest_view.sort_values(["Forecast", "forecast_date", "model"]).tail(12),
        use_container_width=True,
        hide_index=True,
    )

st.subheader("Test-set results")

tab1, tab2 = st.tabs(["24-hour", "72-hour"])

for tab, pred_df, Forecast, best_model, chart_title in [
    (tab1, pred_24H_df, "24H", best_models["24H"], "24-hour Forecast on Test Data"),
    (tab2, pred_72H_df, "72H", best_models["72H"], "72-hour Forecast on Test Data"),
]:
    with tab:
        if pred_df.empty:
            if Forecast == "24H":
                st.write("The 24-hour test file was not found yet.")
            else:
                st.write("The 72-hour test file was not found yet.")
        else:
            pred_cols = [col for col in pred_df.columns if col.endswith("_pred_mean_pm25")]
            model_order = ["baseline", "ridge", "scaled_ridge", "delta_ridge"]

            ordered_pred_cols = []
            for model_name in model_order:
                col_name = f"{model_name}_pred_mean_pm25"
                if col_name in pred_cols:
                    ordered_pred_cols.append(col_name)

            for col_name in pred_cols:
                if col_name not in ordered_pred_cols:
                    ordered_pred_cols.append(col_name)

            needed_cols = ["date", "actual_mean_pm25"] + ordered_pred_cols
            view = pred_df[needed_cols].dropna(subset=["date", "actual_mean_pm25"]).copy()

            pred_fig = go.Figure()
            pred_fig.add_trace(
                go.Scatter(
                    x=view["date"],
                    y=view["actual_mean_pm25"],
                    mode="lines",
                    name="Actual",
                )
            )

            for col_name in ordered_pred_cols:
                model_name = col_name.replace("_pred_mean_pm25", "")
                display_name = model_name.replace("_", " ")
                if best_model == model_name:
                    display_name = display_name + " (best)"

                pred_fig.add_trace(
                    go.Scatter(
                        x=view["date"],
                        y=view[col_name],
                        mode="lines",
                        name=display_name,
                    )
                )

            pred_fig.update_layout(
                title=chart_title,
                xaxis_title="Date",
                yaxis_title="Mean PM2.5",
                hovermode="x unified",
            )
            st.plotly_chart(pred_fig, use_container_width=True)

            if {"Forecast", "mae"}.issubset(metrics_df.columns):
                Forecast_metrics = metrics_df[metrics_df["Forecast"] == Forecast].copy()
                if not Forecast_metrics.empty:
                    if "model" in Forecast_metrics.columns:
                        Forecast_metrics["model"] = Forecast_metrics["model"].str.replace("_", " ")

                    number_cols = [
                        col for col in ["mae", "rmse", "smape", "r2", "delta_mae"]
                        if col in Forecast_metrics.columns
                    ]
                    Forecast_metrics[number_cols] = Forecast_metrics[number_cols].round(3)

                    st.dataframe(
                        Forecast_metrics.sort_values("mae"),
                        use_container_width=True,
                        hide_index=True,
                    )

st.subheader("Maps")
map_col, heatmap_col = st.columns(2)

with map_col:
    if avg_map_file.exists():
        components.html(avg_map_file.read_text(encoding="utf-8"), height=500, scrolling=False)
    else:
        st.write("The average PM2.5 map is missing.")

with heatmap_col:
    if heatmap_file.exists():
        components.html(heatmap_file.read_text(encoding="utf-8"), height=500, scrolling=False)
    else:
        st.write("The PM2.5 heatmap file is missing.")