### Forecast Modeling

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

test_days = 383 #Roughly 20% of full data. 80:20 Train/Test Ratio is my go to.

Path("outputs/reports/modeling").mkdir(parents=True, exist_ok=True)
full_data_file = Path("outputs/reports/statewide_pm25_and_smoke_transport_all_years.csv")
Model_Data = pd.read_csv(full_data_file)
Model_Data["date"] = pd.to_datetime(Model_Data["date"])
Model_Data = Model_Data.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

for col in ["risk_pct", "Wildfire_count", "Wildfire_acres", "smoke_transport_score_5day", "wind_from_deg", "wind_speed_kmh",
]:
    Model_Data[col] = Model_Data[col].fillna(0)

print("Modeling rows:", len(Model_Data))
print("Dates:", Model_Data["date"].min().date(), "to", Model_Data["date"].max().date())

# Targets
Model_Data["target_mean_pm25_24H"] = Model_Data["mean_pm25"].shift(-1)
Model_Data["target_mean_pm25_72H"] = Model_Data["mean_pm25"].shift(-3)
Model_Data["target_delta_pm25_24H"] = Model_Data["target_mean_pm25_24H"] - Model_Data["mean_pm25"]
Model_Data["target_delta_pm25_72H"] = Model_Data["target_mean_pm25_72H"] - Model_Data["mean_pm25"]

# Date features
Model_Data["day_of_week"] = Model_Data["date"].dt.dayofweek
Model_Data["month"] = Model_Data["date"].dt.month
Model_Data["day_of_year"] = Model_Data["date"].dt.dayofyear
Model_Data["doy_sin"] = np.sin(2 * np.pi * Model_Data["day_of_year"] / 365.25)
Model_Data["doy_cos"] = np.cos(2 * np.pi * Model_Data["day_of_year"] / 365.25)

# Wind direction
wind_radians = np.deg2rad(Model_Data["wind_from_deg"])
Model_Data["wind_dir_sin"] = np.sin(wind_radians)
Model_Data["wind_dir_cos"] = np.cos(wind_radians)

# Lag features
Model_Data["pm25_yesterday"] = Model_Data["mean_pm25"].shift(1)
Model_Data["pm25_two_days_ago"] = Model_Data["mean_pm25"].shift(2)
Model_Data["pm25_three_days_ago"] = Model_Data["mean_pm25"].shift(3)
Model_Data["pm25_last_week"] = Model_Data["mean_pm25"].shift(7)

Model_Data["pm25_recent_avg"] = Model_Data["mean_pm25"].shift(1).rolling(3, min_periods=1).mean()
Model_Data["pm25_week_avg"] = Model_Data["mean_pm25"].shift(1).rolling(7, min_periods=1).mean()
Model_Data["pm25_recent_std"] = Model_Data["mean_pm25"].shift(1).rolling(7, min_periods=2).std()
Model_Data["pm25_recent_delta"] = Model_Data["mean_pm25"].shift(1) - Model_Data["mean_pm25"].shift(2)
Model_Data["pm25_week_delta"] = Model_Data["mean_pm25"].shift(1) - Model_Data["mean_pm25"].shift(7)

Model_Data["risk_yesterday"] = Model_Data["risk_pct"].shift(1)
Model_Data["smoke_recent"] = Model_Data["smoke_transport_score_5day"].shift(1).rolling(3, min_periods=1).mean()
Model_Data["smoke_week_avg"] = Model_Data["smoke_transport_score_5day"].shift(1).rolling(7, min_periods=1).mean()
Model_Data["Wildfire_recent"] = Model_Data["Wildfire_acres"].shift(1).rolling(3, min_periods=1).mean()
Model_Data["Wildfire_week_avg"] = Model_Data["Wildfire_acres"].shift(1).rolling(7, min_periods=1).mean()
Model_Data["wind_yesterday"] = Model_Data["wind_speed_kmh"].shift(1)
Model_Data["wind_recent_avg"] = Model_Data["wind_speed_kmh"].shift(1).rolling(3, min_periods=1).mean()

for col in ["pm25_recent_std", "pm25_recent_delta", "pm25_week_delta", "risk_yesterday", "smoke_recent",
            "smoke_week_avg", "Wildfire_recent", "Wildfire_week_avg", "wind_yesterday", "wind_recent_avg",
]:
    Model_Data[col] = Model_Data[col].fillna(0.0)

feature_columns = ["mean_pm25", "median_pm25", "risk_pct", "smoke_transport_score_5day", "Wildfire_count",
                   "Wildfire_acres", "wind_speed_kmh", "wind_dir_sin", "wind_dir_cos", "day_of_week", "month",
                   "doy_sin", "doy_cos", "pm25_yesterday", "pm25_two_days_ago", "pm25_three_days_ago",
                   "pm25_last_week", "pm25_recent_avg", "pm25_week_avg", "pm25_recent_std", "pm25_recent_delta",
                   "pm25_week_delta", "risk_yesterday", "smoke_recent", "smoke_week_avg", "Wildfire_recent",
                   "Wildfire_week_avg", "wind_yesterday", "wind_recent_avg",
]

model_ready_file = Path("outputs/reports/modeling/model_ready_dataset.csv")
Model_Data.to_csv(model_ready_file, index=False)
cutoff_date = Model_Data["date"].max() - pd.Timedelta(days=test_days)

metrics_rows = []
latest_forecast_rows = []
predictions_24H = pd.DataFrame()
predictions_72H = pd.DataFrame()

alpha_choices = [0.01, 0.1, 1.0, 10, 100]

for Forecast_name, target_col, delta_col, days_ahead in [
    ("24H", "target_mean_pm25_24H", "target_delta_pm25_24H", 1),
    ("72H", "target_mean_pm25_72H", "target_delta_pm25_72H", 3),
]:
    usable_rows = Model_Data.dropna(subset=feature_columns + [target_col]).copy()
    if usable_rows.empty:
        continue

    train = usable_rows[usable_rows["date"] <= cutoff_date].copy()
    test = usable_rows[usable_rows["date"] > cutoff_date].copy()

    print(Forecast_name, "train rows:", len(train), "test rows:", len(test))

    if train.empty or test.empty:
        continue

    x_train = train[feature_columns]
    y_train = train[target_col].to_numpy(dtype=float)
    y_train_delta = train[delta_col].to_numpy(dtype=float)

    x_test = test[feature_columns]
    y_test = test[target_col].to_numpy(dtype=float)

    # Models
    baseline_pred = test["mean_pm25"].to_numpy(dtype=float)

    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(x_train, y_train)
    ridge_pred = ridge_model.predict(x_test)

    # Time-series validation for alpha
    best_scaled_alpha = alpha_choices[0]
    best_scaled_mae = np.inf
    best_delta_alpha = alpha_choices[0]
    best_delta_mae = np.inf

    if len(train) >= 20:
        split_count = min(5, len(train) // 10)
        split_count = max(2, split_count)
        tscv = TimeSeriesSplit(n_splits=split_count)

        for alpha in alpha_choices:
            scaled_fold_maes = []
            delta_fold_maes = []

            for cv_train_idx, cv_valid_idx in tscv.split(x_train):
                x_cv_train = x_train.iloc[cv_train_idx]
                x_cv_valid = x_train.iloc[cv_valid_idx]

                y_cv_train = y_train[cv_train_idx]
                y_cv_valid = y_train[cv_valid_idx]

                y_cv_train_delta = y_train_delta[cv_train_idx]
                y_cv_valid_delta = y_train_delta[cv_valid_idx]

                scaled_model = Pipeline([
                    ("scaler", StandardScaler()),
                    ("ridge", Ridge(alpha=alpha)),
                ])
                scaled_model.fit(x_cv_train, y_cv_train)
                scaled_pred = scaled_model.predict(x_cv_valid)
                scaled_fold_maes.append(mean_absolute_error(y_cv_valid, scaled_pred))

                delta_model = Pipeline([
                    ("scaler", StandardScaler()),
                    ("ridge", Ridge(alpha=alpha)),
                ])
                delta_model.fit(x_cv_train, y_cv_train_delta)
                delta_pred = delta_model.predict(x_cv_valid) + x_cv_valid["mean_pm25"].to_numpy(dtype=float)
                delta_fold_maes.append(mean_absolute_error(y_cv_valid, delta_pred))

            avg_scaled_mae = float(np.mean(scaled_fold_maes))
            avg_delta_mae = float(np.mean(delta_fold_maes))

            if avg_scaled_mae < best_scaled_mae:
                best_scaled_mae = avg_scaled_mae
                best_scaled_alpha = alpha

            if avg_delta_mae < best_delta_mae:
                best_delta_mae = avg_delta_mae
                best_delta_alpha = alpha

    print(Forecast_name, "best scaled alpha:", best_scaled_alpha)
    print(Forecast_name, "best delta alpha:", best_delta_alpha)

    scaled_ridge_model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=best_scaled_alpha)),
    ])
    scaled_ridge_model.fit(x_train, y_train)
    scaled_ridge_pred = scaled_ridge_model.predict(x_test)

    delta_ridge_model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=best_delta_alpha)),
    ])
    delta_ridge_model.fit(x_train, y_train_delta)
    delta_ridge_delta_pred = delta_ridge_model.predict(x_test)
    delta_ridge_pred = delta_ridge_delta_pred + test["mean_pm25"].to_numpy(dtype=float)

    current_values = test["mean_pm25"].to_numpy(dtype=float)
    actual_delta = test[delta_col].to_numpy(dtype=float)

    for model_name, predicted_values, extra_notes in [
        ("baseline", baseline_pred, "same as current PM2.5"),
        ("ridge", ridge_pred, "plain ridge just for comparison"),
        ("scaled_ridge", scaled_ridge_pred, f"scaled ridge with alpha {best_scaled_alpha}"),
        ("delta_ridge", delta_ridge_pred, f"scaled ridge on delta with alpha {best_delta_alpha}"),
    ]:
        predicted_values = np.asarray(predicted_values, dtype=float)
        predicted_delta = predicted_values - current_values

        smape_weight_totals = np.abs(y_test) + np.abs(predicted_values)
        smape = np.where(
            smape_weight_totals == 0,
            0.0,
            2.0 * np.abs(predicted_values - y_test) / smape_weight_totals,
        ).mean() * 100

        metrics_rows.append(
            {
                "Forecast": Forecast_name,
                "days_ahead": days_ahead,
                "model": model_name,
                "notes": extra_notes,
                "cutoff_date": cutoff_date.date().isoformat(),
                "train_rows": len(train),
                "test_rows": len(test),
                "mae": float(mean_absolute_error(y_test, predicted_values)),
                "rmse": float(np.sqrt(mean_squared_error(y_test, predicted_values))),
                "smape": float(smape),
                "r2": float(r2_score(y_test, predicted_values)),
                "delta_mae": float(mean_absolute_error(actual_delta, predicted_delta)),
            }
        )

    Forecast_predictions = test[["date", "mean_pm25", target_col, delta_col]].copy()
    Forecast_predictions = Forecast_predictions.rename(
        columns={
            "mean_pm25": "current_mean_pm25",
            target_col: "actual_mean_pm25",
            delta_col: "actual_delta_pm25",
        }
    )

    Forecast_predictions["Forecast"] = Forecast_name
    Forecast_predictions["baseline_pred_mean_pm25"] = baseline_pred
    Forecast_predictions["ridge_pred_mean_pm25"] = ridge_pred
    Forecast_predictions["scaled_ridge_pred_mean_pm25"] = scaled_ridge_pred
    Forecast_predictions["delta_ridge_pred_mean_pm25"] = delta_ridge_pred

    Forecast_predictions["baseline_pred_delta_pm25"] = (
        Forecast_predictions["baseline_pred_mean_pm25"] - Forecast_predictions["current_mean_pm25"]
    )
    Forecast_predictions["ridge_pred_delta_pm25"] = (
        Forecast_predictions["ridge_pred_mean_pm25"] - Forecast_predictions["current_mean_pm25"]
    )
    Forecast_predictions["scaled_ridge_pred_delta_pm25"] = (
        Forecast_predictions["scaled_ridge_pred_mean_pm25"] - Forecast_predictions["current_mean_pm25"]
    )
    Forecast_predictions["delta_ridge_pred_delta_pm25"] = (
        Forecast_predictions["delta_ridge_pred_mean_pm25"] - Forecast_predictions["current_mean_pm25"]
    )

    if Forecast_name == "24H":
        predictions_24H = Forecast_predictions.copy()
    else:
        predictions_72H = Forecast_predictions.copy()

    # Refit on all usable rows for latest forecast
    final_ridge = Ridge(alpha=1.0)
    final_ridge.fit(usable_rows[feature_columns], usable_rows[target_col])

    final_scaled_ridge = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=best_scaled_alpha)),
    ])
    final_scaled_ridge.fit(usable_rows[feature_columns], usable_rows[target_col])

    final_delta_ridge = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=best_delta_alpha)),
    ])
    final_delta_ridge.fit(usable_rows[feature_columns], usable_rows[delta_col])

    latest_features = Model_Data.dropna(subset=feature_columns).copy()
    if latest_features.empty:
        continue

    latest_row = latest_features.iloc[[-1]].copy()
    latest_date = pd.to_datetime(latest_row["date"].iloc[0])
    forecast_date = latest_date + pd.Timedelta(days=days_ahead)

    current_mean = float(latest_row["mean_pm25"].iloc[0])
    latest_x = latest_row[feature_columns]

    ridge_latest = float(final_ridge.predict(latest_x)[0])
    scaled_ridge_latest = float(final_scaled_ridge.predict(latest_x)[0])
    delta_ridge_latest = float(final_delta_ridge.predict(latest_x)[0] + current_mean)

    for model_name, pred_mean in [
        ("baseline", current_mean),
        ("ridge", ridge_latest),
        ("scaled_ridge", scaled_ridge_latest),
        ("delta_ridge", delta_ridge_latest),
    ]:
        latest_forecast_rows.append(
            {
                "source_date": latest_date.date().isoformat(),
                "forecast_date": forecast_date.date().isoformat(),
                "Forecast": Forecast_name,
                "days_ahead": days_ahead,
                "model": model_name,
                "current_mean_pm25": current_mean,
                "pred_mean_pm25": pred_mean,
                "pred_delta_pm25": pred_mean - current_mean,
            }
        )

metrics_df = pd.DataFrame(metrics_rows)
latest_forecasts_df = pd.DataFrame(latest_forecast_rows)

metrics_csv = Path("outputs/reports/modeling/metrics_summary.csv")
latest_forecasts_csv = Path("outputs/reports/modeling/latest_forecasts.csv")
predictions_24H_csv = Path("outputs/reports/modeling/test_predictions_24H.csv")
predictions_72H_csv = Path("outputs/reports/modeling/test_predictions_72H.csv")
predictions_csv = Path("outputs/reports/modeling/test_predictions.csv")
metrics_json = Path("outputs/reports/modeling/metrics_summary.json")

metrics_df.to_csv(metrics_csv, index=False)
latest_forecasts_df.to_csv(latest_forecasts_csv, index=False)
predictions_24H.to_csv(predictions_24H_csv, index=False)
predictions_72H.to_csv(predictions_72H_csv, index=False)
pd.concat([predictions_24H, predictions_72H], ignore_index=True).to_csv(predictions_csv, index=False)

with open(metrics_json, "w", encoding="utf-8") as file:
    json.dump(metrics_rows, file, indent=2)

print("Saved modeling files:")
print("-", model_ready_file)
print("-", metrics_csv)
print("-", latest_forecasts_csv)
print("-", predictions_24H_csv)
print("-", predictions_72H_csv)
print("-", metrics_json)