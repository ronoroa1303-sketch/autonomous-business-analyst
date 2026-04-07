"""
agents/forecast_agent.py

Forecasting Agent — predicts future values from time-series data.

How it works (viva explanation):
  1. Receive a list of {date, value} dicts.
  2. Convert to a pandas DataFrame with columns 'ds' (date) and 'y' (value)
     — Prophet requires this exact naming convention.
  3. Train a Prophet model on the historical data.
  4. Ask Prophet to forecast the next 7 days.
  5. Detect whether the forecasted trend is increasing, decreasing, or stable.
  6. Return the forecast + trend label.

Fallback: if Prophet fails for any reason, we fall back to ARIMA (statsmodels).
"""

import pandas as pd
from datetime import timedelta


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_forecast(time_series: list) -> dict:
    """
    Accepts a list of dicts: [{"date": "YYYY-MM-DD", "value": number}, ...]
    Returns:
      {
        "forecast": [
          {
            "date": "YYYY-MM-DD", 
            "predicted_value": float,
            "lower_bound": float,
            "upper_bound": float
          }, ...
        ],
        "trend":    "increasing" | "decreasing" | "stable"
      }
    or on bad input:
      {"error": "insufficient_data"}
    """

    # ── Step 1: Validate minimum data requirement ────────────────────────────
    # We need at least 5 data points to fit a meaningful model.
    if not time_series or len(time_series) < 5:
        return {"error": "insufficient_data"}

    # ── Step 2: Build the DataFrame ──────────────────────────────────────────
    # Prophet requires columns named exactly 'ds' (datestamp) and 'y' (value).
    df = pd.DataFrame(time_series)          # columns: date, value
    df = df.rename(columns={"date": "ds", "value": "y"})
    df["ds"] = pd.to_datetime(df["ds"])     # ensure proper datetime type
    df["y"]  = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna()                        # remove any rows with bad data
    df = df.sort_values("ds").reset_index(drop=True)

    # Handle missing dates: ensure daily frequency and forward fill
    if not df.empty:
        df = df.set_index("ds")
        df = df.resample("D").ffill()
        df = df.reset_index()

    # Re-check after cleaning
    if len(df) < 5:
        return {"error": "insufficient_data"}

    # ── Step 3: Try Prophet first ────────────────────────────────────────────
    try:
        forecast_rows = _prophet_forecast(df, periods=7)
    except Exception as prophet_err:
        print(f"Prophet failed: {prophet_err} — trying ARIMA fallback")
        try:
            forecast_rows = _arima_forecast(df, periods=7)
        except Exception as arima_err:
            print(f"ARIMA also failed: {arima_err}")
            return {"error": f"Forecasting failed: {arima_err}"}

    # ── Step 4: Detect trend from the forecasted values ──────────────────────
    trend = _detect_trend(forecast_rows)

    return {
        "forecast": forecast_rows,
        "trend":    trend
    }


# ─────────────────────────────────────────────────────────────────────────────
# Prophet forecasting
# ─────────────────────────────────────────────────────────────────────────────

def _prophet_forecast(df: pd.DataFrame, periods: int) -> list:
    """
    Train a Prophet model and return `periods` daily forecasts.

    Prophet is a decomposable time-series model by Meta that handles
    seasonality and holidays automatically — ideal for business data.
    """
    from prophet import Prophet  # import here so ARIMA path works if Prophet isn't installed

    # Initialise Prophet with daily seasonality.
    # We suppress verbose Stan output (seasonality_mode='additive' is default).
    model = Prophet(
        daily_seasonality=False,    # no sub-daily patterns in daily data
        weekly_seasonality=True,    # capture day-of-week patterns
        yearly_seasonality=True,    # capture seasonal patterns
        interval_width=0.80,        # 80% uncertainty interval
    )

    # Fit the model on the historical data
    model.fit(df)

    # Create a future DataFrame: extends `periods` days beyond the last date
    future = model.make_future_dataframe(periods=periods, freq="D")

    # Generate the forecast
    forecast = model.predict(future)

    # Extract only the future rows (not the historical fitted values)
    future_forecast = forecast.tail(periods)

    # Build the output list
    result = []
    for _, row in future_forecast.iterrows():
        result.append({
            "date":            row["ds"].strftime("%Y-%m-%d"),
            "predicted_value": round(float(row["yhat"]), 4),
            "lower_bound":     round(float(row["yhat_lower"]), 4),
            "upper_bound":     round(float(row["yhat_upper"]), 4)
        })

    return result


# ─────────────────────────────────────────────────────────────────────────────
# ARIMA fallback
# ─────────────────────────────────────────────────────────────────────────────

def _arima_forecast(df: pd.DataFrame, periods: int) -> list:
    """
    Fallback forecasting using ARIMA (statsmodels).

    ARIMA(p, d, q):
      p = autoregressive order  → how many past values to use
      d = differencing order    → how many times to difference for stationarity
      q = moving-average order  → how many past errors to use

    We use (2, 1, 2) as a simple but reasonable default.
    """
    from statsmodels.tsa.arima.model import ARIMA

    # Fit ARIMA on the 'y' column
    model  = ARIMA(df["y"].values, order=(2, 1, 2))
    result = model.fit()

    # Forecast `periods` steps ahead and get confidence intervals
    forecast_result = result.get_forecast(steps=periods)
    summary = forecast_result.summary_frame()

    # Build date range starting the day after the last historical date
    last_date = df["ds"].iloc[-1]
    result_rows = []
    for i in range(periods):
        future_date = last_date + timedelta(days=i + 1)
        mean_val = summary["mean"].iloc[i]
        lower_val = summary["mean_ci_lower"].iloc[i]
        upper_val = summary["mean_ci_upper"].iloc[i]

        result_rows.append({
            "date":            future_date.strftime("%Y-%m-%d"),
            "predicted_value": round(float(mean_val), 4),
            "lower_bound":     round(float(lower_val), 4),
            "upper_bound":     round(float(upper_val), 4)
        })

    return result_rows


# ─────────────────────────────────────────────────────────────────────────────
# Trend detection
# ─────────────────────────────────────────────────────────────────────────────

def _detect_trend(forecast_rows: list) -> str:
    """
    Compare the first and last predicted values to determine trend direction.

    Simple rule:
      - If last > first by more than 1% → "increasing"
      - If last < first by more than 1% → "decreasing"
      - Otherwise                       → "stable"

    We use a 1% threshold to avoid labelling tiny noise as a trend.
    """
    if len(forecast_rows) < 2:
        return "stable"

    first_val = forecast_rows[0]["predicted_value"]
    last_val  = forecast_rows[-1]["predicted_value"]

    if first_val == 0:
        # Avoid division by zero; compare absolute difference
        if last_val > first_val:
            return "increasing"
        elif last_val < first_val:
            return "decreasing"
        return "stable"

    change_pct = (last_val - first_val) / abs(first_val)

    if change_pct > 0.01:       # more than 1% increase
        return "increasing"
    elif change_pct < -0.01:    # more than 1% decrease
        return "decreasing"
    else:
        return "stable"
