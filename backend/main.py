from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import requests
from prophet import Prophet
from datetime import date, datetime
import warnings
import time
import os
import json
warnings.filterwarnings("ignore")

# ── Disk cache (survives Render restarts, TTL = 23 hours) ─────────────────────
CACHE_DIR = "/tmp/stock_cache"
CACHE_TTL = 23 * 3600  # refresh once per day
os.makedirs(CACHE_DIR, exist_ok=True)

def _cache_path(ticker: str) -> str:
    return os.path.join(CACHE_DIR, f"{ticker}.json")

def _cache_get(ticker: str):
    path = _cache_path(ticker)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            obj = json.load(f)
        if time.time() - obj["ts"] < CACHE_TTL:
            print(f"Disk cache hit for {ticker}")
            return pd.DataFrame(obj["data"])
    except Exception:
        pass
    return None

def _cache_set(ticker: str, df: pd.DataFrame):
    path = _cache_path(ticker)
    try:
        with open(path, "w") as f:
            json.dump({"ts": time.time(), "data": df.assign(ds=df["ds"].astype(str)).to_dict(orient="records")}, f)
        print(f"Disk cached {ticker}")
    except Exception as e:
        print(f"Cache write failed: {e}")

app = FastAPI(title="Stock Forecaster API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

TIINGO_TOKEN = "9f2bf23fbabeda4660241d973bc01f8d6661b849"

class ForecastPoint(BaseModel):
    date: str
    predicted: float
    lower: float
    upper: float

class MetricsModel(BaseModel):
    mae: float
    rmse: float
    mape: float
    ci_coverage: float

class ForecastResponse(BaseModel):
    ticker: str
    forecast_days: int
    last_actual_date: str
    last_actual_price: float
    forecast: list[ForecastPoint]
    metrics: MetricsModel


def fetch_data(ticker: str) -> pd.DataFrame:
    cached = _cache_get(ticker)
    if cached is not None:
        cached["ds"] = pd.to_datetime(cached["ds"])
        return cached.copy()

    # Use last 5 years only — older data hurts more than it helps for near-term forecasting
    start_date = (date.today().replace(year=date.today().year - 5)).isoformat()
    url = f"https://api.tiingo.com/tiingo/daily/{ticker}/prices?startDate={start_date}&token={TIINGO_TOKEN}"

    print(f"Fetching data for {ticker} from Tiingo...")
    response = requests.get(url, headers={"Content-Type": "application/json"}, timeout=30)

    if response.status_code == 429:
        raise HTTPException(status_code=429, detail="Too many requests to data provider. Please wait a minute and try again.")
    if response.status_code == 404:
        raise HTTPException(status_code=404, detail=f"Ticker '{ticker}' not found on Tiingo.")
    if response.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Tiingo error {response.status_code}: {response.text[:200]}")

    data = response.json()

    if not data:
        raise HTTPException(status_code=404, detail=f"No data returned for ticker '{ticker}'.")

    df = pd.DataFrame(data)
    df["ds"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    # Use adjClose — accounts for splits and dividends, cleaner signal
    df = df.rename(columns={"adjClose": "y"})
    df["y"] = df["y"].astype(float)
    df = df.sort_values("ds").reset_index(drop=True)
    df = df[["ds", "y"]].dropna(subset=["y"])

    print(f"Fetched {len(df)} rows for {ticker}, last date: {df['ds'].max()}")
    _cache_set(ticker, df)
    return df.copy()


def train_and_forecast(df, forecast_days, interval_width):
    model = Prophet(
        seasonality_mode="multiplicative",
        changepoint_prior_scale=0.05,
        changepoint_range=0.9,
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        interval_width=interval_width,
    )
    model.fit(df[["ds", "y"]])
    future = model.make_future_dataframe(periods=forecast_days, freq="B")
    forecast = model.predict(future)
    future_only = forecast[forecast["ds"] > df["ds"].max()]
    return future_only[["ds", "yhat", "yhat_lower", "yhat_upper"]]


def evaluate_model(df, test_days, interval_width):
    split_idx = len(df) - test_days
    train_eval = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()
    model = Prophet(
        seasonality_mode="multiplicative",
        changepoint_prior_scale=0.05,
        changepoint_range=0.9,
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        interval_width=interval_width,
    )
    model.fit(train_eval[["ds", "y"]])
    test_forecast = model.predict(test[["ds"]])
    merged = test[["ds", "y"]].merge(
        test_forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]], on="ds"
    )
    actual = merged["y"].values
    predicted = merged["yhat"].values
    mae = float(np.mean(np.abs(actual - predicted)))
    rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
    mape = float(np.mean(np.abs((actual - predicted) / actual)) * 100)
    coverage = float(((actual >= merged["yhat_lower"].values) & (actual <= merged["yhat_upper"].values)).mean())
    return {"mae": mae, "rmse": rmse, "mape": mape, "ci_coverage": coverage}


@app.get("/")
def root():
    return {"status": "ok", "message": "Stock Forecaster API is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/test-data")
def test_data(ticker: str = "AAPL"):
    """Debug endpoint — tests Tiingo directly"""
    url = f"https://api.tiingo.com/tiingo/daily/{ticker}/prices?startDate=2024-01-01&token={TIINGO_TOKEN}"
    response = requests.get(url, headers={"Content-Type": "application/json"}, timeout=30)
    if response.status_code != 200:
        return {"status": "error", "code": response.status_code, "detail": response.text[:200]}
    data = response.json()
    return {"status": "ok", "rows": len(data), "first": data[:1], "last": data[-1:] if data else []}


@app.get("/forecast", response_model=ForecastResponse)
def get_forecast(
    ticker: str = "AAPL",
    forecast_days: int = 30,
    test_days: int = 120,
    interval_width: float = 0.95,
):
    import traceback
    try:
        ticker = ticker.upper().strip()
        forecast_days = min(max(forecast_days, 1), 90)
        interval_width = min(max(interval_width, 0.5), 0.99)
        df = fetch_data(ticker)
        if len(df) < 365:
            raise HTTPException(status_code=400, detail=f"Not enough data for {ticker}.")
        forecast_df = train_and_forecast(df, forecast_days, interval_width)
        metrics = evaluate_model(df, test_days, interval_width)
        forecast_points = [
            ForecastPoint(
                date=row["ds"].strftime("%Y-%m-%d"),
                predicted=round(float(row["yhat"]), 2),
                lower=round(float(row["yhat_lower"]), 2),
                upper=round(float(row["yhat_upper"]), 2),
            )
            for _, row in forecast_df.iterrows()
        ]
        return ForecastResponse(
            ticker=ticker,
            forecast_days=forecast_days,
            last_actual_date=df["ds"].max().strftime("%Y-%m-%d"),
            last_actual_price=round(float(df["y"].iloc[-1]), 2),
            forecast=forecast_points,
            metrics=MetricsModel(**{k: round(v, 4) for k, v in metrics.items()}),
        )
    except HTTPException:
        raise
    except Exception as e:
        print("FULL ERROR:", traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tickers")
def suggested_tickers():
    return {
        "suggested": [
            {"ticker": "AAPL", "name": "Apple"},
            {"ticker": "MSFT", "name": "Microsoft"},
            {"ticker": "TSLA", "name": "Tesla"},
            {"ticker": "NVDA", "name": "Nvidia"},
            {"ticker": "AMZN", "name": "Amazon"},
            {"ticker": "GOOGL", "name": "Google"},
            {"ticker": "SPY", "name": "S&P 500 ETF"},
        ]
    }