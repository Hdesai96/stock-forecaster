from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import requests
from prophet import Prophet
from datetime import date
import warnings
warnings.filterwarnings("ignore")

app = FastAPI(title="Stock Forecaster API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = "CX0LKRLDG6QO5UJ0"

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
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&outputsize=full&apikey={API_KEY}"
    
    print(f"Fetching data for {ticker} from Alpha Vantage...")
    response = requests.get(url, timeout=30)
    data = response.json()
    
    print(f"Response keys: {list(data.keys())}")
    
    if "Time Series (Daily)" not in data:
        error_msg = data.get("Note") or data.get("Information") or data.get("Error Message") or "Unknown error"
        print(f"Error from Alpha Vantage: {error_msg}")
        raise HTTPException(status_code=404, detail=f"No data found for ticker '{ticker}': {error_msg}")
    
    ts = data["Time Series (Daily)"]
    df = pd.DataFrame.from_dict(ts, orient="index")
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df.rename(columns={"4. close": "y", "5. volume": "volume"})
    df["y"] = df["y"].astype(float)
    df["volume"] = df["volume"].astype(float)
    df.index.name = "ds"
    df = df.reset_index()
    df["ds"] = pd.to_datetime(df["ds"]).dt.tz_localize(None)
    df = df[["ds", "y", "volume"]]
    df = df.dropna(subset=["y"])
    
    print(f"Fetched {len(df)} rows for {ticker}, last date: {df['ds'].max()}")
    return df


def train_and_forecast(df, forecast_days, interval_width):
    model = Prophet(
        seasonality_mode="multiplicative",
        changepoint_prior_scale=0.3,
        yearly_seasonality=True,
        weekly_seasonality=True,
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
        changepoint_prior_scale=0.3,
        yearly_seasonality=True,
        weekly_seasonality=True,
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
    """Debug endpoint — tests Alpha Vantage directly"""
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&outputsize=compact&apikey={API_KEY}"
    response = requests.get(url, timeout=30)
    data = response.json()
    keys = list(data.keys())
    first_key = list(data.get("Time Series (Daily)", {}).keys())[:2] if "Time Series (Daily)" in data else []
    return {"response_keys": keys, "first_dates": first_key, "status": "ok" if "Time Series (Daily)" in data else "error"}


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