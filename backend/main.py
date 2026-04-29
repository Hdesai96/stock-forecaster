"""
Stock Forecaster API — FastAPI backend
Fixes applied:
  - Stooq is now first source (most reliable on cloud IPs)
  - Removed the buggy 60s rate-limit guard that locked users out after upstream failures
  - Added stale-cache fallback: serve expired cache rather than 502
  - Tiingo token moved to environment variable
  - Prophet now models log-returns, not raw prices (correct for stock data)
  - Naive baseline added to metrics so users see what the model actually adds
  - Honest cross-validation-style metric on a held-out tail
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import requests
from prophet import Prophet
from datetime import date, timedelta
import warnings
import time
import os
import json
import pandas_datareader.data as web
import yfinance as yf

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
# On Render free tier, /tmp wipes on every dyno restart.
# Add a Persistent Disk in Render dashboard (mount at /data) and set
# CACHE_DIR env var to "/data/stock_cache" to survive restarts.
CACHE_DIR = os.getenv("CACHE_DIR", "/tmp/stock_cache")
CACHE_TTL = 23 * 3600  # 23 hours
TIINGO_TOKEN = os.getenv("TIINGO_TOKEN", "")  # set in Render env vars
os.makedirs(CACHE_DIR, exist_ok=True)


# ── Disk cache ────────────────────────────────────────────────────────────────
def _cache_path(ticker: str) -> str:
    return os.path.join(CACHE_DIR, f"{ticker}.json")


def _cache_get(ticker: str, allow_stale: bool = False):
    path = _cache_path(ticker)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            obj = json.load(f)
        age = time.time() - obj["ts"]
        if age < CACHE_TTL or allow_stale:
            tag = "stale" if age >= CACHE_TTL else "fresh"
            print(f"Disk cache hit ({tag}, age={int(age)}s) for {ticker}")
            return pd.DataFrame(obj["data"])
    except Exception as e:
        print(f"Cache read failed for {ticker}: {e}")
    return None


def _cache_set(ticker: str, df: pd.DataFrame):
    path = _cache_path(ticker)
    try:
        payload = df.assign(ds=df["ds"].astype(str)).to_dict(orient="records")
        with open(path, "w") as f:
            json.dump({"ts": time.time(), "data": payload}, f)
        print(f"Disk cached {ticker} ({len(df)} rows)")
    except Exception as e:
        print(f"Cache write failed: {e}")


# ── Data sources ──────────────────────────────────────────────────────────────
def _fetch_stooq(ticker: str) -> pd.DataFrame:
    """Stooq — no API key, no IP blocks on cloud, EOD daily data. Most reliable."""
    start = date.today() - timedelta(days=365 * 5)
    raw = web.DataReader(f"{ticker}.US", "stooq", start=start)
    if raw.empty:
        raise ValueError("No data returned from Stooq")
    raw = raw.sort_index()
    df = raw[["Close"]].rename(columns={"Close": "y"}).copy()
    df.index.name = "ds"
    df = df.reset_index()
    df["ds"] = pd.to_datetime(df["ds"]).dt.tz_localize(None)
    df["y"] = df["y"].astype(float)
    df = df.dropna(subset=["y"])
    return df


def _fetch_yfinance(ticker: str) -> pd.DataFrame:
    """yfinance fallback — Yahoo blocks many cloud IPs, so this often fails on Render."""
    start = (date.today() - timedelta(days=365 * 5)).isoformat()
    raw = yf.download(ticker, start=start, progress=False, auto_adjust=True)
    if raw is None or raw.empty:
        raise ValueError("No data from yfinance")
    # Handle multi-index columns from newer yfinance
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw[["Close"]].rename(columns={"Close": "y"}).copy()
    df.index.name = "ds"
    df = df.reset_index()
    df["ds"] = pd.to_datetime(df["ds"]).dt.tz_localize(None)
    df["y"] = df["y"].astype(float)
    df = df.dropna(subset=["y"])
    return df


def _fetch_tiingo(ticker: str) -> pd.DataFrame:
    """Tiingo — requires API key, last resort because of free-tier quota limits."""
    if not TIINGO_TOKEN:
        raise ValueError("TIINGO_TOKEN not set")
    start_date = (date.today() - timedelta(days=365 * 5)).isoformat()
    url = (
        f"https://api.tiingo.com/tiingo/daily/{ticker}/prices"
        f"?startDate={start_date}&token={TIINGO_TOKEN}"
    )
    r = requests.get(url, headers={"Content-Type": "application/json"}, timeout=30)
    if r.status_code == 429:
        raise ValueError("Tiingo rate limited")
    if r.status_code == 404:
        raise ValueError(f"Ticker {ticker} not found on Tiingo")
    if r.status_code != 200:
        raise ValueError(f"Tiingo error {r.status_code}: {r.text[:200]}")
    data = r.json()
    if not data:
        raise ValueError(f"No data for {ticker}")
    df = pd.DataFrame(data)
    df["ds"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = df.rename(columns={"adjClose": "y"})
    df["y"] = df["y"].astype(float)
    df = df.sort_values("ds").reset_index(drop=True)
    return df[["ds", "y"]].dropna(subset=["y"])


def fetch_data(ticker: str) -> pd.DataFrame:
    # 1. Fresh cache
    cached = _cache_get(ticker)
    if cached is not None:
        cached["ds"] = pd.to_datetime(cached["ds"])
        return cached.copy()

    # 2. Try upstreams in reliability order
    df = None
    last_err = None
    for source, fn in [
        ("Stooq", _fetch_stooq),
        ("yfinance", _fetch_yfinance),
        ("Tiingo", _fetch_tiingo),
    ]:
        try:
            print(f"Fetching {ticker} from {source}...")
            df = fn(ticker)
            if df is not None and len(df) >= 365:
                print(f"{source}: {len(df)} rows for {ticker}")
                break
            df = None
        except Exception as e:
            last_err = e
            print(f"{source} failed for {ticker}: {e}")
            continue

    if df is not None and len(df) >= 365:
        _cache_set(ticker, df)
        return df.copy()

    # 3. Stale cache fallback — better than 502
    stale = _cache_get(ticker, allow_stale=True)
    if stale is not None and len(stale) >= 365:
        print(f"Serving stale cache for {ticker}")
        stale["ds"] = pd.to_datetime(stale["ds"])
        return stale.copy()

    raise HTTPException(
        status_code=502,
        detail=f"All data sources failed for {ticker}. Last error: {last_err}",
    )


# ── Models: Prophet on log-returns + naive baseline ───────────────────────────
def _prophet_on_returns(df: pd.DataFrame, forecast_days: int, interval_width: float):
    """Fit Prophet on log-returns (stationary), then cumsum back to prices."""
    df = df.copy().sort_values("ds").reset_index(drop=True)
    df["log_y"] = np.log(df["y"])
    df["ret"] = df["log_y"].diff()
    df_ret = df.dropna(subset=["ret"])[["ds", "ret"]].rename(columns={"ret": "y"})

    model = Prophet(
        seasonality_mode="additive",
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.01,
        interval_width=interval_width,
    )
    model.fit(df_ret)

    future = model.make_future_dataframe(periods=forecast_days, freq="B")
    fc = model.predict(future)
    fc = fc[fc["ds"] > df["ds"].max()].copy().reset_index(drop=True)

    # Cumsum log-returns back into log prices, then exponentiate
    last_log_price = float(np.log(df["y"].iloc[-1]))
    fc["yhat"] = np.exp(last_log_price + fc["yhat"].cumsum())
    fc["yhat_lower"] = np.exp(last_log_price + fc["yhat_lower"].cumsum())
    fc["yhat_upper"] = np.exp(last_log_price + fc["yhat_upper"].cumsum())
    return fc[["ds", "yhat", "yhat_lower", "yhat_upper"]]


def _naive_forecast(df: pd.DataFrame, forecast_days: int, interval_width: float):
    """Tomorrow = today. The honest baseline that most stock models fail to beat."""
    df = df.copy().sort_values("ds").reset_index(drop=True)
    last_price = float(df["y"].iloc[-1])
    last_date = df["ds"].max()

    # Estimate daily vol from log returns for confidence band
    log_ret = np.log(df["y"]).diff().dropna()
    sigma = float(log_ret.std())

    # Z-score for the requested CI (e.g. 0.95 -> 1.96)
    from scipy.stats import norm
    z = float(norm.ppf(0.5 + interval_width / 2))

    rows = []
    for h in range(1, forecast_days + 1):
        next_date = last_date + pd.tseries.offsets.BDay(h)
        # Random-walk uncertainty grows with sqrt(h)
        band = z * sigma * np.sqrt(h)
        rows.append({
            "ds": next_date,
            "yhat": last_price,
            "yhat_lower": last_price * np.exp(-band),
            "yhat_upper": last_price * np.exp(band),
        })
    return pd.DataFrame(rows)


# ── Evaluation: Prophet vs Naive on a held-out tail ───────────────────────────
def _metrics(actual: np.ndarray, predicted: np.ndarray, lower: np.ndarray, upper: np.ndarray):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mae = float(np.mean(np.abs(actual - predicted)))
    rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
    mape = float(np.mean(np.abs((actual - predicted) / actual)) * 100)
    coverage = float(((actual >= lower) & (actual <= upper)).mean())
    return {"mae": mae, "rmse": rmse, "mape": mape, "ci_coverage": coverage}


def evaluate_models(df: pd.DataFrame, test_days: int, interval_width: float):
    """Evaluate Prophet vs naive baseline on the last `test_days` business days."""
    test_days = min(test_days, max(30, len(df) // 5))  # cap at 20% of history
    split_idx = len(df) - test_days
    train = df.iloc[:split_idx].copy().reset_index(drop=True)
    test = df.iloc[split_idx:].copy().reset_index(drop=True)

    # Prophet eval
    try:
        fc = _prophet_on_returns(train, len(test), interval_width)
        # Align by position (both are business-day-spaced from the same start)
        n = min(len(fc), len(test))
        prophet_metrics = _metrics(
            test["y"].values[:n],
            fc["yhat"].values[:n],
            fc["yhat_lower"].values[:n],
            fc["yhat_upper"].values[:n],
        )
    except Exception as e:
        print(f"Prophet eval failed: {e}")
        prophet_metrics = {"mae": float("nan"), "rmse": float("nan"),
                           "mape": float("nan"), "ci_coverage": float("nan")}

    # Naive eval
    naive_fc = _naive_forecast(train, len(test), interval_width)
    n = min(len(naive_fc), len(test))
    naive_metrics = _metrics(
        test["y"].values[:n],
        naive_fc["yhat"].values[:n],
        naive_fc["yhat_lower"].values[:n],
        naive_fc["yhat_upper"].values[:n],
    )

    return prophet_metrics, naive_metrics


# ── API schemas ───────────────────────────────────────────────────────────────
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
    naive_forecast: list[ForecastPoint]
    metrics: MetricsModel
    naive_metrics: MetricsModel
    beats_naive: bool
    history: list[ForecastPoint]  # last 180 trading days for charting


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Stock Forecaster API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def prewarm_cache():
    """Pre-warm popular tickers in the background, with delays to be polite."""
    import threading

    def _warm():
        for ticker in ["AAPL", "MSFT", "TSLA", "NVDA", "AMZN", "GOOGL", "SPY"]:
            try:
                if _cache_get(ticker) is None:
                    print(f"Pre-warming cache for {ticker}...")
                    fetch_data(ticker)
                    time.sleep(3)
            except Exception as e:
                print(f"Pre-warm failed for {ticker}: {e}")

    threading.Thread(target=_warm, daemon=True).start()


@app.get("/")
def root():
    return {"status": "ok", "message": "Stock Forecaster API is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


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


@app.get("/forecast", response_model=ForecastResponse)
def get_forecast(
    ticker: str = "AAPL",
    forecast_days: int = 30,
    test_days: int = 60,
    interval_width: float = 0.95,
):
    import traceback
    try:
        ticker = ticker.upper().strip()
        # Cap at 30 — beyond this, intervals dominate and the forecast is meaningless
        forecast_days = min(max(forecast_days, 1), 30)
        test_days = min(max(test_days, 30), 180)
        interval_width = min(max(interval_width, 0.5), 0.99)

        df = fetch_data(ticker)
        if len(df) < 365:
            raise HTTPException(status_code=400, detail=f"Not enough data for {ticker}.")

        # Forecasts on full history
        prophet_fc = _prophet_on_returns(df, forecast_days, interval_width)
        naive_fc = _naive_forecast(df, forecast_days, interval_width)

        # Honest backtest metrics
        prophet_m, naive_m = evaluate_models(df, test_days, interval_width)
        beats_naive = (
            not np.isnan(prophet_m["mape"])
            and prophet_m["mape"] < naive_m["mape"]
        )

        def _to_points(fc_df):
            return [
                ForecastPoint(
                    date=row["ds"].strftime("%Y-%m-%d"),
                    predicted=round(float(row["yhat"]), 2),
                    lower=round(float(row["yhat_lower"]), 2),
                    upper=round(float(row["yhat_upper"]), 2),
                )
                for _, row in fc_df.iterrows()
            ]

        # Last 180 days of history for the chart
        hist_tail = df.tail(180).copy()
        history = [
            ForecastPoint(
                date=row["ds"].strftime("%Y-%m-%d"),
                predicted=round(float(row["y"]), 2),
                lower=round(float(row["y"]), 2),
                upper=round(float(row["y"]), 2),
            )
            for _, row in hist_tail.iterrows()
        ]

        return ForecastResponse(
            ticker=ticker,
            forecast_days=forecast_days,
            last_actual_date=df["ds"].max().strftime("%Y-%m-%d"),
            last_actual_price=round(float(df["y"].iloc[-1]), 2),
            forecast=_to_points(prophet_fc),
            naive_forecast=_to_points(naive_fc),
            metrics=MetricsModel(**{k: round(v, 4) for k, v in prophet_m.items()}),
            naive_metrics=MetricsModel(**{k: round(v, 4) for k, v in naive_m.items()}),
            beats_naive=bool(beats_naive),
            history=history,
        )
    except HTTPException:
        raise
    except Exception as e:
        print("FULL ERROR:", traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))