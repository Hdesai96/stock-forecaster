"""
Stock Forecaster — single-file Streamlit app.
All forecasting logic runs inside Streamlit — no separate FastAPI backend needed.

Deploy on Render with:
  Build:  pip install -r requirements.txt
  Start:  streamlit run app.py --server.port $PORT --server.address 0.0.0.0
"""

import os
import time
import json
import warnings
from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
from prophet import Prophet
from scipy.stats import norm

import pandas_datareader.data as web
import yfinance as yf

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
CACHE_DIR = os.getenv("CACHE_DIR", "/tmp/stock_cache")
CACHE_TTL = 23 * 3600
TIINGO_TOKEN = os.getenv("TIINGO_TOKEN", "")
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
            return pd.DataFrame(obj["data"])
    except Exception:
        pass
    return None


def _cache_set(ticker: str, df: pd.DataFrame):
    path = _cache_path(ticker)
    try:
        payload = df.assign(ds=df["ds"].astype(str)).to_dict(orient="records")
        with open(path, "w") as f:
            json.dump({"ts": time.time(), "data": payload}, f)
    except Exception:
        pass


# ── Data sources ──────────────────────────────────────────────────────────────
def _fetch_stooq(ticker: str) -> pd.DataFrame:
    start = date.today() - timedelta(days=365 * 5)
    raw = web.DataReader(f"{ticker}.US", "stooq", start=start)
    if raw.empty:
        raise ValueError("No data from Stooq")
    raw = raw.sort_index()
    df = raw[["Close"]].rename(columns={"Close": "y"}).copy()
    df.index.name = "ds"
    df = df.reset_index()
    df["ds"] = pd.to_datetime(df["ds"]).dt.tz_localize(None)
    df["y"] = df["y"].astype(float)
    return df.dropna(subset=["y"])


def _fetch_yfinance(ticker: str) -> pd.DataFrame:
    start = (date.today() - timedelta(days=365 * 5)).isoformat()
    raw = yf.download(ticker, start=start, progress=False, auto_adjust=True)
    if raw is None or raw.empty:
        raise ValueError("No data from yfinance")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw[["Close"]].rename(columns={"Close": "y"}).copy()
    df.index.name = "ds"
    df = df.reset_index()
    df["ds"] = pd.to_datetime(df["ds"]).dt.tz_localize(None)
    df["y"] = df["y"].astype(float)
    return df.dropna(subset=["y"])


def _fetch_tiingo(ticker: str) -> pd.DataFrame:
    if not TIINGO_TOKEN:
        raise ValueError("TIINGO_TOKEN not set")
    start_date = (date.today() - timedelta(days=365 * 5)).isoformat()
    url = (
        f"https://api.tiingo.com/tiingo/daily/{ticker}/prices"
        f"?startDate={start_date}&token={TIINGO_TOKEN}"
    )
    r = requests.get(url, headers={"Content-Type": "application/json"}, timeout=30)
    if r.status_code != 200:
        raise ValueError(f"Tiingo {r.status_code}")
    data = r.json()
    if not data:
        raise ValueError("Empty Tiingo response")
    df = pd.DataFrame(data)
    df["ds"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = df.rename(columns={"adjClose": "y"})
    df["y"] = df["y"].astype(float)
    df = df.sort_values("ds").reset_index(drop=True)
    return df[["ds", "y"]].dropna(subset=["y"])


def fetch_data(ticker: str) -> pd.DataFrame:
    cached = _cache_get(ticker)
    if cached is not None:
        cached["ds"] = pd.to_datetime(cached["ds"])
        return cached.copy()

    df = None
    last_err = None
    for source, fn in [
        ("Stooq", _fetch_stooq),
        ("yfinance", _fetch_yfinance),
        ("Tiingo", _fetch_tiingo),
    ]:
        try:
            df = fn(ticker)
            if df is not None and len(df) >= 365:
                break
            df = None
        except Exception as e:
            last_err = e
            continue

    if df is not None and len(df) >= 365:
        _cache_set(ticker, df)
        return df.copy()

    stale = _cache_get(ticker, allow_stale=True)
    if stale is not None and len(stale) >= 365:
        stale["ds"] = pd.to_datetime(stale["ds"])
        return stale.copy()

    raise RuntimeError(f"All data sources failed for {ticker}. Last error: {last_err}")


# ── Models ────────────────────────────────────────────────────────────────────
def prophet_on_returns(df: pd.DataFrame, forecast_days: int, interval_width: float):
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

    last_log_price = float(np.log(df["y"].iloc[-1]))
    fc["yhat"] = np.exp(last_log_price + fc["yhat"].cumsum())
    fc["yhat_lower"] = np.exp(last_log_price + fc["yhat_lower"].cumsum())
    fc["yhat_upper"] = np.exp(last_log_price + fc["yhat_upper"].cumsum())
    return fc[["ds", "yhat", "yhat_lower", "yhat_upper"]]


def naive_forecast(df: pd.DataFrame, forecast_days: int, interval_width: float):
    df = df.copy().sort_values("ds").reset_index(drop=True)
    last_price = float(df["y"].iloc[-1])
    last_date = df["ds"].max()
    log_ret = np.log(df["y"]).diff().dropna()
    sigma = float(log_ret.std())
    z = float(norm.ppf(0.5 + interval_width / 2))

    rows = []
    for h in range(1, forecast_days + 1):
        next_date = last_date + pd.tseries.offsets.BDay(h)
        band = z * sigma * np.sqrt(h)
        rows.append({
            "ds": next_date,
            "yhat": last_price,
            "yhat_lower": last_price * np.exp(-band),
            "yhat_upper": last_price * np.exp(band),
        })
    return pd.DataFrame(rows)


def _metrics(actual, predicted, lower, upper):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return {
        "mae": float(np.mean(np.abs(actual - predicted))),
        "rmse": float(np.sqrt(np.mean((actual - predicted) ** 2))),
        "mape": float(np.mean(np.abs((actual - predicted) / actual)) * 100),
        "ci_coverage": float(((actual >= lower) & (actual <= upper)).mean()),
    }


def evaluate_models(df: pd.DataFrame, test_days: int, interval_width: float):
    test_days = min(test_days, max(30, len(df) // 5))
    split_idx = len(df) - test_days
    train = df.iloc[:split_idx].copy().reset_index(drop=True)
    test = df.iloc[split_idx:].copy().reset_index(drop=True)

    try:
        fc = prophet_on_returns(train, len(test), interval_width)
        n = min(len(fc), len(test))
        prophet_m = _metrics(
            test["y"].values[:n], fc["yhat"].values[:n],
            fc["yhat_lower"].values[:n], fc["yhat_upper"].values[:n],
        )
    except Exception:
        prophet_m = {"mae": np.nan, "rmse": np.nan, "mape": np.nan, "ci_coverage": np.nan}

    nfc = naive_forecast(train, len(test), interval_width)
    n = min(len(nfc), len(test))
    naive_m = _metrics(
        test["y"].values[:n], nfc["yhat"].values[:n],
        nfc["yhat_lower"].values[:n], nfc["yhat_upper"].values[:n],
    )
    return prophet_m, naive_m


@st.cache_data(ttl=3600, show_spinner=False)
def run_full_forecast(ticker: str, forecast_days: int, interval_width: float):
    df = fetch_data(ticker)
    if len(df) < 365:
        raise ValueError(f"Not enough data for {ticker}.")
    prophet_fc = prophet_on_returns(df, forecast_days, interval_width)
    naive_fc = naive_forecast(df, forecast_days, interval_width)
    prophet_m, naive_m = evaluate_models(df, 60, interval_width)
    return df, prophet_fc, naive_fc, prophet_m, naive_m


# ── UI ────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Stock Forecaster", page_icon="📈", layout="wide")

st.sidebar.title("📈 Stock Forecaster")
st.sidebar.markdown("Prophet-based daily forecast with naive baseline comparison.")

SUGGESTED = [
    ("AAPL", "Apple"), ("MSFT", "Microsoft"), ("TSLA", "Tesla"),
    ("NVDA", "Nvidia"), ("AMZN", "Amazon"), ("GOOGL", "Google"),
    ("SPY", "S&P 500 ETF"),
]

choice = st.sidebar.selectbox(
    "Pick a ticker",
    options=[t for t, _ in SUGGESTED],
    format_func=lambda t: f"{t} — {dict(SUGGESTED)[t]}",
)
custom = st.sidebar.text_input("...or enter another ticker", value="").strip().upper()
ticker = custom if custom else choice

forecast_days = st.sidebar.slider("Forecast horizon (business days)", 5, 30, 20)
interval_width = st.sidebar.slider("Confidence interval", 0.50, 0.99, 0.95, 0.01)
run = st.sidebar.button("Run forecast", type="primary", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.caption(
    "⚠️ Stock prices are nearly random walks. "
    "This is a learning project, not investment advice."
)

st.title("Stock Price Forecaster")
st.markdown(
    "Built with **Prophet**, with Stooq/yfinance/Tiingo as data sources and a "
    "**naive baseline** (`tomorrow = today`) for honest accuracy comparison."
)

if not run:
    st.info("👈 Pick a ticker and click **Run forecast** to start.")
    st.stop()

try:
    with st.spinner(f"Fetching data and training Prophet for {ticker} (first run can take ~30s)..."):
        df, prophet_fc, naive_fc, prophet_m, naive_m = run_full_forecast(
            ticker, forecast_days, interval_width
        )
except Exception as e:
    st.error(f"Could not forecast {ticker}: {e}")
    st.stop()

last_price = float(df["y"].iloc[-1])
last_date = df["ds"].max().strftime("%Y-%m-%d")
last_pred = prophet_fc.iloc[-1]
delta_pct = (last_pred["yhat"] - last_price) / last_price * 100

c1, c2, c3, c4 = st.columns(4)
c1.metric("Ticker", ticker)
c2.metric("Last actual price", f"${last_price:.2f}")
c2.caption(f"as of {last_date}")
c3.metric(
    f"Forecast (+{forecast_days}d)",
    f"${last_pred['yhat']:.2f}",
    f"{delta_pct:+.2f}%",
)
c3.caption(last_pred["ds"].strftime("%Y-%m-%d"))
c4.metric(
    "Forecast range",
    f"${last_pred['yhat_lower']:.2f} – ${last_pred['yhat_upper']:.2f}",
)
c4.caption(f"{int(interval_width*100)}% confidence interval")

# Chart
hist = df.tail(180)
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=hist["ds"], y=hist["y"],
    name="Historical price", mode="lines",
    line=dict(color="#1f77b4", width=2),
))
fig.add_trace(go.Scatter(
    x=prophet_fc["ds"], y=prophet_fc["yhat_upper"],
    mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip",
))
fig.add_trace(go.Scatter(
    x=prophet_fc["ds"], y=prophet_fc["yhat_lower"],
    mode="lines", line=dict(width=0),
    fill="tonexty", fillcolor="rgba(255, 127, 14, 0.18)",
    name=f"{int(interval_width*100)}% CI", hoverinfo="skip",
))
fig.add_trace(go.Scatter(
    x=prophet_fc["ds"], y=prophet_fc["yhat"],
    name="Prophet forecast", mode="lines",
    line=dict(color="#ff7f0e", width=2),
))
fig.add_trace(go.Scatter(
    x=naive_fc["ds"], y=naive_fc["yhat"],
    name="Naive (last price)", mode="lines",
    line=dict(color="#888", width=1.5, dash="dash"),
))
fig.update_layout(
    title=f"{ticker} — last 180 days + {forecast_days}d forecast",
    xaxis_title="Date", yaxis_title="Price (USD)",
    hovermode="x unified", height=520,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=20, r=20, t=60, b=20),
)
st.plotly_chart(fig, use_container_width=True)

# Honest evaluation
st.subheader("How accurate is this, really?")
st.markdown(
    "Both models were back-tested on a held-out tail of the data. "
    f"CI coverage should be close to {int(interval_width*100)}%."
)

eval_df = pd.DataFrame({
    "Metric": ["MAE ($)", "RMSE ($)", "MAPE (%)", "CI coverage"],
    "Prophet": [
        f"{prophet_m['mae']:.2f}", f"{prophet_m['rmse']:.2f}",
        f"{prophet_m['mape']:.2f}%", f"{prophet_m['ci_coverage']*100:.1f}%",
    ],
    "Naive (last price)": [
        f"{naive_m['mae']:.2f}", f"{naive_m['rmse']:.2f}",
        f"{naive_m['mape']:.2f}%", f"{naive_m['ci_coverage']*100:.1f}%",
    ],
})
st.dataframe(eval_df, hide_index=True, use_container_width=True)

beats = (
    not np.isnan(prophet_m["mape"])
    and prophet_m["mape"] < naive_m["mape"]
)
if beats:
    st.success(
        f"✅ Prophet beat the naive baseline "
        f"(MAPE {prophet_m['mape']:.2f}% vs {naive_m['mape']:.2f}%)."
    )
else:
    st.warning(
        f"⚠️ Prophet did **not** beat the naive baseline "
        f"(MAPE {prophet_m['mape']:.2f}% vs {naive_m['mape']:.2f}%). "
        "This is normal — daily stock prices are close to a random walk."
    )

with st.expander("Why the naive comparison matters"):
    st.markdown(
        "- **Stock prices are roughly a random walk with drift.** The best estimate "
        "of tomorrow's price is usually today's price.\n"
        "- A model that can't beat that baseline is adding noise, not signal.\n"
        "- The honest takeaway from any retail-grade stock forecaster is the "
        "**confidence interval width**, not the point forecast.\n"
        "- For real alpha you'd need alternative data, regime models, or "
        "higher-frequency signals — not Prophet on daily closes."
    )

with st.expander("📋 Forecast table"):
    show = prophet_fc.copy()
    show["ds"] = show["ds"].dt.strftime("%Y-%m-%d")
    show["yhat"] = show["yhat"].round(2)
    show["yhat_lower"] = show["yhat_lower"].round(2)
    show["yhat_upper"] = show["yhat_upper"].round(2)
    show = show.rename(columns={
        "ds": "Date", "yhat": "Predicted",
        "yhat_lower": "Lower", "yhat_upper": "Upper",
    })
    st.dataframe(show, hide_index=True, use_container_width=True)