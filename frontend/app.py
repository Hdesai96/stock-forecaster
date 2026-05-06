"""
Stock Forecaster — enhanced Streamlit app.
Improvements over v1:
  - SPY lagged return as Prophet regressor (captures market-wide moves)
  - Implied volatility from the options chain (market's own uncertainty estimate)
  - Earnings date markers on the chart
  - RSI · MACD · Bollinger Bands technical panel
  - EWMA realized volatility chart with IV overlay
  - Regime detection (bull / bear / sideways via MA crossover)
  - Direction probability (momentum-based P(up) signal)
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
from plotly.subplots import make_subplots
from prophet import Prophet
from scipy.stats import norm
import yfinance as yf

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
CACHE_DIR = os.getenv("CACHE_DIR", "/tmp/stock_cache")
CACHE_TTL = 23 * 3600
TIINGO_TOKEN = os.getenv("TIINGO_TOKEN", "")
os.makedirs(CACHE_DIR, exist_ok=True)


# ── Disk cache ────────────────────────────────────────────────────────────────
def _cache_path(key: str) -> str:
    return os.path.join(CACHE_DIR, f"{key}.json")


def _cache_get(key: str, allow_stale: bool = False):
    path = _cache_path(key)
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


def _cache_set(key: str, df: pd.DataFrame):
    path = _cache_path(key)
    try:
        payload = df.assign(ds=df["ds"].astype(str)).to_dict(orient="records")
        with open(path, "w") as f:
            json.dump({"ts": time.time(), "data": payload}, f)
    except Exception:
        pass


# ── Data fetching ─────────────────────────────────────────────────────────────
def _fetch_stooq(ticker: str) -> pd.DataFrame:
    """Direct Stooq CSV — no API key. Filters non-CSV lines to handle rate-limit messages."""
    from io import StringIO
    url = f"https://stooq.com/q/d/l/?s={ticker.lower()}.us&i=d"
    r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    if r.status_code != 200:
        raise ValueError(f"Stooq HTTP {r.status_code}")
    # Keep only lines that look like CSV rows (≥4 commas = OHLCV + date)
    lines = [ln for ln in r.text.splitlines() if ln.count(",") >= 4]
    if not lines:
        raise ValueError("Stooq returned no CSV rows (possible rate-limit or unknown ticker)")
    df = pd.read_csv(StringIO("\n".join(lines)))
    if "Close" not in df.columns:
        raise ValueError(f"Stooq: unexpected columns {df.columns.tolist()}")
    df = df.rename(columns={"Date": "ds", "Close": "y"})
    df["ds"] = pd.to_datetime(df["ds"]).dt.tz_localize(None)
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df[["ds", "y"]].dropna().sort_values("ds").reset_index(drop=True)
    cutoff = pd.Timestamp(date.today() - timedelta(days=365 * 5))
    return df[df["ds"] >= cutoff].reset_index(drop=True)


def _fetch_yfinance(ticker: str) -> pd.DataFrame:
    """Try Ticker.history() first (different endpoint), fall back to yf.download()."""
    cutoff = (date.today() - timedelta(days=365 * 5)).isoformat()

    # Attempt 1: Ticker.history() — uses a different Yahoo endpoint
    try:
        t = yf.Ticker(ticker)
        raw = t.history(period="5y", auto_adjust=True, actions=False)
        if raw is not None and not raw.empty:
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            df = raw[["Close"]].rename(columns={"Close": "y"}).copy()
            df.index.name = "ds"
            df = df.reset_index()
            df["ds"] = pd.to_datetime(df["ds"]).dt.tz_localize(None)
            df["y"] = df["y"].astype(float)
            result = df.dropna(subset=["y"])
            if len(result) >= 365:
                return result
    except Exception:
        pass

    # Attempt 2: yf.download()
    raw = yf.download(ticker, start=cutoff, progress=False, auto_adjust=True)
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
    for _source, fn in [("Stooq", _fetch_stooq), ("yfinance", _fetch_yfinance), ("Tiingo", _fetch_tiingo)]:
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

    raise RuntimeError(
        f"All data sources failed for {ticker}. Last error: {last_err}. "
        "Tip: set a TIINGO_TOKEN environment variable in Render for a reliable fallback."
    )


# ── Technical indicators ───────────────────────────────────────────────────────
def compute_technicals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("ds").reset_index(drop=True)

    # RSI-14
    delta = df["y"].diff()
    avg_gain = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
    avg_loss = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    # Bollinger Bands (20-day, 2σ)
    df["bb_mid"] = df["y"].rolling(20).mean()
    bb_std = df["y"].rolling(20).std()
    df["bb_upper"] = df["bb_mid"] + 2 * bb_std
    df["bb_lower"] = df["bb_mid"] - 2 * bb_std
    df["bb_pct"] = (df["y"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

    # MACD (12 / 26 / 9)
    ema12 = df["y"].ewm(span=12, adjust=False).mean()
    ema26 = df["y"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # EWMA realized volatility — annualised %
    log_ret = np.log(df["y"]).diff()
    df["ewma_vol"] = log_ret.ewm(span=21, adjust=False).std() * np.sqrt(252) * 100

    # Moving averages for regime detection
    df["ma50"] = df["y"].rolling(50).mean()
    df["ma200"] = df["y"].rolling(200).mean()

    return df


def detect_regime(df: pd.DataFrame) -> str:
    recent = df.dropna(subset=["ma50", "ma200"]).tail(1)
    if recent.empty:
        return "unknown"
    ma50 = float(recent["ma50"].iloc[0])
    ma200 = float(recent["ma200"].iloc[0])
    price = float(recent["y"].iloc[0])
    if ma50 > ma200 * 1.02 and price > ma50:
        return "bull"
    if ma50 < ma200 * 0.98 and price < ma50:
        return "bear"
    return "sideways"


# ── Options implied volatility ─────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_implied_vol(ticker: str):
    try:
        t = yf.Ticker(ticker)
        exps = t.options
        if not exps:
            return None
        chain = t.option_chain(exps[0])
        calls = chain.calls.copy()
        if calls.empty:
            return None
        # Get spot price
        hist = t.history(period="1d")
        if hist.empty:
            return None
        spot = float(hist["Close"].iloc[-1])
        calls["moneyness"] = (calls["strike"] - spot).abs()
        atm = calls.nsmallest(1, "moneyness")
        iv = float(atm["impliedVolatility"].iloc[0])
        return round(iv * 100, 2)      # return as %
    except Exception:
        return None


# ── Earnings calendar ──────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_earnings_dates(ticker: str):
    try:
        t = yf.Ticker(ticker)
        edf = t.earnings_dates
        if edf is None or edf.empty:
            return []
        idx = pd.DatetimeIndex(edf.index)
        if idx.tz is not None:
            idx = idx.tz_convert(None)
        today = pd.Timestamp.today()
        mask = (idx >= today - pd.Timedelta(days=60)) & (idx <= today + pd.Timedelta(days=90))
        return idx[mask].strftime("%Y-%m-%d").tolist()
    except Exception:
        return []


# ── Direction probability ──────────────────────────────────────────────────────
def direction_probability(df: pd.DataFrame) -> dict:
    """Momentum + RSI heuristic → P(price higher in 5 trading days)."""
    df = df.copy().sort_values("ds").reset_index(drop=True)
    log_ret = np.log(df["y"]).diff().dropna()
    ret_5d  = float(log_ret.tail(5).sum())
    ret_20d = float(log_ret.tail(20).sum())
    vol_21d = float(log_ret.tail(21).std()) or 1e-8
    rsi = float(df["rsi"].dropna().iloc[-1]) if "rsi" in df.columns and df["rsi"].notna().any() else 50.0

    momentum = np.tanh(ret_5d / vol_21d) * 0.5 + np.tanh(ret_20d / (vol_21d * 2)) * 0.3
    rsi_adj  = (rsi - 50) / 100 * 0.2
    raw      = momentum + rsi_adj
    p_up     = float(min(max(1 / (1 + np.exp(-raw * 3)), 0.1), 0.9))

    return {
        "p_up": p_up,
        "p_down": 1 - p_up,
        "ret_5d_pct": ret_5d * 100,
        "ret_20d_pct": ret_20d * 100,
        "rsi": rsi,
        "vol_ann_pct": vol_21d * np.sqrt(252) * 100,
    }


# ── Forecasting models ─────────────────────────────────────────────────────────
def prophet_on_returns(
    df: pd.DataFrame,
    forecast_days: int,
    interval_width: float,
    spy_df: pd.DataFrame = None,
):
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

    use_spy = spy_df is not None and len(spy_df) > 10
    if use_spy:
        # Yesterday's SPY log-return → today's stock return (1-day lag)
        spy_clean = spy_df.drop_duplicates(subset=["ds"]).set_index("ds")["y"]
        spy_log_ret = np.log(spy_clean).diff().shift(1)
        spy_log_ret.name = "spy_lag1"
        df_ret = df_ret.drop_duplicates(subset=["ds"])
        df_ret = df_ret.join(spy_log_ret, on="ds", how="left")
        df_ret["spy_lag1"] = df_ret["spy_lag1"].fillna(0.0)
        model.add_regressor("spy_lag1", standardize=True)

    model.fit(df_ret)

    future = model.make_future_dataframe(periods=forecast_days, freq="B")

    if use_spy:
        spy_map = df_ret.set_index("ds")["spy_lag1"].to_dict()
        future["spy_lag1"] = future["ds"].map(spy_map).fillna(0.0)

    fc = model.predict(future)
    fc = fc[fc["ds"] > df["ds"].max()].copy().reset_index(drop=True)

    last_log_price = float(np.log(df["y"].iloc[-1]))
    fc["yhat"]       = np.exp(last_log_price + fc["yhat"].cumsum())
    fc["yhat_lower"] = np.exp(last_log_price + fc["yhat_lower"].cumsum())
    fc["yhat_upper"] = np.exp(last_log_price + fc["yhat_upper"].cumsum())
    return fc[["ds", "yhat", "yhat_lower", "yhat_upper"]]


def naive_forecast(df: pd.DataFrame, forecast_days: int, interval_width: float):
    df = df.copy().sort_values("ds").reset_index(drop=True)
    last_price = float(df["y"].iloc[-1])
    last_date  = df["ds"].max()
    sigma = float(np.log(df["y"]).diff().dropna().std())
    z = float(norm.ppf(0.5 + interval_width / 2))

    rows = []
    for h in range(1, forecast_days + 1):
        next_date = last_date + pd.tseries.offsets.BDay(h)
        band = z * sigma * np.sqrt(h)
        rows.append({
            "ds": next_date, "yhat": last_price,
            "yhat_lower": last_price * np.exp(-band),
            "yhat_upper": last_price * np.exp(band),
        })
    return pd.DataFrame(rows)


def _metrics(actual, predicted, lower, upper):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return {
        "mae":         float(np.mean(np.abs(actual - predicted))),
        "rmse":        float(np.sqrt(np.mean((actual - predicted) ** 2))),
        "mape":        float(np.mean(np.abs((actual - predicted) / actual)) * 100),
        "ci_coverage": float(((actual >= lower) & (actual <= upper)).mean()),
    }


def evaluate_models(df, test_days, interval_width, spy_df=None):
    test_days = min(test_days, max(30, len(df) // 5))
    train = df.iloc[: len(df) - test_days].copy().reset_index(drop=True)
    test  = df.iloc[len(df) - test_days :].copy().reset_index(drop=True)

    try:
        fc = prophet_on_returns(train, len(test), interval_width, spy_df=spy_df)
        n = min(len(fc), len(test))
        prophet_m = _metrics(
            test["y"].values[:n], fc["yhat"].values[:n],
            fc["yhat_lower"].values[:n], fc["yhat_upper"].values[:n],
        )
    except Exception:
        prophet_m = {k: np.nan for k in ("mae", "rmse", "mape", "ci_coverage")}

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

    spy_df = None
    if ticker != "SPY":
        try:
            spy_df = fetch_data("SPY")
        except Exception:
            pass

    df_tech    = compute_technicals(df)
    prophet_fc = prophet_on_returns(df, forecast_days, interval_width, spy_df=spy_df)
    naive_fc   = naive_forecast(df, forecast_days, interval_width)
    prophet_m, naive_m = evaluate_models(df, 60, interval_width, spy_df=spy_df)
    return df_tech, prophet_fc, naive_fc, prophet_m, naive_m


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Stock Forecaster", page_icon="📈", layout="wide")

st.sidebar.title("📈 Stock Forecaster")
st.sidebar.markdown(
    "Prophet + SPY regressor · implied volatility · technicals · regime detection"
)

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

forecast_days  = st.sidebar.slider("Forecast horizon (business days)", 5, 30, 20)
interval_width = st.sidebar.slider("Confidence interval", 0.50, 0.99, 0.95, 0.01)
run = st.sidebar.button("Run forecast", type="primary", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.caption(
    "⚠️ Stock prices are nearly random walks. "
    "This is a learning project, not investment advice."
)

st.title("Stock Price Forecaster")
st.markdown(
    "**Prophet** with SPY market regressor · implied volatility · "
    "earnings markers · technical indicators · regime detection"
)

if not run:
    st.info("👈 Pick a ticker and click **Run forecast** to start.")
    st.stop()

# ── Run forecast ──────────────────────────────────────────────────────────────
try:
    with st.spinner(f"Fetching data and training Prophet for {ticker}..."):
        df, prophet_fc, naive_fc, prophet_m, naive_m = run_full_forecast(
            ticker, forecast_days, interval_width
        )
except Exception as e:
    st.error(f"Could not forecast {ticker}: {e}")
    st.stop()

with st.spinner("Fetching options IV and earnings dates..."):
    iv       = fetch_implied_vol(ticker)
    earnings = fetch_earnings_dates(ticker)

regime   = detect_regime(df)
dir_prob = direction_probability(df)

last_price     = float(df["y"].iloc[-1])
last_date      = df["ds"].max().strftime("%Y-%m-%d")
last_pred      = prophet_fc.iloc[-1]
delta_pct      = (last_pred["yhat"] - last_price) / last_price * 100
current_rsi    = float(df["rsi"].dropna().iloc[-1]) if df["rsi"].notna().any() else float("nan")
current_ewma   = float(df["ewma_vol"].dropna().iloc[-1]) if df["ewma_vol"].notna().any() else float("nan")

# ── Metrics row ───────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Ticker", ticker)
c2.metric("Last Price", f"${last_price:.2f}")
c2.caption(f"as of {last_date}")
c3.metric(f"Forecast +{forecast_days}d", f"${last_pred['yhat']:.2f}", f"{delta_pct:+.2f}%")
c3.caption(last_pred["ds"].strftime("%Y-%m-%d"))
c4.metric("Realized Vol", f"{current_ewma:.1f}%" if not np.isnan(current_ewma) else "N/A")
c4.caption("EWMA 21d, annualised")
c5.metric("Implied Vol (ATM)", f"{iv:.1f}%" if iv else "N/A")
c5.caption("Nearest expiry, ATM call")
REGIME_LABEL = {"bull": "🟢 Bull", "bear": "🔴 Bear", "sideways": "🟡 Sideways", "unknown": "⚪ Unknown"}
c6.metric("Regime", REGIME_LABEL.get(regime, "⚪"))
c6.caption("50d vs 200d MA")

# ── Price + forecast chart ────────────────────────────────────────────────────
hist = df.tail(180)
fig  = go.Figure()

# Bollinger Band fill
fig.add_trace(go.Scatter(
    x=hist["ds"], y=hist["bb_upper"],
    mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip",
))
fig.add_trace(go.Scatter(
    x=hist["ds"], y=hist["bb_lower"],
    mode="lines", line=dict(width=0),
    fill="tonexty", fillcolor="rgba(100,149,237,0.10)",
    name="Bollinger Bands (20d)", hoverinfo="skip",
))

# Moving averages
fig.add_trace(go.Scatter(
    x=hist["ds"], y=hist["ma50"],
    mode="lines", line=dict(color="rgba(255,165,0,0.7)", width=1.2, dash="dot"),
    name="MA 50",
))
fig.add_trace(go.Scatter(
    x=hist["ds"], y=hist["ma200"],
    mode="lines", line=dict(color="rgba(148,0,211,0.7)", width=1.2, dash="dot"),
    name="MA 200",
))

# Historical price
fig.add_trace(go.Scatter(
    x=hist["ds"], y=hist["y"],
    name="Historical price", mode="lines",
    line=dict(color="#1f77b4", width=2),
))

# Forecast CI
fig.add_trace(go.Scatter(
    x=prophet_fc["ds"], y=prophet_fc["yhat_upper"],
    mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip",
))
fig.add_trace(go.Scatter(
    x=prophet_fc["ds"], y=prophet_fc["yhat_lower"],
    mode="lines", line=dict(width=0),
    fill="tonexty", fillcolor="rgba(255,127,14,0.18)",
    name=f"{int(interval_width*100)}% CI", hoverinfo="skip",
))
fig.add_trace(go.Scatter(
    x=prophet_fc["ds"], y=prophet_fc["yhat"],
    name="Prophet + SPY forecast", mode="lines",
    line=dict(color="#ff7f0e", width=2),
))
fig.add_trace(go.Scatter(
    x=naive_fc["ds"], y=naive_fc["yhat"],
    name="Naive (last price)", mode="lines",
    line=dict(color="#888", width=1.5, dash="dash"),
))

# Earnings markers
for ed in earnings:
    fig.add_vline(
        x=ed, line_width=1.5, line_dash="dot", line_color="crimson",
        annotation_text="📅 Earnings", annotation_position="top right",
    )

fig.update_layout(
    title=f"{ticker} — last 180 days + {forecast_days}d forecast",
    xaxis_title="Date", yaxis_title="Price (USD)",
    hovermode="x unified", height=520,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=20, r=20, t=60, b=20),
)
st.plotly_chart(fig, use_container_width=True)

# ── Direction signal + IV commentary ─────────────────────────────────────────
col_a, col_b = st.columns([1, 2])

with col_a:
    st.subheader("5-day direction signal")
    p_up  = dir_prob["p_up"]
    arrow = "📈" if p_up > 0.55 else ("📉" if p_up < 0.45 else "➡️")
    st.metric(f"{arrow} P(up next 5d)", f"{p_up * 100:.0f}%")
    st.progress(p_up)
    st.caption(
        f"5d ret: {dir_prob['ret_5d_pct']:+.2f}%  ·  "
        f"20d ret: {dir_prob['ret_20d_pct']:+.2f}%  ·  "
        f"RSI: {dir_prob['rsi']:.0f}"
    )
    st.caption("Momentum heuristic only — not a trading signal.")

with col_b:
    if earnings:
        st.warning(
            f"📅 **Earnings dates detected:** {', '.join(earnings)}. "
            "Prophet does not model these jumps — treat forecasts near these dates with extra caution."
        )
    if iv and not np.isnan(current_ewma) and current_ewma > 0:
        ratio = iv / current_ewma
        direction = (
            "higher than" if ratio > 1.15 else
            "lower than" if ratio < 0.85 else
            "in line with"
        )
        st.info(
            f"**Options market IV: {iv:.1f}%** vs realized vol {current_ewma:.1f}% "
            f"— the market is pricing in *{direction}* recent realised volatility "
            f"(IV/RV = {ratio:.2f}×). "
            "The model CI uses realised vol; the true market uncertainty is closer to IV."
        )

# ── Technical indicators ───────────────────────────────────────────────────────
with st.expander("📊 Technical Indicators — RSI · MACD · Bollinger %B", expanded=False):
    th = df.tail(180)
    fig_t = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07,
        subplot_titles=("RSI (14)", "MACD (12 / 26 / 9)", "Bollinger %B"),
    )

    # RSI
    fig_t.add_trace(go.Scatter(x=th["ds"], y=th["rsi"], mode="lines",
                                name="RSI", line=dict(color="#2196F3", width=1.5)), row=1, col=1)
    fig_t.add_hline(y=70, line_dash="dash", line_color="red",   row=1, col=1)
    fig_t.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=1)
    fig_t.update_yaxes(range=[0, 100], row=1, col=1)

    # MACD histogram + lines
    bar_colors = ["#26a69a" if v >= 0 else "#ef5350" for v in th["macd_hist"]]
    fig_t.add_trace(go.Bar(x=th["ds"], y=th["macd_hist"],
                            name="Histogram", marker_color=bar_colors, opacity=0.7), row=2, col=1)
    fig_t.add_trace(go.Scatter(x=th["ds"], y=th["macd"], mode="lines",
                                name="MACD", line=dict(color="#FF9800", width=1.5)), row=2, col=1)
    fig_t.add_trace(go.Scatter(x=th["ds"], y=th["macd_signal"], mode="lines",
                                name="Signal", line=dict(color="#9C27B0", width=1.5, dash="dash")), row=2, col=1)

    # Bollinger %B
    fig_t.add_trace(go.Scatter(x=th["ds"], y=th["bb_pct"], mode="lines",
                                name="BB %B", line=dict(color="#00BCD4", width=1.5),
                                fill="tozeroy", fillcolor="rgba(0,188,212,0.08)"), row=3, col=1)
    fig_t.add_hline(y=1, line_dash="dash", line_color="red",   row=3, col=1)
    fig_t.add_hline(y=0, line_dash="dash", line_color="green", row=3, col=1)

    fig_t.update_layout(height=600, showlegend=False,
                         hovermode="x unified", margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_t, use_container_width=True)

# ── Volatility chart ───────────────────────────────────────────────────────────
with st.expander("📉 Realized vs Implied Volatility", expanded=False):
    fig_v = go.Figure()
    fig_v.add_trace(go.Scatter(
        x=df["ds"], y=df["ewma_vol"], mode="lines",
        name="Realized vol (EWMA 21d)", line=dict(color="#E91E63", width=2),
        fill="tozeroy", fillcolor="rgba(233,30,99,0.08)",
    ))
    if iv:
        fig_v.add_hline(
            y=iv, line_dash="dash", line_color="navy",
            annotation_text=f"ATM IV: {iv:.1f}%",
            annotation_position="top right",
        )
    fig_v.update_layout(
        title="Annualised Volatility — realized (EWMA) vs options-implied",
        xaxis_title="Date", yaxis_title="Volatility (%)",
        height=380, hovermode="x unified",
        margin=dict(l=20, r=20, t=50, b=20),
    )
    st.plotly_chart(fig_v, use_container_width=True)
    st.caption(
        "Realized vol = 21-day EWMA of daily log-returns × √252. "
        "Implied vol = nearest-expiry ATM call option (from yfinance). "
        "When IV > RV the options market expects more turbulence ahead."
    )

# ── Accuracy table ────────────────────────────────────────────────────────────
st.subheader("How accurate is this, really?")
st.markdown(
    "Both models were back-tested on a held-out tail of the data "
    f"(~60 trading days). CI coverage should be near {int(interval_width*100)}%."
)

eval_df = pd.DataFrame({
    "Metric": ["MAE ($)", "RMSE ($)", "MAPE (%)", "CI coverage"],
    "Prophet + SPY regressor": [
        f"{prophet_m['mae']:.2f}", f"{prophet_m['rmse']:.2f}",
        f"{prophet_m['mape']:.2f}%", f"{prophet_m['ci_coverage']*100:.1f}%",
    ],
    "Naive (last price)": [
        f"{naive_m['mae']:.2f}", f"{naive_m['rmse']:.2f}",
        f"{naive_m['mape']:.2f}%", f"{naive_m['ci_coverage']*100:.1f}%",
    ],
})
st.dataframe(eval_df, hide_index=True, use_container_width=True)

beats = not np.isnan(prophet_m["mape"]) and prophet_m["mape"] < naive_m["mape"]
if beats:
    st.success(
        f"✅ Prophet + SPY beat the naive baseline "
        f"(MAPE {prophet_m['mape']:.2f}% vs {naive_m['mape']:.2f}%)."
    )
else:
    st.warning(
        f"⚠️ Prophet did **not** beat the naive baseline "
        f"(MAPE {prophet_m['mape']:.2f}% vs {naive_m['mape']:.2f}%). "
        "This is normal — daily prices are close to a random walk."
    )

with st.expander("Why the naive comparison matters"):
    st.markdown(
        "- **Stock prices are roughly a random walk with drift.** The best estimate "
        "of tomorrow's price is usually today's price.\n"
        "- A model that can't beat that baseline is adding noise, not signal.\n"
        "- The honest use of this tool is reading the **confidence interval width** "
        "and **implied vs realized volatility** — not the point forecast.\n"
        "- For real edge you'd need alternative data (earnings transcripts, "
        "options flow, insider filings) and shorter horizons (1–3 days)."
    )

with st.expander("📋 Forecast table"):
    show = prophet_fc.copy()
    show["ds"]          = show["ds"].dt.strftime("%Y-%m-%d")
    show["yhat"]        = show["yhat"].round(2)
    show["yhat_lower"]  = show["yhat_lower"].round(2)
    show["yhat_upper"]  = show["yhat_upper"].round(2)
    show = show.rename(columns={
        "ds": "Date", "yhat": "Predicted",
        "yhat_lower": "Lower", "yhat_upper": "Upper",
    })
    st.dataframe(show, hide_index=True, use_container_width=True)
