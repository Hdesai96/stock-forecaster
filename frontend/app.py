"""
Stock Forecaster — Streamlit frontend
Calls the FastAPI backend and shows Prophet forecast vs naive baseline,
with honest disclaimers about uncertainty.
"""

import os
import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────────
API_URL = os.getenv("API_URL", "https://stock-forecaster-pq9a.onrender.com")
# If your backend is at a different URL, set API_URL env var or edit above.
# When testing locally with `uvicorn main:app --reload`, use http://localhost:8000

st.set_page_config(
    page_title="Stock Forecaster",
    page_icon="📈",
    layout="wide",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("📈 Stock Forecaster")
st.sidebar.markdown("Prophet-based daily forecast with naive baseline comparison.")

SUGGESTED = [
    ("AAPL", "Apple"),
    ("MSFT", "Microsoft"),
    ("TSLA", "Tesla"),
    ("NVDA", "Nvidia"),
    ("AMZN", "Amazon"),
    ("GOOGL", "Google"),
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

# ── Caching the API call ──────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def get_forecast(ticker: str, forecast_days: int, interval_width: float):
    r = requests.get(
        f"{API_URL}/forecast",
        params={
            "ticker": ticker,
            "forecast_days": forecast_days,
            "interval_width": interval_width,
        },
        timeout=120,
    )
    r.raise_for_status()
    return r.json()


# ── Main panel ────────────────────────────────────────────────────────────────
st.title("Stock Price Forecaster")
st.markdown(
    "Built with **Prophet** (Facebook's time-series library), Stooq/yfinance/Tiingo "
    "as data sources, and a **naive baseline** (`tomorrow = today`) for honest comparison."
)

if not run:
    st.info("👈 Pick a ticker and click **Run forecast** to start.")
    st.stop()

# Fetch
try:
    with st.spinner(f"Fetching data and training Prophet for {ticker}..."):
        data = get_forecast(ticker, forecast_days, interval_width)
except requests.HTTPError as e:
    code = e.response.status_code
    if code == 429:
        st.error("Too many requests — the data provider is rate limiting. Wait a minute and retry.")
    elif code == 404:
        st.error(f"Ticker `{ticker}` not found.")
    else:
        st.error(f"Backend error ({code}): {e.response.text[:300]}")
    st.stop()
except Exception as e:
    st.error(f"Could not reach the API: {e}")
    st.stop()

# ── Headline numbers ──────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Ticker", data["ticker"])
col2.metric("Last actual price", f"${data['last_actual_price']:.2f}")
col2.caption(f"as of {data['last_actual_date']}")

last_pred = data["forecast"][-1]
delta = last_pred["predicted"] - data["last_actual_price"]
delta_pct = (delta / data["last_actual_price"]) * 100
col3.metric(
    f"Forecast (+{forecast_days}d)",
    f"${last_pred['predicted']:.2f}",
    f"{delta_pct:+.2f}%",
)
col3.caption(f"on {last_pred['date']}")

col4.metric(
    "Forecast range",
    f"${last_pred['lower']:.2f} – ${last_pred['upper']:.2f}",
)
col4.caption(f"{int(interval_width*100)}% confidence interval")

# ── Chart ─────────────────────────────────────────────────────────────────────
hist = pd.DataFrame(data["history"])
hist["date"] = pd.to_datetime(hist["date"])

fc = pd.DataFrame(data["forecast"])
fc["date"] = pd.to_datetime(fc["date"])

naive = pd.DataFrame(data["naive_forecast"])
naive["date"] = pd.to_datetime(naive["date"])

fig = go.Figure()

# Historical actual prices
fig.add_trace(go.Scatter(
    x=hist["date"], y=hist["predicted"],
    name="Historical price", mode="lines",
    line=dict(color="#1f77b4", width=2),
))

# Confidence interval (filled area)
fig.add_trace(go.Scatter(
    x=fc["date"], y=fc["upper"],
    mode="lines", line=dict(width=0),
    showlegend=False, hoverinfo="skip",
))
fig.add_trace(go.Scatter(
    x=fc["date"], y=fc["lower"],
    mode="lines", line=dict(width=0),
    fill="tonexty", fillcolor="rgba(255, 127, 14, 0.18)",
    name=f"{int(interval_width*100)}% CI", hoverinfo="skip",
))

# Prophet forecast
fig.add_trace(go.Scatter(
    x=fc["date"], y=fc["predicted"],
    name="Prophet forecast", mode="lines",
    line=dict(color="#ff7f0e", width=2),
))

# Naive baseline
fig.add_trace(go.Scatter(
    x=naive["date"], y=naive["predicted"],
    name="Naive (last price)", mode="lines",
    line=dict(color="#888", width=1.5, dash="dash"),
))

fig.update_layout(
    title=f"{data['ticker']} — last 180 days + {forecast_days}d forecast",
    xaxis_title="Date",
    yaxis_title="Price (USD)",
    hovermode="x unified",
    height=520,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=20, r=20, t=60, b=20),
)
st.plotly_chart(fig, use_container_width=True)

# ── Honest model evaluation ───────────────────────────────────────────────────
st.subheader("How accurate is this, really?")
st.markdown(
    "Both models were back-tested on a held-out tail of the data. "
    "Lower is better for MAE / RMSE / MAPE. CI coverage should be close to the "
    f"chosen confidence level ({int(interval_width*100)}%)."
)

m = data["metrics"]
nm = data["naive_metrics"]

eval_df = pd.DataFrame({
    "Metric": ["MAE ($)", "RMSE ($)", "MAPE (%)", "CI coverage"],
    "Prophet": [
        f"{m['mae']:.2f}",
        f"{m['rmse']:.2f}",
        f"{m['mape']:.2f}%",
        f"{m['ci_coverage']*100:.1f}%",
    ],
    "Naive (last price)": [
        f"{nm['mae']:.2f}",
        f"{nm['rmse']:.2f}",
        f"{nm['mape']:.2f}%",
        f"{nm['ci_coverage']*100:.1f}%",
    ],
})
st.dataframe(eval_df, hide_index=True, use_container_width=True)

if data["beats_naive"]:
    st.success(
        f"✅ Prophet beat the naive baseline on this ticker "
        f"(MAPE {m['mape']:.2f}% vs {nm['mape']:.2f}%)."
    )
else:
    st.warning(
        f"⚠️ Prophet did **not** beat the naive baseline on this ticker "
        f"(MAPE {m['mape']:.2f}% vs {nm['mape']:.2f}%). "
        "This is normal — daily stock prices are close to a random walk, "
        "and most simple models can't beat 'tomorrow ≈ today'."
    )

with st.expander("Why the naive comparison matters"):
    st.markdown(
        "- **Stock prices are roughly a random walk with drift.** The best estimate of "
        "tomorrow's price is usually today's price.\n"
        "- A model that can't beat that baseline is adding noise, not signal — "
        "even if its raw MAPE looks impressive (e.g. 2%).\n"
        "- The honest takeaway from any retail-grade stock forecaster is the "
        "**confidence interval width**, not the point forecast.\n"
        "- For real alpha, you'd need alternative data, regime models, "
        "or higher-frequency signals — not Prophet on daily closes."
    )

# ── Forecast table ────────────────────────────────────────────────────────────
with st.expander("📋 Forecast table"):
    show = fc.copy()
    show["date"] = show["date"].dt.strftime("%Y-%m-%d")
    show = show.rename(columns={
        "date": "Date",
        "predicted": "Predicted",
        "lower": "Lower",
        "upper": "Upper",
    })
    st.dataframe(show, hide_index=True, use_container_width=True)