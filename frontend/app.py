import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go

API_URL = "https://stock-forecaster-api-h5de.onrender.com"

SUGGESTED = [
    ("AAPL", "Apple"),
    ("MSFT", "Microsoft"),
    ("TSLA", "Tesla"),
    ("NVDA", "Nvidia"),
    ("AMZN", "Amazon"),
    ("GOOGL", "Google"),
    ("SPY", "S&P 500 ETF"),
]

st.set_page_config(page_title="Stock Forecaster", page_icon="📈", layout="wide")

st.markdown("""
    <style>
    .metric-card {
        background: #1e1e2e;
        border-radius: 10px;
        padding: 16px 20px;
        text-align: center;
    }
    .metric-label { color: #aaa; font-size: 13px; margin-bottom: 4px; }
    .metric-value { color: #fff; font-size: 24px; font-weight: 700; }
    .ticker-btn button {
        background: #2a2a3e !important;
        border: 1px solid #444 !important;
        color: #fff !important;
        border-radius: 8px !important;
    }
    </style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📈 Stock Forecaster")
st.caption("Powered by Meta's Prophet model · Data from Tiingo")

st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")

    ticker_input = st.text_input("Ticker Symbol", value="AAPL", max_chars=10).upper().strip()

    st.markdown("**Quick pick:**")
    cols = st.columns(2)
    for i, (t, name) in enumerate(SUGGESTED):
        if cols[i % 2].button(t, key=f"btn_{t}", help=name, use_container_width=True):
            ticker_input = t
            st.session_state["ticker"] = t

    st.divider()
    forecast_days = st.slider("Forecast Days", min_value=7, max_value=90, value=30, step=1)
    confidence = st.select_slider(
        "Confidence Interval",
        options=[0.80, 0.85, 0.90, 0.95, 0.99],
        value=0.95,
        format_func=lambda x: f"{int(x*100)}%",
    )

    st.divider()
    run = st.button("🔮 Generate Forecast", type="primary", use_container_width=True)

# use session state ticker if set by quick pick
ticker = st.session_state.get("ticker", ticker_input) if "ticker" in st.session_state else ticker_input

# ── Forecast ──────────────────────────────────────────────────────────────────
if run:
    st.session_state["ticker"] = ticker_input  # sync back
    ticker = ticker_input

    with st.spinner(f"Forecasting {ticker} — this takes ~2 minutes while the model trains..."):
        try:
            resp = requests.get(
                f"{API_URL}/forecast",
                params={"ticker": ticker, "forecast_days": forecast_days, "interval_width": confidence},
                timeout=300,
            )
            if resp.status_code == 404:
                st.error(f"Ticker **{ticker}** not found. Check the symbol and try again.")
                st.stop()
            elif resp.status_code != 200:
                st.error(f"API error {resp.status_code}: {resp.json().get('detail', 'Unknown error')}")
                st.stop()

            data = resp.json()
        except requests.exceptions.Timeout:
            st.error("Request timed out. The server may be waking up — try again in 30 seconds.")
            st.stop()
        except Exception as e:
            st.error(f"Connection error: {e}")
            st.stop()

    df = pd.DataFrame(data["forecast"])
    df["date"] = pd.to_datetime(df["date"])

    # ── Key numbers ───────────────────────────────────────────────────────────
    last_price = data["last_actual_price"]
    end_price = df["predicted"].iloc[-1]
    change = end_price - last_price
    change_pct = (change / last_price) * 100
    direction = "▲" if change >= 0 else "▼"
    color = "#00c853" if change >= 0 else "#ff1744"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current Price", f"${last_price:,.2f}")
    c2.metric(f"{forecast_days}-Day Forecast", f"${end_price:,.2f}", f"{direction} {abs(change_pct):.1f}%")
    c3.metric("Model Error (MAPE)", f"{data['metrics']['mape']:.1f}%")
    c4.metric("CI Coverage", f"{data['metrics']['ci_coverage']*100:.0f}%")

    st.divider()

    # ── Chart ─────────────────────────────────────────────────────────────────
    fig = go.Figure()

    # Confidence band
    fig.add_trace(go.Scatter(
        x=pd.concat([df["date"], df["date"][::-1]]),
        y=pd.concat([df["upper"], df["lower"][::-1]]),
        fill="toself",
        fillcolor="rgba(99, 110, 250, 0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        name=f"{int(confidence*100)}% Confidence Band",
        hoverinfo="skip",
    ))

    # Forecast line
    fig.add_trace(go.Scatter(
        x=df["date"],
        y=df["predicted"],
        mode="lines+markers",
        name="Forecast",
        line=dict(color="#636EFA", width=2.5),
        marker=dict(size=5),
        hovertemplate="<b>%{x|%b %d, %Y}</b><br>Predicted: $%{y:,.2f}<extra></extra>",
    ))

    # Current price marker
    fig.add_hline(
        y=last_price,
        line_dash="dash",
        line_color="#FFA500",
        annotation_text=f"Current: ${last_price:,.2f}",
        annotation_position="top left",
    )

    fig.update_layout(
        title=f"{ticker} — {forecast_days}-Day Price Forecast",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        hovermode="x unified",
        template="plotly_dark",
        height=480,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=60, b=20),
    )

    st.plotly_chart(fig, use_container_width=True)

    # ── Model metrics ─────────────────────────────────────────────────────────
    with st.expander("Model Performance Metrics"):
        m = data["metrics"]
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("MAE", f"${m['mae']:,.2f}", help="Mean Absolute Error in dollars")
        mc2.metric("RMSE", f"${m['rmse']:,.2f}", help="Root Mean Squared Error in dollars")
        mc3.metric("MAPE", f"{m['mape']:.2f}%", help="Mean Absolute Percentage Error")
        mc4.metric("CI Coverage", f"{m['ci_coverage']*100:.1f}%", help="% of actual prices within confidence band")
        st.caption("Metrics are evaluated on the last 120 trading days held out from training.")

    # ── Forecast table ────────────────────────────────────────────────────────
    with st.expander("View Forecast Data"):
        display_df = df.copy()
        display_df["date"] = display_df["date"].dt.strftime("%b %d, %Y")
        display_df.columns = ["Date", "Predicted ($)", "Lower ($)", "Upper ($)"]
        st.dataframe(display_df, use_container_width=True, hide_index=True)

else:
    # ── Empty state ───────────────────────────────────────────────────────────
    st.markdown("""
    ### How it works
    1. **Pick a ticker** from the sidebar or type your own (e.g. `AAPL`, `TSLA`, `NVDA`)
    2. **Choose forecast days** — how far ahead to predict (7–90 days)
    3. **Click Generate Forecast** and wait ~2 minutes while the model trains

    The app uses Meta's **Prophet** time-series model trained on years of daily closing prices.
    It predicts future prices along with a confidence interval showing the likely range.

    > ⚠️ This is for educational purposes only — not financial advice.
    """)
