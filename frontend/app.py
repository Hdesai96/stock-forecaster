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

st.set_page_config(page_title="Stock Forecaster", page_icon="", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Hide Streamlit branding */
    #MainMenu, footer, header { visibility: hidden; }

    /* Main background */
    .stApp { background-color: #13131f; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1a1a2e;
        border-right: 1px solid #2a2a45;
    }
    [data-testid="stSidebar"] * { font-family: 'Inter', sans-serif; }

    /* Ticker quick-pick buttons */
    div[data-testid="stHorizontalBlock"] button {
        background-color: #22223a !important;
        border: 1px solid #35355a !important;
        color: #d0d0f0 !important;
        border-radius: 6px !important;
        font-size: 12px !important;
        font-weight: 500 !important;
        letter-spacing: 0.05em !important;
        padding: 4px 8px !important;
        transition: all 0.2s ease !important;
    }
    div[data-testid="stHorizontalBlock"] button:hover {
        background-color: #252540 !important;
        border-color: #4a4a80 !important;
        color: #ffffff !important;
    }

    /* Primary button */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        letter-spacing: 0.03em !important;
        padding: 10px !important;
        transition: opacity 0.2s ease !important;
    }
    .stButton > button[kind="primary"]:hover { opacity: 0.9 !important; }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: #1e1e35;
        border: 1px solid #2e2e50;
        border-radius: 10px;
        padding: 18px 20px !important;
    }
    [data-testid="stMetricLabel"] { color: #8888aa !important; font-size: 12px !important; font-weight: 500 !important; letter-spacing: 0.06em !important; text-transform: uppercase; }
    [data-testid="stMetricValue"] { color: #ffffff !important; font-size: 26px !important; font-weight: 700 !important; }
    [data-testid="stMetricDelta"] { font-size: 13px !important; font-weight: 500 !important; }

    /* Divider */
    hr { border-color: #2a2a45 !important; }

    /* Text input */
    .stTextInput input {
        background-color: #22223a !important;
        border: 1px solid #35355a !important;
        border-radius: 8px !important;
        color: #f0f0ff !important;
        font-family: 'Inter', sans-serif !important;
    }

    /* Slider */
    .stSlider [data-baseweb="slider"] { padding: 4px 0; }

    /* Expander */
    [data-testid="stExpander"] {
        background: #1e1e35;
        border: 1px solid #2e2e50 !important;
        border-radius: 10px !important;
    }

    /* Dataframe */
    [data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }

    /* Section label */
    .section-label {
        color: #8888aa;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-bottom: 8px;
    }

    /* Hero */
    .hero-title {
        font-size: 36px;
        font-weight: 700;
        color: #f0f0ff;
        letter-spacing: -0.5px;
        line-height: 1.2;
    }
    .hero-sub {
        font-size: 14px;
        color: #8888aa;
        margin-top: 4px;
        letter-spacing: 0.02em;
    }

    /* Empty state card */
    .empty-card {
        background: #1e1e35;
        border: 1px solid #2e2e50;
        border-radius: 12px;
        padding: 36px 40px;
        color: #9090b0;
        line-height: 1.8;
        font-size: 15px;
    }
    .empty-card h3 { color: #c8c8e0; font-size: 18px; font-weight: 600; margin-bottom: 12px; }
    .empty-card code { background: #2a2a45; padding: 2px 6px; border-radius: 4px; font-size: 13px; color: #a78bfa; }
    .empty-card .disclaimer {
        margin-top: 20px;
        padding: 12px 16px;
        background: #22223a;
        border-left: 3px solid #4f46e5;
        border-radius: 4px;
        font-size: 13px;
        color: #6b6b8a;
    }
    </style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">Stock Forecaster</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">AI-powered price forecasting &nbsp;·&nbsp; Meta Prophet model &nbsp;·&nbsp; Data by Tiingo</div>', unsafe_allow_html=True)
st.divider()

# ── Sidebar ────────────────────────────────────────────────────────────────────
if "ticker" not in st.session_state:
    st.session_state["ticker"] = "AAPL"

with st.sidebar:
    st.markdown('<div class="section-label">Ticker</div>', unsafe_allow_html=True)

    def _on_text_change():
        st.session_state["ticker"] = st.session_state["_ticker_input"].upper().strip()

    st.text_input(
        "",
        value=st.session_state["ticker"],
        max_chars=10,
        key="_ticker_input",
        label_visibility="collapsed",
        on_change=_on_text_change,
    )

    st.markdown('<div class="section-label" style="margin-top:16px">Quick Select</div>', unsafe_allow_html=True)
    cols = st.columns(2)
    for i, (t, name) in enumerate(SUGGESTED):
        if cols[i % 2].button(t, key=f"btn_{t}", help=name, use_container_width=True):
            st.session_state["ticker"] = t
            st.rerun()

    st.divider()

    st.markdown('<div class="section-label">Forecast Horizon</div>', unsafe_allow_html=True)
    forecast_days = st.slider("", min_value=7, max_value=90, value=30, step=1, label_visibility="collapsed")
    st.caption(f"{forecast_days} trading days ahead")

    st.markdown('<div class="section-label" style="margin-top:12px">Confidence Interval</div>', unsafe_allow_html=True)
    confidence = st.select_slider(
        "",
        options=[0.80, 0.85, 0.90, 0.95, 0.99],
        value=0.95,
        format_func=lambda x: f"{int(x*100)}%",
        label_visibility="collapsed",
    )

    st.divider()
    run = st.button("Generate Forecast", type="primary", use_container_width=True)

ticker = st.session_state["ticker"].upper().strip()

# ── Forecast ───────────────────────────────────────────────────────────────────
if run:

    with st.spinner(f"Training model for {ticker} — this takes about 2 minutes..."):
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

    last_price = data["last_actual_price"]
    end_price = df["predicted"].iloc[-1]
    change = end_price - last_price
    change_pct = (change / last_price) * 100
    direction = "+" if change >= 0 else ""

    # ── Metrics ────────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current Price", f"${last_price:,.2f}")
    c2.metric(f"{forecast_days}-Day Forecast", f"${end_price:,.2f}", f"{direction}{change_pct:.1f}%")
    c3.metric("Model Error (MAPE)", f"{data['metrics']['mape']:.1f}%")
    c4.metric("CI Coverage", f"{data['metrics']['ci_coverage']*100:.0f}%")

    st.divider()

    # ── Chart ──────────────────────────────────────────────────────────────────
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=pd.concat([df["date"], df["date"][::-1]]),
        y=pd.concat([df["upper"], df["lower"][::-1]]),
        fill="toself",
        fillcolor="rgba(79, 70, 229, 0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        name=f"{int(confidence*100)}% Confidence Band",
        hoverinfo="skip",
    ))

    fig.add_trace(go.Scatter(
        x=df["date"],
        y=df["predicted"],
        mode="lines",
        name="Forecast",
        line=dict(color="#7c3aed", width=2.5),
        hovertemplate="<b>%{x|%b %d, %Y}</b><br>$%{y:,.2f}<extra></extra>",
    ))

    fig.add_hline(
        y=last_price,
        line_dash="dot",
        line_color="rgba(148, 163, 184, 0.4)",
        annotation_text=f"Last close  ${last_price:,.2f}",
        annotation_font_color="#94a3b8",
        annotation_font_size=12,
        annotation_position="top left",
    )

    fig.update_layout(
        paper_bgcolor="#13131f",
        plot_bgcolor="#13131f",
        font=dict(family="Inter", color="#6b6b8a"),
        title=dict(
            text=f"<b>{ticker}</b>  ·  {forecast_days}-Day Forecast",
            font=dict(size=16, color="#f0f0ff"),
            x=0,
            xanchor="left",
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor="#22223a",
            zeroline=False,
            tickfont=dict(size=11),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#22223a",
            zeroline=False,
            tickprefix="$",
            tickfont=dict(size=11),
        ),
        hovermode="x unified",
        height=460,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=12),
        ),
        margin=dict(l=10, r=10, t=60, b=10),
    )

    st.plotly_chart(fig, use_container_width=True)

    # ── Details ────────────────────────────────────────────────────────────────
    with st.expander("Model Performance"):
        m = data["metrics"]
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("MAE", f"${m['mae']:,.2f}", help="Mean Absolute Error in dollars")
        mc2.metric("RMSE", f"${m['rmse']:,.2f}", help="Root Mean Squared Error in dollars")
        mc3.metric("MAPE", f"{m['mape']:.2f}%", help="Mean Absolute Percentage Error")
        mc4.metric("CI Coverage", f"{m['ci_coverage']*100:.1f}%", help="% of actual prices within confidence band")
        st.caption("Evaluated on the last 120 trading days held out from training.")

    with st.expander("Forecast Table"):
        display_df = df.copy()
        display_df["date"] = display_df["date"].dt.strftime("%b %d, %Y")
        display_df.columns = ["Date", "Predicted ($)", "Lower ($)", "Upper ($)"]
        st.dataframe(display_df, use_container_width=True, hide_index=True)

else:
    st.markdown("""
    <div class="empty-card">
        <h3>How it works</h3>
        <ol>
            <li>Pick a stock from <b>Quick Select</b> or type your own ticker — e.g. <code>AAPL</code>, <code>TSLA</code>, <code>NVDA</code></li>
            <li>Set how many days ahead you want to forecast (7 – 90)</li>
            <li>Click <b>Generate Forecast</b> and wait ~2 minutes while the model trains on years of price history</li>
        </ol>
        <div class="disclaimer">
            For educational purposes only &nbsp;·&nbsp; Not financial advice
        </div>
    </div>
    """, unsafe_allow_html=True)
