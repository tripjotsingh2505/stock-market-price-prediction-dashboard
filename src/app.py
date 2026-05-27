# ============================================================
# RELIANCE STOCK PRICE PREDICTION DASHBOARD
# Midnight Blue Edition — by Tripjot Singh
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from datetime import date, timedelta
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Reliance Stock Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# MIDNIGHT BLUE PREMIUM CSS — DEEP SPACE THEME
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600;700&display=swap');

    /* ── Base ── */
    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
        background-color: #0A0E1A;
        color: #E8EAED;
    }
    .stApp {
        background-color: #0A0E1A;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0D1226 0%, #0A0E1A 100%);
        border-right: 1px solid #1E2A3A;
    }
    [data-testid="stSidebar"] * {
        color: #E8EAED !important;
    }
    [data-testid="stSidebar"] .stSelectbox > div > div {
        background-color: #0D1226 !important;
        border: 1px solid #1E2A3A !important;
        color: #E8EAED !important;
    }
    [data-testid="stSidebar"] .stRadio > div {
        background: transparent !important;
    }

    /* ── Header Banner ── */
    .dashboard-header {
        background: linear-gradient(135deg, #0D1226 0%, #0F1830 50%, #0D1226 100%);
        border: 1px solid #1E2A3A;
        border-left: 4px solid #00D4FF;
        border-radius: 6px;
        padding: 22px 30px;
        margin-bottom: 22px;
        position: relative;
        overflow: hidden;
        box-shadow: 0 0 30px #00D4FF08, inset 0 0 60px #00D4FF04;
    }
    .dashboard-header::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background: repeating-linear-gradient(
            0deg,
            transparent,
            transparent 2px,
            #00D4FF04 2px,
            #00D4FF04 4px
        );
        pointer-events: none;
    }
    .dashboard-header h1 {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.6rem;
        font-weight: 600;
        color: #E8EAED;
        margin: 0 0 6px 0;
        letter-spacing: 2px;
        text-transform: uppercase;
        text-shadow: 0 0 20px #00D4FF30;
    }
    .dashboard-header p {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        color: #8892A4;
        margin: 0;
        letter-spacing: 1px;
    }
    .live-dot {
        display: inline-block;
        width: 8px; height: 8px;
        background: #00D4FF;
        border-radius: 50%;
        animation: pulse 1.5s infinite;
        margin-right: 6px;
        vertical-align: middle;
        box-shadow: 0 0 8px #00D4FF;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); box-shadow: 0 0 8px #00D4FF; }
        50% { opacity: 0.5; transform: scale(1.3); box-shadow: 0 0 16px #00D4FF; }
    }

    /* ── KPI Cards ── */
    .kpi-card {
        background: linear-gradient(135deg, #0D1226 0%, #0F1830 100%);
        border: 1px solid #1E2A3A;
        border-top: 2px solid #00D4FF;
        border-radius: 6px;
        padding: 16px 18px;
        margin-bottom: 12px;
        font-family: 'IBM Plex Mono', monospace;
        transition: border-color 0.2s, box-shadow 0.2s;
    }
    .kpi-card:hover {
        border-color: #00D4FF;
        box-shadow: 0 0 18px #00D4FF12;
    }
    .kpi-label {
        font-size: 0.65rem;
        color: #8892A4;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 1.5rem;
        font-weight: 600;
        color: #E8EAED;
    }
    .kpi-delta-pos { color: #00FF88; font-size: 0.8rem; }
    .kpi-delta-neg { color: #FF4560; font-size: 0.8rem; }

    /* ── Section Headers ── */
    .section-header {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem;
        letter-spacing: 3px;
        text-transform: uppercase;
        color: #8892A4;
        border-bottom: 1px solid #1E2A3A;
        padding-bottom: 8px;
        margin: 24px 0 16px 0;
    }
    .section-header span {
        color: #00D4FF;
    }

    /* ── Signal Badges ── */
    .signal-bullish {
        background: #00FF8812;
        border: 1px solid #00FF88;
        color: #00FF88;
        padding: 6px 14px;
        border-radius: 4px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 1px;
    }
    .signal-bearish {
        background: #FF456012;
        border: 1px solid #FF4560;
        color: #FF4560;
        padding: 6px 14px;
        border-radius: 4px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 1px;
    }
    .signal-neutral {
        background: #00D4FF12;
        border: 1px solid #00D4FF;
        color: #00D4FF;
        padding: 6px 14px;
        border-radius: 4px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 1px;
    }

    /* ── Metrics ── */
    [data-testid="metric-container"] {
        background: #0D1226;
        border: 1px solid #1E2A3A;
        border-top: 2px solid #00D4FF;
        padding: 12px 16px;
        border-radius: 6px;
    }
    [data-testid="metric-container"] label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.65rem !important;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: #8892A4 !important;
    }
    [data-testid="metric-container"] [data-testid="metric-value"] {
        font-family: 'IBM Plex Mono', monospace;
        color: #E8EAED !important;
        font-size: 1.3rem !important;
    }
    [data-testid="metric-container"] [data-testid="metric-delta"] svg { display: none; }

    /* ── Tabs ── */
    [data-baseweb="tab-list"] {
        background: #0D1226 !important;
        border-bottom: 1px solid #1E2A3A !important;
        gap: 0 !important;
    }
    [data-baseweb="tab"] {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.72rem !important;
        letter-spacing: 1.5px !important;
        text-transform: uppercase !important;
        color: #8892A4 !important;
        padding: 12px 24px !important;
        border-bottom: 2px solid transparent !important;
        background: transparent !important;
    }
    [aria-selected="true"][data-baseweb="tab"] {
        color: #00D4FF !important;
        border-bottom: 2px solid #00D4FF !important;
        background: #00D4FF08 !important;
        text-shadow: 0 0 12px #00D4FF60;
    }

    /* ── Inputs ── */
    .stSelectbox > div > div,
    .stMultiSelect > div > div {
        background: #0D1226 !important;
        border: 1px solid #1E2A3A !important;
        color: #E8EAED !important;
        border-radius: 4px !important;
        font-family: 'IBM Plex Mono', monospace !important;
    }
    .stRadio > div > label {
        color: #E8EAED !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.8rem !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #00A8CC, #00D4FF) !important;
        color: #0A0E1A !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-weight: 700 !important;
        letter-spacing: 1.5px !important;
        text-transform: uppercase !important;
        border: none !important;
        border-radius: 4px !important;
        padding: 8px 20px !important;
        font-size: 0.75rem !important;
        box-shadow: 0 0 16px #00D4FF30 !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #00D4FF, #40E8FF) !important;
        box-shadow: 0 0 24px #00D4FF50 !important;
        transform: translateY(-1px);
    }

    /* ── Expander ── */
    [data-testid="stExpander"] {
        background: #0D122680 !important;
        border: 1px solid #1E2A3A !important;
        border-radius: 4px !important;
    }
    [data-testid="stExpander"] summary {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.72rem !important;
        color: #8892A4 !important;
        letter-spacing: 1px !important;
    }

    /* ── Prediction Box ── */
    .prediction-box {
        background: linear-gradient(135deg, #0D1226, #0F1830);
        border: 1px solid #1E2A3A;
        border-left: 4px solid #00D4FF;
        border-radius: 6px;
        padding: 20px 24px;
        font-family: 'IBM Plex Mono', monospace;
        margin: 12px 0;
        box-shadow: 0 0 20px #00D4FF08;
    }
    .pred-label {
        font-size: 0.65rem;
        color: #8892A4;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    .pred-value {
        font-size: 2rem;
        font-weight: 600;
        color: #E8EAED;
    }

    /* ── Divider ── */
    hr {
        border-color: #1E2A3A !important;
        margin: 16px 0 !important;
    }

    /* ── Info/Warning ── */
    [data-testid="stInfo"] {
        background: #00D4FF0D !important;
        border: 1px solid #00D4FF30 !important;
        color: #A8D8EA !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.8rem !important;
    }
    [data-testid="stWarning"] {
        background: #FF456010 !important;
        border: 1px solid #FF456040 !important;
        color: #FF8C69 !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.8rem !important;
    }
    [data-testid="stSuccess"] {
        background: #00FF8810 !important;
        border: 1px solid #00FF8840 !important;
        color: #00FF88 !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.8rem !important;
    }
    [data-testid="stError"] {
        background: #FF456010 !important;
        border: 1px solid #FF456040 !important;
        color: #FF4560 !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.8rem !important;
    }

    /* ── Spinner ── */
    .stSpinner > div {
        border-top-color: #00D4FF !important;
    }
    .stSpinner p {
        font-family: 'IBM Plex Mono', monospace !important;
        color: #8892A4 !important;
    }

    /* ── Footer ── */
    .dashboard-footer {
        text-align: center;
        padding: 24px;
        border-top: 1px solid #1E2A3A;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.68rem;
        color: #8892A4;
        letter-spacing: 1px;
        margin-top: 32px;
    }
    .dashboard-footer a { color: #00D4FF; text-decoration: none; }
    .dashboard-footer a:hover { color: #40E8FF; text-shadow: 0 0 8px #00D4FF; }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #0A0E1A; }
    ::-webkit-scrollbar-thumb { background: #1E2A3A; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #00D4FF40; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# PLOTLY THEME — MIDNIGHT BLUE DEEP SPACE
# ============================================================
BLOOMBERG_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#0A0E1A",
    plot_bgcolor="#0D1226",
    font=dict(family="IBM Plex Mono, monospace", color="#E8EAED", size=11),
    title_font=dict(family="IBM Plex Mono, monospace", size=13, color="#E8EAED"),
    xaxis=dict(
        gridcolor="#1E2A3A", linecolor="#1E2A3A",
        tickfont=dict(color="#8892A4", size=10),
        title_font=dict(color="#8892A4"),
        showgrid=True
    ),
    yaxis=dict(
        gridcolor="#1E2A3A", linecolor="#1E2A3A",
        tickfont=dict(color="#8892A4", size=10),
        title_font=dict(color="#8892A4"),
        showgrid=True
    ),
    legend=dict(
        bgcolor="#0D1226", bordercolor="#1E2A3A", borderwidth=1,
        font=dict(color="#E8EAED", size=10)
    ),
    hoverlabel=dict(
        bgcolor="#0D1226", bordercolor="#00D4FF",
        font=dict(color="#E8EAED", family="IBM Plex Mono", size=11)
    ),
    margin=dict(l=50, r=20, t=50, b=40)
)


# ============================================================
# DATA LOADING
# ============================================================
@st.cache_data(ttl=3600)
def load_data():
    csv_path = "data/reliance_featured_data.csv"
    try:
        ticker = yf.Ticker("RELIANCE.NS")
        df = ticker.history(
            start="2010-01-01",
            end=date.today().strftime("%Y-%m-%d"),
            auto_adjust=False
        )
        if df.empty:
            raise Exception("Empty data from Yahoo Finance")
        df = df.reset_index()
        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
        df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
        df = df.dropna().sort_values("Date")
        df = df.set_index("Date", drop=False)
        return df
    except Exception:
        st.warning("⚠ Yahoo Finance unavailable. Loading cached dataset.")
        df = pd.read_csv(csv_path)
        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
        df = df.dropna(subset=["Date", "Open", "High", "Low", "Close", "Volume"])
        df = df.sort_values("Date")
        df = df.set_index("Date", drop=False)
        return df


# ============================================================
# FEATURE ENGINEERING
# ============================================================
def add_features(df):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date", drop=False)

    df['MA_7']   = df['Close'].rolling(7).mean()
    df['MA_21']  = df['Close'].rolling(21).mean()
    df['MA_50']  = df['Close'].rolling(50).mean()
    df['MA_200'] = df['Close'].rolling(200).mean()

    delta = df['Close'].diff()
    gain  = delta.where(delta > 0, 0).rolling(14).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + gain / loss))

    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD']            = exp12 - exp26
    df['MACD_Signal']     = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Histogram']  = df['MACD'] - df['MACD_Signal']

    df['BB_Middle'] = df['Close'].rolling(20).mean()
    std20           = df['Close'].rolling(20).std()
    df['BB_Upper']  = df['BB_Middle'] + std20 * 2
    df['BB_Lower']  = df['BB_Middle'] - std20 * 2
    df['BB_Width']  = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']

    for lag in [1, 2, 3, 5]:
        df[f'Close_Lag{lag}'] = df['Close'].shift(lag)

    df['Year']        = df.index.year
    df['Month']       = df.index.month
    df['Day']         = df.index.day
    df['Day_of_Week'] = df.index.dayofweek
    df['Quarter']     = df.index.quarter

    df['Daily_Return'] = df['Close'].pct_change() * 100
    df['ATR']          = (df['High'] - df['Low']).rolling(14).mean()
    df['Price_Range']  = df['High'] - df['Low']
    df['Price_Change'] = df['Close'] - df['Open']

    obv = [0]
    for i in range(1, len(df)):
        if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
            obv.append(obv[-1] + df['Volume'].iloc[i])
        elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
            obv.append(obv[-1] - df['Volume'].iloc[i])
        else:
            obv.append(obv[-1])
    df['OBV'] = obv

    return df.dropna()


# ============================================================
# MODEL TRAINING
# ============================================================
def train_models(df_hash):
    df = st.session_state['df_featured'].copy()

    exclude_cols = ['Open', 'High', 'Low', 'Volume', 'Date']
    feature_cols = [c for c in df.columns if c not in exclude_cols + ['Close']]

    df['Target'] = df['Close'].shift(-1)
    df = df.dropna()

    X = df[feature_cols]
    y = df['Target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train_sc = scaler_X.fit_transform(X_train)
    y_train_sc = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
    X_test_sc  = scaler_X.transform(X_test)

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_sc, y_train_sc)

    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train_sc, y_train_sc)

    xgb_model = xgb.XGBRegressor(
        n_estimators=100, learning_rate=0.05,
        max_depth=3, subsample=0.8,
        random_state=42, n_jobs=-1, verbosity=0
    )
    xgb_model.fit(X_train_sc, y_train_sc)

    return {
        'ridge': ridge, 'rf': rf, 'xgb': xgb_model,
        'scaler_X': scaler_X, 'scaler_y': scaler_y,
        'X_test': X_test, 'y_test': y_test,
        'X_test_scaled': X_test_sc,
        'X_train': X_train,
        'feature_names': feature_cols
    }


def get_predictions(models, model_key):
    pred_sc = models[model_key].predict(models['X_test_scaled'])
    return models['scaler_y'].inverse_transform(
        pred_sc.reshape(-1, 1)
    ).ravel()


def calc_metrics(actual, predicted):
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae  = mean_absolute_error(actual, predicted)
    r2   = r2_score(actual, predicted)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    return rmse, mae, r2, mape


# ============================================================
# HELPER: Chart Help
# ============================================================
def chart_help(text):
    with st.expander("📖 What does this chart show?", expanded=False):
        st.caption(text)


# ============================================================
# HELPER: Section Header
# ============================================================
def section_header(icon, title):
    st.markdown(
        f'<p class="section-header">{icon} <span>{title}</span></p>',
        unsafe_allow_html=True
    )


# ============================================================
# TAB 1 — OVERVIEW
# ============================================================
def tab_overview(raw_df, df, start_date, end_date):
    # KPI Row
    section_header("◈", "LIVE MARKET SNAPSHOT")

    latest = raw_df.iloc[-1]
    prev   = raw_df.iloc[-2]
    change     = latest['Close'] - prev['Close']
    change_pct = (change / prev['Close']) * 100
    week_ago   = raw_df.iloc[-6]['Close'] if len(raw_df) >= 6 else prev['Close']
    week_chg   = ((latest['Close'] - week_ago) / week_ago) * 100
    month_ago  = raw_df.iloc[-22]['Close'] if len(raw_df) >= 22 else prev['Close']
    month_chg  = ((latest['Close'] - month_ago) / month_ago) * 100
    year_high  = raw_df['High'].tail(252).max()
    year_low   = raw_df['Low'].tail(252).min()

    c1, c2, c3, c4, c5, c6 = st.columns(6)

    delta_class = "kpi-delta-pos" if change >= 0 else "kpi-delta-neg"
    delta_sym   = "▲" if change >= 0 else "▼"

    def kpi(col, label, value, delta=None, dclass="kpi-delta-pos"):
        with col:
            delta_html = f'<div class="{dclass}">{delta}</div>' if delta else ''
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{value}</div>
                {delta_html}
            </div>""", unsafe_allow_html=True)

    kpi(c1, "LAST PRICE (₹)", f"{latest['Close']:.2f}",
        f"{delta_sym} {abs(change):.2f} ({abs(change_pct):.2f}%)", delta_class)
    kpi(c2, "DAY HIGH (₹)",  f"{latest['High']:.2f}")
    kpi(c3, "DAY LOW (₹)",   f"{latest['Low']:.2f}")
    kpi(c4, "52W HIGH (₹)",  f"{year_high:.2f}")
    kpi(c5, "52W LOW (₹)",   f"{year_low:.2f}")

    wk_cls = "kpi-delta-pos" if week_chg >= 0 else "kpi-delta-neg"
    kpi(c6, "1M RETURN",
        f"{'▲' if month_chg >= 0 else '▼'} {abs(month_chg):.2f}%",
        f"1W: {'▲' if week_chg >= 0 else '▼'}{abs(week_chg):.2f}%", wk_cls)

    # ── Price Chart ──
    section_header("◈", "PRICE CHART")

    pc1, pc2 = st.columns([1, 2])
    with pc1:
        chart_style = st.radio(
            "Chart type", ["Candlestick", "Line"], horizontal=True
        )
    with pc2:
        overlays = st.multiselect(
            "Overlays",
            ["MA 7", "MA 21", "MA 50", "MA 200", "Bollinger Bands"],
            default=["MA 50", "MA 200"]
        )

    filt_raw = raw_df[
        (raw_df.index >= pd.Timestamp(start_date)) &
        (raw_df.index <= pd.Timestamp(end_date))
    ]
    filt_df = df[
        (df.index >= pd.Timestamp(start_date)) &
        (df.index <= pd.Timestamp(end_date))
    ]

    fig = go.Figure()

    if chart_style == "Candlestick":
        fig.add_trace(go.Candlestick(
            x=filt_raw.index,
            open=filt_raw['Open'], high=filt_raw['High'],
            low=filt_raw['Low'],   close=filt_raw['Close'],
            increasing_line_color='#00FF88',
            decreasing_line_color='#FF4560',
            increasing_fillcolor='rgba(0,255,136,0.12)',
            decreasing_fillcolor='rgba(255,69,96,0.12)',
            name='RELIANCE'
        ))
    else:
        fig.add_trace(go.Scatter(
            x=filt_raw.index, y=filt_raw['Close'],
            name='Close', mode='lines',
            line=dict(color='#00D4FF', width=2)
        ))

    ma_cfg = {
        "MA 7":  ('MA_7',   '#00CEC9'),
        "MA 21": ('MA_21',  '#55EFC4'),
        "MA 50": ('MA_50',  '#00D4FF'),
        "MA 200":('MA_200', '#A29BFE')
    }
    for label, (col, clr) in ma_cfg.items():
        if label in overlays and col in filt_df.columns:
            fig.add_trace(go.Scatter(
                x=filt_df.index, y=filt_df[col],
                name=label, mode='lines',
                line=dict(color=clr, width=1.4, dash='solid')
            ))

    if "Bollinger Bands" in overlays and 'BB_Upper' in filt_df.columns:
        fig.add_trace(go.Scatter(
            x=filt_df.index, y=filt_df['BB_Upper'],
            name='BB Upper', mode='lines',
            line=dict(color='#74B9FF', width=1, dash='dot'), showlegend=True
        ))
        fig.add_trace(go.Scatter(
            x=filt_df.index, y=filt_df['BB_Lower'],
            name='BB Lower', mode='lines',
            line=dict(color='#74B9FF', width=1, dash='dot'),
            fill='tonexty', fillcolor='rgba(116,185,255,0.03)', showlegend=False
        ))

    fig.update_layout(
        **BLOOMBERG_LAYOUT,
        title='RELIANCE.NS — Price History',
        xaxis_title='Date', yaxis_title='Price (₹)',
        height=500, hovermode='x unified',
        xaxis_rangeslider_visible=False
    )
    st.plotly_chart(fig, use_container_width=True)
    chart_help("🕯️ Green candle = price rose that day, Red = price fell. Lines (MA) show the average price trend over time.")

    # ── Volume ──
    section_header("◈", "VOLUME ANALYSIS")

    vol_opts = st.multiselect(
        "Volume indicators",
        ["Daily Volume", "OBV"],
        default=["Daily Volume"],
        key="vol_overview"
    )

    if "OBV" in vol_opts:
        fig_v = make_subplots(specs=[[{"secondary_y": True}]])
    else:
        fig_v = go.Figure()

    if "Daily Volume" in vol_opts:
        colors = ['#00FF88' if c >= o else '#FF4560'
                  for c, o in zip(filt_raw['Close'], filt_raw['Open'])]
        if "OBV" in vol_opts:
            fig_v.add_trace(go.Bar(
                x=filt_raw.index, y=filt_raw['Volume'],
                name='Volume', marker_color=colors, opacity=0.7
            ), secondary_y=False)
        else:
            fig_v.add_trace(go.Bar(
                x=filt_raw.index, y=filt_raw['Volume'],
                name='Volume', marker_color=colors, opacity=0.75
            ))

    if "OBV" in vol_opts and 'OBV' in filt_df.columns:
        fig_v.add_trace(go.Scatter(
            x=filt_df.index, y=filt_df['OBV'],
            name='OBV', mode='lines',
            line=dict(color='#00D4FF', width=1.5)
        ), secondary_y=True)
        fig_v.update_yaxes(
            title_text="Volume", secondary_y=False,
            title_font=dict(color="#8892A4")
        )
        fig_v.update_yaxes(
            title_text="OBV", secondary_y=True,
            title_font=dict(color="#8892A4")
        )

    if "OBV" in vol_opts:
        fig_v.update_layout(
            **BLOOMBERG_LAYOUT,
            title='Trading Volume',
            xaxis_title='Date',
            height=300, bargap=0.1
        )
    else:
        fig_v.update_layout(
            **BLOOMBERG_LAYOUT,
            title='Trading Volume',
            xaxis_title='Date', yaxis_title='Volume',
            height=300, bargap=0.1
        )
    st.plotly_chart(fig_v, use_container_width=True)
    chart_help("📦 Bars show how many shares were traded each day. High volume = lots of interest. OBV line rising = more people buying than selling.")


# ============================================================
# TAB 2 — TECHNICAL ANALYSIS
# ============================================================
def tab_technical(df, start_date, end_date):
    filt = df[
        (df.index >= pd.Timestamp(start_date)) &
        (df.index <= pd.Timestamp(end_date))
    ]

    # ── RSI ──
    section_header("◈", "RSI — RELATIVE STRENGTH INDEX")
    rsi_opts = st.multiselect(
        "RSI display",
        ["RSI Line", "Overbought/Oversold", "RSI Fill Zones"],
        default=["RSI Line", "Overbought/Oversold", "RSI Fill Zones"],
        key="rsi_opts"
    )

    fig_rsi = go.Figure()
    if "RSI Line" in rsi_opts:
        # Color-coded RSI
        rsi_colors = filt['RSI'].apply(
            lambda x: '#FF4560' if x > 70 else ('#00FF88' if x < 30 else '#00D4FF')
        )
        fig_rsi.add_trace(go.Scatter(
            x=filt.index, y=filt['RSI'],
            name='RSI', mode='lines',
            line=dict(color='#00D4FF', width=1.8)
        ))
    if "RSI Fill Zones" in rsi_opts:
        fig_rsi.add_hrect(y0=70, y1=100,
            fillcolor="rgba(255,69,96,0.06)", line_width=0,
            annotation_text="OVERBOUGHT", annotation_position="right",
            annotation_font_color="#FF4560", annotation_font_size=9
        )
        fig_rsi.add_hrect(y0=0, y1=30,
            fillcolor="rgba(0,255,136,0.06)", line_width=0,
            annotation_text="OVERSOLD", annotation_position="right",
            annotation_font_color="#00FF88", annotation_font_size=9
        )
    if "Overbought/Oversold" in rsi_opts:
        fig_rsi.add_hline(y=70, line_dash="dash",
                          line_color="#FF4560", line_width=1)
        fig_rsi.add_hline(y=30, line_dash="dash",
                          line_color="#00FF88", line_width=1)
        fig_rsi.add_hline(y=50, line_dash="dot",
                          line_color="#8892A4", line_width=0.8)

    rsi_yaxis = {**BLOOMBERG_LAYOUT['yaxis'], 'range': [0, 100], 'title': 'RSI Value'}
    fig_rsi.update_layout(
        **{k: v for k, v in BLOOMBERG_LAYOUT.items() if k != 'yaxis'},
        title='RSI (14-Period)',
        xaxis_title='Date',
        height=340,
        yaxis=rsi_yaxis
    )
    st.plotly_chart(fig_rsi, use_container_width=True)
    chart_help("📊 RSI measures if the stock is too expensive (above 70 = overbought) or too cheap (below 30 = oversold) right now.")

    # ── MACD ──
    section_header("◈", "MACD — MOMENTUM OSCILLATOR")
    macd_opts = st.multiselect(
        "MACD display",
        ["MACD Line", "Signal Line", "Histogram"],
        default=["MACD Line", "Signal Line", "Histogram"],
        key="macd_opts"
    )

    fig_macd = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3], shared_xaxes=True)

    if "Histogram" in macd_opts:
        hist_colors = filt['MACD_Histogram'].apply(
            lambda x: '#00FF88' if x >= 0 else '#FF4560'
        )
        fig_macd.add_trace(go.Bar(
            x=filt.index, y=filt['MACD_Histogram'],
            name='Histogram', marker_color=hist_colors, opacity=0.7
        ), row=1, col=1)

    if "MACD Line" in macd_opts:
        fig_macd.add_trace(go.Scatter(
            x=filt.index, y=filt['MACD'],
            name='MACD', mode='lines',
            line=dict(color='#00D4FF', width=1.8)
        ), row=1, col=1)

    if "Signal Line" in macd_opts:
        fig_macd.add_trace(go.Scatter(
            x=filt.index, y=filt['MACD_Signal'],
            name='Signal', mode='lines',
            line=dict(color='#FF4560', width=1.5)
        ), row=1, col=1)

    # Add zero line
    fig_macd.add_hline(y=0, line_dash="dot", line_color="#8892A4",
                       line_width=0.8, row=1, col=1)

    fig_macd.update_layout(
        **BLOOMBERG_LAYOUT,
        title='MACD (12, 26, 9)',
        height=380, bargap=0.1
    )
    st.plotly_chart(fig_macd, use_container_width=True)
    chart_help("📈 MACD crossing above the signal line = good time to buy. Crossing below = possible sell. Green bars = momentum building up.")

    # ── Bollinger Bands ──
    section_header("◈", "BOLLINGER BANDS — VOLATILITY TRACKER")

    bb_opts = st.multiselect(
        "Bollinger display",
        ["Price", "Upper Band", "Middle Band (MA20)", "Lower Band", "Band Fill"],
        default=["Price", "Upper Band", "Middle Band (MA20)", "Lower Band", "Band Fill"],
        key="bb_opts"
    )

    fig_bb = go.Figure()

    if "Band Fill" in bb_opts and 'BB_Upper' in filt.columns:
        fig_bb.add_trace(go.Scatter(
            x=filt.index, y=filt['BB_Upper'],
            mode='lines', line=dict(color='rgba(0,0,0,0)'),
            showlegend=False, name='BB_fill_top'
        ))
        fig_bb.add_trace(go.Scatter(
            x=filt.index, y=filt['BB_Lower'],
            mode='lines', line=dict(color='rgba(0,0,0,0)'),
            fill='tonexty', fillcolor='rgba(116,185,255,0.03)',
            showlegend=False, name='BB_fill_bot'
        ))

    if "Upper Band" in bb_opts:
        fig_bb.add_trace(go.Scatter(
            x=filt.index, y=filt['BB_Upper'],
            name='Upper Band', mode='lines',
            line=dict(color='#74B9FF', width=1, dash='dot')
        ))
    if "Middle Band (MA20)" in bb_opts:
        fig_bb.add_trace(go.Scatter(
            x=filt.index, y=filt['BB_Middle'],
            name='MA 20', mode='lines',
            line=dict(color='#00D4FF', width=1.2, dash='dash')
        ))
    if "Lower Band" in bb_opts:
        fig_bb.add_trace(go.Scatter(
            x=filt.index, y=filt['BB_Lower'],
            name='Lower Band', mode='lines',
            line=dict(color='#74B9FF', width=1, dash='dot')
        ))
    if "Price" in bb_opts:
        fig_bb.add_trace(go.Scatter(
            x=filt.index, y=filt['Close'],
            name='Close Price', mode='lines',
            line=dict(color='#00FF88', width=1.5)
        ))

    fig_bb.update_layout(
        **BLOOMBERG_LAYOUT,
        title='Bollinger Bands (20, 2σ)',
        xaxis_title='Date', yaxis_title='Price (₹)',
        height=400
    )
    st.plotly_chart(fig_bb, use_container_width=True)
    chart_help("〰️ The two outer lines act like a price 'channel'. When price hits the top line it may be too high; bottom line = may be too low. Narrow channel = big move coming soon.")

    # ── Live Signal Box ──
    section_header("◈", "LIVE TRADING SIGNALS")

    latest = df.iloc[-1]

    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown("**RSI Signal**")
        rsi_val = latest['RSI']
        if rsi_val > 70:
            st.markdown(f'<span class="signal-bearish">🔴 OVERBOUGHT — RSI {rsi_val:.1f}</span>', unsafe_allow_html=True)
            st.caption("Price may cool down. Monitor for reversal.")
        elif rsi_val < 30:
            st.markdown(f'<span class="signal-bullish">🟢 OVERSOLD — RSI {rsi_val:.1f}</span>', unsafe_allow_html=True)
            st.caption("Potential recovery zone. Watch for bounce.")
        else:
            st.markdown(f'<span class="signal-neutral">🟡 NEUTRAL — RSI {rsi_val:.1f}</span>', unsafe_allow_html=True)
            st.caption("No extreme zone. Trend is stable.")

    with s2:
        st.markdown("**MACD Signal**")
        if latest['MACD'] > latest['MACD_Signal']:
            st.markdown('<span class="signal-bullish">🟢 BULLISH — MACD > Signal</span>', unsafe_allow_html=True)
            st.caption("Positive momentum. Upside likely.")
        else:
            st.markdown('<span class="signal-bearish">🔴 BEARISH — MACD < Signal</span>', unsafe_allow_html=True)
            st.caption("Negative momentum. Caution advised.")

    with s3:
        st.markdown("**MA Crossover Signal**")
        ma50, ma200, close = latest['MA_50'], latest['MA_200'], latest['Close']
        if close > ma50 > ma200:
            st.markdown('<span class="signal-bullish">🟢 GOLDEN CROSS</span>', unsafe_allow_html=True)
            st.caption("Price above MA50 & MA200. Strong uptrend.")
        elif close < ma50 < ma200:
            st.markdown('<span class="signal-bearish">🔴 DEATH CROSS</span>', unsafe_allow_html=True)
            st.caption("Price below MA50 & MA200. Strong downtrend.")
        else:
            st.markdown('<span class="signal-neutral">🟡 MIXED SIGNALS</span>', unsafe_allow_html=True)
            st.caption("Conflicting MA signals. Wait for clarity.")


# ============================================================
# TAB 3 — ML PREDICTIONS
# ============================================================
def tab_ml(df, raw_df, model_choice):
    with st.spinner("⚙ Training ML models on latest data..."):
        models = train_models(id(df))

    # Model selection
    if "Ridge" in model_choice:
        sel_key, model_name = 'ridge', "Ridge Regression"
    elif "Random Forest" in model_choice:
        sel_key, model_name = 'rf', "Random Forest"
    else:
        sel_key, model_name = 'xgb', "XGBoost"

    preds    = get_predictions(models, sel_key)
    rmse, mae, r2, mape = calc_metrics(models['y_test'].values, preds)

    # ── Performance Metrics ──
    section_header("◈", f"{model_name.upper()} — PERFORMANCE METRICS")
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("R² Score",  f"{r2:.4f}",  delta="Accuracy")
    with c2: st.metric("RMSE",      f"₹{rmse:.2f}", delta="Prediction Error")
    with c3: st.metric("MAE",       f"₹{mae:.2f}",  delta="Avg Error")
    with c4: st.metric("MAPE",      f"{mape:.2f}%", delta="% Error")

    st.markdown("---")

    # ── Actual vs Predicted ──
    section_header("◈", "ACTUAL VS PREDICTED PRICE")

    pred_opts = st.multiselect(
        "Series to display",
        ["Actual Price", "Predicted Price", "Error Band"],
        default=["Actual Price", "Predicted Price"],
        key="pred_opts"
    )

    fig_pred = go.Figure()
    actual = models['y_test'].values
    idx    = models['X_test'].index

    if "Error Band" in pred_opts:
        upper = preds + mae
        lower = preds - mae
        fig_pred.add_trace(go.Scatter(
            x=np.concatenate([idx, idx[::-1]]),
            y=np.concatenate([upper, lower[::-1]]),
            fill='toself', fillcolor='rgba(0,212,255,0.06)',
            line=dict(color='rgba(0,0,0,0)'),
            name='Error Band (±MAE)', showlegend=True
        ))

    if "Actual Price" in pred_opts:
        fig_pred.add_trace(go.Scatter(
            x=idx, y=actual,
            name='Actual Price', mode='lines',
            line=dict(color='#00D4FF', width=2)
        ))

    if "Predicted Price" in pred_opts:
        fig_pred.add_trace(go.Scatter(
            x=idx, y=preds,
            name=f'Predicted ({model_name})', mode='lines',
            line=dict(color='#FF4560', width=1.5, dash='solid')
        ))

    fig_pred.update_layout(
        **BLOOMBERG_LAYOUT,
        title=f'Actual vs Predicted — {model_name}',
        xaxis_title='Date', yaxis_title='Price (₹)',
        height=420, hovermode='x unified'
    )
    st.plotly_chart(fig_pred, use_container_width=True)
    chart_help(f"🤖 Cyan line shows actual prices, Red line shows predicted prices. The shaded band shows error range; closer lines mean better prediction.")

    # ── Next Day Prediction ──
    section_header("◈", "NEXT TRADING DAY FORECAST")

    latest_feat = df[models['feature_names']].iloc[-1].values.reshape(1, -1)
    latest_sc   = models['scaler_X'].transform(latest_feat)
    next_pred   = models['scaler_y'].inverse_transform(
        models[sel_key].predict(latest_sc).reshape(-1, 1)
    )[0][0]
    curr_price  = raw_df['Close'].iloc[-1]
    diff        = next_pred - curr_price
    diff_pct    = (diff / curr_price) * 100

    pc1, pc2, pc3 = st.columns(3)
    with pc1:
        st.markdown(f"""
        <div class="prediction-box">
            <div class="pred-label">CURRENT PRICE</div>
            <div class="pred-value">₹{curr_price:.2f}</div>
        </div>""", unsafe_allow_html=True)
    with pc2:
        st.markdown(f"""
        <div class="prediction-box">
            <div class="pred-label">NEXT DAY FORECAST ({model_name})</div>
            <div class="pred-value">₹{next_pred:.2f}</div>
            <div class="{'kpi-delta-pos' if diff >= 0 else 'kpi-delta-neg'}">{'+' if diff >= 0 else ''}{diff:.2f} ({diff_pct:+.2f}%)</div>
        </div>""", unsafe_allow_html=True)
    with pc3:
        if diff > 0:
            st.success(f"📈 BUY SIGNAL\n\nModel expects ₹{abs(diff):.2f} upside ({diff_pct:+.2f}%)")
        else:
            st.error(f"📉 SELL SIGNAL\n\nModel expects ₹{abs(diff):.2f} downside ({diff_pct:+.2f}%)")

    st.markdown("---")

    # ── Residual Analysis ──
    section_header("◈", "RESIDUAL ANALYSIS")

    residuals = actual - preds
    rc1, rc2 = st.columns(2)

    with rc1:
        fig_res = go.Figure()
        fig_res.add_trace(go.Scatter(
            x=idx, y=residuals,
            mode='lines+markers',
            line=dict(color='#00D4FF', width=1),
            marker=dict(size=2),
            name='Residual'
        ))
        fig_res.add_hline(y=0, line_dash="dash", line_color="#FF4560", line_width=1)
        fig_res.update_layout(
            **BLOOMBERG_LAYOUT,
            title='Residuals Over Time',
            xaxis_title='Date', yaxis_title='Residual (₹)',
            height=300
        )
        st.plotly_chart(fig_res, use_container_width=True)

    with rc2:
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=residuals, nbinsx=50,
            name='Residual Distribution',
            marker_color='#00D4FF', opacity=0.75
        ))
        fig_hist.update_layout(
            **BLOOMBERG_LAYOUT,
            title='Residual Distribution',
            xaxis_title='Residual (₹)', yaxis_title='Frequency',
            height=300
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    chart_help("🔍 Shows the model's mistakes over time. Dots near zero = accurate. Bell-shaped histogram = errors are balanced, not biased.")


# ============================================================
# TAB 4 — ADVANCED ANALYTICS
# ============================================================
def tab_advanced(df, raw_df):
    with st.spinner("⚙ Training models for comparison..."):
        models = train_models(id(df))

    # ── 1. Model Comparison ──
    section_header("◈", "ALL MODELS COMPARISON")

    preds_ridge = get_predictions(models, 'ridge')
    preds_rf    = get_predictions(models, 'rf')
    preds_xgb   = get_predictions(models, 'xgb')
    actual      = models['y_test'].values
    idx         = models['X_test'].index

    fig_comp = go.Figure()
    fig_comp.add_trace(go.Scatter(
        x=idx, y=actual, name='Actual',
        mode='lines', line=dict(color='#00D4FF', width=2.5)
    ))
    fig_comp.add_trace(go.Scatter(
        x=idx, y=preds_ridge, name='Ridge Regression',
        mode='lines', line=dict(color='#00FF88', width=1.5, dash='dash')
    ))
    fig_comp.add_trace(go.Scatter(
        x=idx, y=preds_xgb, name='XGBoost',
        mode='lines', line=dict(color='#FF4560', width=1.5, dash='dot')
    ))
    fig_comp.add_trace(go.Scatter(
        x=idx, y=preds_rf, name='Random Forest',
        mode='lines', line=dict(color='#74B9FF', width=1.5, dash='longdash')
    ))
    fig_comp.update_layout(
        **BLOOMBERG_LAYOUT,
        title='All Models — Actual vs Predicted',
        xaxis_title='Date', yaxis_title='Price (₹)',
        height=420, hovermode='x unified'
    )
    st.plotly_chart(fig_comp, use_container_width=True)

    # Metrics table
    all_metrics = {
        'Ridge Regression': calc_metrics(actual, preds_ridge),
        'XGBoost':          calc_metrics(actual, preds_xgb),
        'Random Forest':    calc_metrics(actual, preds_rf),
    }
    mt_df = pd.DataFrame(all_metrics, index=['RMSE (₹)', 'MAE (₹)', 'R² Score', 'MAPE (%)']).T
    mt_df = mt_df.round(4).sort_values('R² Score', ascending=False)

    section_header("◈", "MODEL PERFORMANCE SCORECARD")
    # Sort models by R² descending before rendering medals
    sorted_models = sorted(all_metrics.items(), key=lambda x: x[1][2], reverse=True)
    c1, c2, c3 = st.columns(3)
    for i, (mname, (rmse, mae, r2, mape)) in enumerate(sorted_models):
        col = [c1, c2, c3][i]
        medal = ["🥇", "🥈", "🥉"][i]
        with col:
            st.markdown(f"""
            <div class="prediction-box">
                <div class="pred-label">{medal} {mname.upper()}</div>
                <div class="pred-value" style="font-size:1.3rem">R² {r2:.4f}</div>
                <div style="font-family: 'IBM Plex Mono'; font-size:0.75rem; color:#8892A4; margin-top:8px">
                RMSE ₹{rmse:.2f} &nbsp;|&nbsp; MAE ₹{mae:.2f} &nbsp;|&nbsp; MAPE {mape:.2f}%
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── 2. Feature Importance ──
    section_header("◈", "FEATURE IMPORTANCE — XGBoost")

    importances = models['xgb'].feature_importances_
    feat_names  = models['feature_names']
    fi_df = pd.DataFrame({'Feature': feat_names, 'Importance': importances})
    fi_df = fi_df.sort_values('Importance', ascending=True).tail(20)

    fig_fi = go.Figure(go.Bar(
        x=fi_df['Importance'], y=fi_df['Feature'],
        orientation='h',
        marker=dict(
            color=fi_df['Importance'],
            colorscale=[[0, '#1A1A2E'], [0.5, '#8892A4'], [1, '#00D4FF']],
            showscale=False
        )
    ))
    fig_fi.update_layout(
        **BLOOMBERG_LAYOUT,
        title='Top 20 Feature Importances (XGBoost)',
        xaxis_title='Importance Score', yaxis_title='',
        height=500
    )
    st.plotly_chart(fig_fi, use_container_width=True)
    chart_help("🔑 Longer bars show the most important features that helped the model make predictions.")

    st.markdown("---")

    # ── 3. Returns Analysis ──
    section_header("◈", "RETURNS DISTRIBUTION ANALYSIS")

    df_copy = df.copy()
    df_copy['Daily_Return'] = df_copy['Close'].pct_change() * 100
    returns = df_copy['Daily_Return'].dropna()

    r1, r2 = st.columns(2)
    with r1:
        fig_ret = go.Figure()
        fig_ret.add_trace(go.Histogram(
            x=returns, nbinsx=80,
            name='Daily Returns',
            marker_color='#00D4FF', opacity=0.8
        ))
        # Add normal distribution overlay
        mu, sigma = returns.mean(), returns.std()
        x_norm = np.linspace(returns.min(), returns.max(), 200)
        y_norm = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_norm - mu) / sigma) ** 2)
        y_norm_scaled = y_norm * len(returns) * (returns.max() - returns.min()) / 80
        fig_ret.add_trace(go.Scatter(
            x=x_norm, y=y_norm_scaled,
            name='Normal Distribution',
            line=dict(color='#FF4560', width=2, dash='dash')
        ))
        fig_ret.update_layout(
            **BLOOMBERG_LAYOUT,
            title='Daily Returns Distribution',
            xaxis_title='Return (%)', yaxis_title='Frequency',
            height=350
        )
        st.plotly_chart(fig_ret, use_container_width=True)

    with r2:
        # Rolling volatility
        roll_vol = returns.rolling(30).std() * np.sqrt(252)
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Scatter(
            x=roll_vol.index, y=roll_vol,
            name='30D Annualised Volatility', mode='lines',
            line=dict(color='#00D4FF', width=1.8),
            fill='tozeroy', fillcolor='rgba(0,212,255,0.06)'
        ))
        fig_vol.update_layout(
            **BLOOMBERG_LAYOUT,
            title='Rolling 30-Day Annualised Volatility',
            xaxis_title='Date', yaxis_title='Volatility (%)',
            height=350
        )
        st.plotly_chart(fig_vol, use_container_width=True)

    chart_help("📉 The returns chart shows how often small or large price moves happen. The volatility chart shows risk levels, where a higher line means a more uncertain period.")

    st.markdown("---")

    # ── 4. Seasonality Analysis ──
    section_header("◈", "SEASONALITY ANALYSIS")

    df_copy['Month']  = df_copy.index.month
    df_copy['Year']   = df_copy.index.year
    df_copy['Weekday']= df_copy.index.dayofweek

    month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    day_names   = ['Mon','Tue','Wed','Thu','Fri']

    sa1, sa2 = st.columns(2)
    with sa1:
        monthly_ret = df_copy.groupby('Month')['Daily_Return'].mean()
        monthly_ret.index = [month_names[i-1] for i in monthly_ret.index]
        colors_m = ['#00FF88' if x >= 0 else '#FF4560' for x in monthly_ret.values]
        fig_month = go.Figure(go.Bar(
            x=monthly_ret.index, y=monthly_ret.values,
            marker_color=colors_m, name='Avg Daily Return'
        ))
        fig_month.update_layout(
            **BLOOMBERG_LAYOUT,
            title='Average Daily Return by Month',
            xaxis_title='Month', yaxis_title='Avg Return (%)',
            height=320
        )
        st.plotly_chart(fig_month, use_container_width=True)

    with sa2:
        dow_ret = df_copy.groupby('Weekday')['Daily_Return'].mean()
        day_map = {0:'Mon', 1:'Tue', 2:'Wed', 3:'Thu', 4:'Fri'}
        dow_ret.index = [day_map.get(i, str(i)) for i in dow_ret.index]
        colors_d = ['#00FF88' if x >= 0 else '#FF4560' for x in dow_ret.values]
        fig_day = go.Figure(go.Bar(
            x=dow_ret.index, y=dow_ret.values,
            marker_color=colors_d, name='Avg Daily Return'
        ))
        fig_day.update_layout(
            **BLOOMBERG_LAYOUT,
            title='Average Daily Return by Day of Week',
            xaxis_title='Day', yaxis_title='Avg Return (%)',
            height=320
        )
        st.plotly_chart(fig_day, use_container_width=True)

    chart_help("📅 Green bars show profitable months or weekdays on average. Win Rate above 50% means the stock moved up more often than down.")

    st.markdown("---")

    # ── 5. Risk Metrics ──
    section_header("◈", "RISK METRICS DASHBOARD")

    returns_clean = returns.dropna()
    total_return  = ((raw_df['Close'].iloc[-1] / raw_df['Close'].iloc[0]) - 1) * 100
    ann_return    = (((raw_df['Close'].iloc[-1] / raw_df['Close'].iloc[0]) ** (252 / len(raw_df))) - 1) * 100
    ann_vol       = returns_clean.std() * np.sqrt(252)
    sharpe        = (ann_return - 6.5) / ann_vol  # 6.5% risk-free rate (India)
    cum_ret       = (1 + returns_clean / 100).cumprod()
    roll_max      = cum_ret.cummax()
    drawdown      = (cum_ret - roll_max) / roll_max * 100
    max_drawdown  = drawdown.min()
    var_95        = np.percentile(returns_clean, 5)
    cvar_95       = returns_clean[returns_clean <= var_95].mean()

    rm1, rm2, rm3, rm4 = st.columns(4)
    with rm1: st.metric("Total Return",    f"{total_return:.1f}%",  delta="Since 2010")
    with rm2: st.metric("Ann. Return",     f"{ann_return:.2f}%",    delta="Per Year (CAGR)")
    with rm3: st.metric("Sharpe Ratio",    f"{sharpe:.2f}",          delta="Risk-Adj Return")
    with rm4: st.metric("Max Drawdown",    f"{max_drawdown:.2f}%",  delta="Worst Peak-to-Trough")

    rm5, rm6, rm7, rm8 = st.columns(4)
    with rm5: st.metric("Ann. Volatility", f"{ann_vol:.2f}%",        delta="Annual Risk")
    with rm6: st.metric("VaR (95%)",       f"{var_95:.2f}%",          delta="Daily Loss (95% CI)")
    with rm7: st.metric("CVaR (95%)",      f"{cvar_95:.2f}%",         delta="Expected Tail Loss")
    with rm8:
        win_rate = (returns_clean > 0).mean() * 100
        st.metric("Win Rate",          f"{win_rate:.1f}%",         delta="Days Positive")

    # Drawdown chart
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=drawdown.index, y=drawdown.values,
        name='Drawdown', mode='lines',
        line=dict(color='#FF4560', width=1.5),
        fill='tozeroy', fillcolor='rgba(255,69,96,0.08)'
    ))
    fig_dd.update_layout(
        **BLOOMBERG_LAYOUT,
        title='Drawdown Chart (from Peak)',
        xaxis_title='Date', yaxis_title='Drawdown (%)',
        height=300
    )
    st.plotly_chart(fig_dd, use_container_width=True)
    chart_help("📉 Drawdown shows how much the stock fell from its highest point during the selected period. A deeper dip means investors faced a larger loss.")

    st.markdown("---")

    # ── 6. Yearly Performance Heatmap ──
    section_header("◈", "YEARLY & MONTHLY RETURN HEATMAP")

    df_copy['Month_num'] = df_copy.index.month
    monthly_pivot = df_copy.groupby(['Year', 'Month_num'])['Daily_Return'].mean().unstack()
    monthly_pivot.columns = month_names
    monthly_pivot = monthly_pivot[monthly_pivot.index >= 2015]

    fig_heat = px.imshow(
        monthly_pivot,
        color_continuous_scale=[[0, '#FF4560'], [0.5, '#0D1117'], [1, '#00FF88']],
        aspect='auto',
        text_auto='.2f',
        labels=dict(color="Avg Return (%)")
    )
    fig_heat.update_layout(
        **BLOOMBERG_LAYOUT,
        title='Monthly Average Daily Return Heatmap (%) — 2015 to Present',
        xaxis_title='Month', yaxis_title='Year',
        height=400,
        coloraxis_colorbar=dict(
            tickfont=dict(color='#8892A4'),
            title=dict(font=dict(color='#8892A4'))
        )
    )
    fig_heat.update_traces(
        textfont=dict(family="IBM Plex Mono", size=9, color="#00D4FF")
    )
    st.plotly_chart(fig_heat, use_container_width=True)
    chart_help("🟩 Green cells show strong months, while Red cells show weaker months. This helps identify Reliance’s historical performance trends.")


# ============================================================
# MAIN APP
# ============================================================
def main():
    # ── Header ──
    st.markdown("""
    <div class="dashboard-header">
        <h1>📊 Reliance Stock Intelligence Terminal</h1>
        <p>
            <span class="live-dot"></span>
            RELIANCE.NS &nbsp;|&nbsp; NSE INDIA &nbsp;|&nbsp;
            MODELS: RIDGE · RANDOM FOREST · XGBOOST &nbsp;|&nbsp;
            DATA: YAHOO FINANCE API &nbsp;|&nbsp; BY TRIPJOT SINGH
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar ──
    st.sidebar.title("⚙ TERMINAL CONFIG")
    st.sidebar.markdown("---")

    model_choice = st.sidebar.selectbox(
        "🤖 Prediction Model",
        [
            "Ridge Regression (Best — R² 95.67%)",
            "XGBoost (R² 94.89%)",
            "Random Forest (R² 92.48%)"
        ]
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**📅 Chart Date Range**")
    start_date = st.sidebar.date_input("Start Date", value=date(2020, 1, 1))
    end_date   = st.sidebar.date_input("End Date",   value=date.today())

    st.sidebar.markdown("---")
    retrain_btn = st.sidebar.button(
        "🔄 Refresh Data & Retrain",
        use_container_width=True
    )

    st.sidebar.markdown("---")
    st.sidebar.info(
        "Reliance Industries (NSE: RELIANCE)\n\n"
        "16 years of live OHLCV data.\n"
        "30+ engineered features.\n"
        "3 ML models trained & compared."
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("⚠ For educational purposes only. Not financial advice.")

    # ── Data Load ──
    with st.spinner("⚙ Fetching market data..."):
        if retrain_btn:
            st.cache_data.clear()
        raw_df = load_data()
        df     = add_features(raw_df.copy())
        st.session_state['df_featured'] = df

    # ── Tabs ──
    tab1, tab2, tab3, tab4 = st.tabs([
        "◈ Overview",
        "◈ Technical Analysis",
        "◈ ML Predictions",
        "◈ Advanced Analytics"
    ])

    with tab1:
        tab_overview(raw_df, df, start_date, end_date)

    with tab2:
        tab_technical(df, start_date, end_date)

    with tab3:
        tab_ml(df, raw_df, model_choice)

    with tab4:
        tab_advanced(df, raw_df)

    # ── Footer ──
    st.markdown("""
    <div class="dashboard-footer">
        <p>
            📊 RELIANCE STOCK INTELLIGENCE TERMINAL &nbsp;|&nbsp;
            Built with Python · Streamlit · Plotly · Scikit-learn · XGBoost
        </p>
        <p>
            <a href="https://github.com/tripjotsingh2505/stock-market-price-prediction-dashboard">
                GitHub
            </a>
            &nbsp;|&nbsp;
            <a href="https://www.linkedin.com/in/tripjot-singh-7a75a0284">LinkedIn</a>
            &nbsp;|&nbsp; Tripjot Singh
        </p>
        <p>⚠ Educational purposes only. Not financial advice. Always do your own research.</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
