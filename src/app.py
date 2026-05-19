# ============================================
# RELIANCE STOCK PRICE PREDICTION DASHBOARD
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pickle
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from datetime import date, timedelta
import warnings
warnings.filterwarnings('ignore')

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Reliance Stock Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================
# DATA LOADING FUNCTION
# ============================================
@st.cache_data(ttl=3600)
def load_data():
    try:
        ticker = yf.Ticker("RELIANCE.NS")

        df = ticker.history(
            start="2010-01-01",
            end=date.today().strftime("%Y-%m-%d")
        )

        if df.empty:
            raise Exception("Empty dataframe")

        df.reset_index(inplace=True)
        return df

    except Exception:
        st.warning("Yahoo Finance rate limited. Loading local dataset.")

        df = pd.read_csv("data/reliance_featured_data.csv")

        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])

        return df

# ============================================
# FEATURE ENGINEERING FUNCTION
# ============================================
def add_features(df):
    # Moving Averages
    df['MA_7']   = df['Close'].rolling(window=7).mean()
    df['MA_21']  = df['Close'].rolling(window=21).mean()
    df['MA_50']  = df['Close'].rolling(window=50).mean()
    df['MA_200'] = df['Close'].rolling(window=200).mean()

    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']

    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    df['BB_Upper']  = df['BB_Middle'] + (df['Close'].rolling(window=20).std() * 2)
    df['BB_Lower']  = df['BB_Middle'] - (df['Close'].rolling(window=20).std() * 2)

    # Lag Features
    df['Close_Lag1'] = df['Close'].shift(1)
    df['Close_Lag2'] = df['Close'].shift(2)
    df['Close_Lag3'] = df['Close'].shift(3)
    df['Close_Lag5'] = df['Close'].shift(5)

    # Date Features
    df['Year']        = df.index.year
    df['Month']       = df.index.month
    df['Day']         = df.index.day
    df['Day_of_Week'] = df.index.dayofweek
    df['Quarter']     = df.index.quarter

    # Extra Features
    df['Daily_Return'] = df['Close'].pct_change() * 100
    df['ATR']          = (df['High'] - df['Low']).rolling(window=14).mean()
    df['Price_Range']  = df['High'] - df['Low']
    df['Price_Change'] = df['Close'] - df['Open']

    # OBV
    obv = [0]
    for i in range(1, len(df)):
        if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
            obv.append(obv[-1] + df['Volume'].iloc[i])
        elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
            obv.append(obv[-1] - df['Volume'].iloc[i])
        else:
            obv.append(obv[-1])
    df['OBV'] = obv

    df = df.dropna()
    return df



# ============================================
# MODEL TRAINING FUNCTION
# ============================================
def train_models(df):
    # Target variable
    df['Target'] = df['Close'].shift(-1)
    df = df.dropna()

    # Features aur Target
    X = df.drop(columns=['Target'])
    y = df['Target']

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        shuffle=False
    )

    # Scaling
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(
        y_train.values.reshape(-1, 1)
    ).ravel()

    X_test_scaled = scaler_X.transform(X_test)

    # 1. Ridge Regression
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_scaled, y_train_scaled)

    # 2. Random Forest
    rf = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train_scaled, y_train_scaled)

    # 3. XGBoost
    xgb_model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(X_train_scaled, y_train_scaled)

    return {
        'ridge': ridge,
        'rf': rf,
        'xgb': xgb_model,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'X_test': X_test,
        'y_test': y_test,
        'X_test_scaled': X_test_scaled,
        'feature_names': X.columns.tolist()
    }


# ============================================
# MAIN DASHBOARD
# ============================================
def main():
    # Header
    st.title("📈 Reliance Industries Stock Price Prediction Dashboard")
    st.markdown("**Live data powered by Yahoo Finance | ML Models: Ridge Regression, Random Forest, XGBoost**")
    st.markdown("---")

    # ============================================
    # SIDEBAR
    # ============================================
    st.sidebar.title("⚙️ Configuration")
    st.sidebar.markdown("---")

    # Model Selection
    model_choice = st.sidebar.selectbox(
        "🤖 Select Prediction Model:",
        ["Ridge Regression (Best - R² 95.67%)",
         "Random Forest (R² 92.48%)",
         "XGBoost (R² 94.89%)"]
    )

    # Date Range
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📅 Chart Date Range")
    start_date = st.sidebar.date_input(
        "Start Date",
        value=date(2020, 1, 1)
    )
    end_date = st.sidebar.date_input(
        "End Date",
        value=date.today()
    )

    # Retrain Button
    st.sidebar.markdown("---")
    retrain_btn = st.sidebar.button(
        "🔄 Update Latest Data & Retrain Model",
        use_container_width=True
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 About")
    st.sidebar.info(
        "This dashboard predicts Reliance Industries "
        "stock prices using advanced Machine Learning models "
        "trained on live market data fetched from Yahoo Finance."
    )

    # ============================================
    # DATA LOAD
    # ============================================
    with st.spinner("📊 Fetching live market data..."):
        if retrain_btn:
            st.cache_data.clear()
        raw_df = load_data()
        df = add_features(raw_df.copy())

    # ============================================
    # LIVE METRICS — TOP CARDS
    # ============================================
    st.subheader("📊 Live Market Overview")

    latest = raw_df.iloc[-1]
    prev   = raw_df.iloc[-2]
    change = latest['Close'] - prev['Close']
    change_pct = (change / prev['Close']) * 100

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("💰 Current Price",
                  f"₹{latest['Close']:.2f}",
                  f"{change:+.2f} ({change_pct:+.2f}%)")
    with col2:
        st.metric("📈 Day High", f"₹{latest['High']:.2f}")
    with col3:
        st.metric("📉 Day Low", f"₹{latest['Low']:.2f}")
    with col4:
        st.metric("🔓 Open Price", f"₹{latest['Open']:.2f}")
    with col5:
        st.metric("📦 Volume", f"{latest['Volume']:,.0f}")

    st.markdown("---")


# ============================================
    # PRICE CHART
    # ============================================
    st.subheader("📈 Stock Price History")

    # Date filter
    filtered_df = raw_df[(raw_df.index >= pd.Timestamp(start_date)) &
                         (raw_df.index <= pd.Timestamp(end_date))]

    fig_price = go.Figure()

    # Candlestick
    fig_price.add_trace(go.Candlestick(
        x=filtered_df.index,
        open=filtered_df['Open'],
        high=filtered_df['High'],
        low=filtered_df['Low'],
        close=filtered_df['Close'],
        increasing_line_color='#26a69a',
        decreasing_line_color='#ef5350',
        name='Price'
    ))

    # Moving Averages
    df_filtered = df[(df.index >= pd.Timestamp(start_date)) &
                     (df.index <= pd.Timestamp(end_date))]

    fig_price.add_trace(go.Scatter(
        x=df_filtered.index, y=df_filtered['MA_50'],
        name='MA 50', line=dict(color='#FFE66D', width=1.5)
    ))
    fig_price.add_trace(go.Scatter(
        x=df_filtered.index, y=df_filtered['MA_200'],
        name='MA 200', line=dict(color='#A29BFE', width=1.5)
    ))

    fig_price.update_layout(
        title='Reliance Industries - Candlestick Chart with Moving Averages',
        xaxis_title='Date',
        yaxis_title='Price (₹)',
        template='plotly_dark',
        height=500,
        hovermode='x unified',
        xaxis_rangeslider_visible=False
    )

    st.plotly_chart(fig_price, use_container_width=True)

    # ============================================
    # TECHNICAL INDICATORS
    # ============================================
    st.subheader("📊 Technical Indicators")

    col1, col2 = st.columns(2)

    with col1:
        # RSI Chart
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(
            x=df_filtered.index, y=df_filtered['RSI'],
            name='RSI', line=dict(color='#FF6B6B', width=1.5)
        ))
        fig_rsi.add_hline(y=70, line_dash="dash",
                          line_color="#ef5350",
                          annotation_text="Overbought (70)")
        fig_rsi.add_hline(y=30, line_dash="dash",
                          line_color="#26a69a",
                          annotation_text="Oversold (30)")
        fig_rsi.update_layout(
            title='RSI Indicator',
            template='plotly_dark',
            height=300
        )
        st.plotly_chart(fig_rsi, use_container_width=True)

    with col2:
        # MACD Chart
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(
            x=df_filtered.index, y=df_filtered['MACD'],
            name='MACD', line=dict(color='#4ECDC4', width=1.5)
        ))
        fig_macd.add_trace(go.Scatter(
            x=df_filtered.index, y=df_filtered['MACD_Signal'],
            name='Signal', line=dict(color='#FF6B6B', width=1.5)
        ))
        fig_macd.add_trace(go.Bar(
            x=df_filtered.index,
            y=df_filtered['MACD_Histogram'],
            name='Histogram',
            marker_color=df_filtered['MACD_Histogram'].apply(
                lambda x: '#26a69a' if x >= 0 else '#ef5350'
            )
        ))
        fig_macd.update_layout(
            title='MACD Indicator',
            template='plotly_dark',
            height=300
        )
        st.plotly_chart(fig_macd, use_container_width=True)

    st.markdown("---")

# ============================================
    # MODEL TRAINING & PREDICTION
    # ============================================
    st.subheader("🤖 Machine Learning Prediction")

    with st.spinner("🔄 Training ML models on latest data..."):
        models = train_models(df.copy())

    # Model select karo
    if "Ridge" in model_choice:
        selected_model = models['ridge']
        model_name = "Ridge Regression"
    elif "Random Forest" in model_choice:
        selected_model = models['rf']
        model_name = "Random Forest"
    else:
        selected_model = models['xgb']
        model_name = "XGBoost"

    # Predictions
    pred_scaled = selected_model.predict(models['X_test_scaled'])
    predictions = models['scaler_y'].inverse_transform(
        pred_scaled.reshape(-1, 1)
    ).ravel()

    # Metrics
    rmse = np.sqrt(mean_squared_error(models['y_test'], predictions))
    mae  = mean_absolute_error(models['y_test'], predictions)
    r2   = r2_score(models['y_test'], predictions)
    mape = np.mean(np.abs((models['y_test'] - predictions) / models['y_test'])) * 100

    # Metrics Cards
    st.markdown(f"### 📊 {model_name} — Model Performance")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("R² Score", f"{r2:.4f}")
    with col2:
        st.metric("RMSE", f"₹{rmse:.2f}")
    with col3:
        st.metric("MAE", f"₹{mae:.2f}")
    with col4:
        st.metric("MAPE", f"{mape:.2f}%")

    st.markdown("---")

    # ============================================
    # ACTUAL VS PREDICTED CHART
    # ============================================
    st.subheader("📈 Actual vs Predicted Price")

    fig_pred = go.Figure()

    fig_pred.add_trace(go.Scatter(
        x=models['X_test'].index,
        y=models['y_test'],
        name='Actual Price',
        line=dict(color='white', width=2)
    ))

    fig_pred.add_trace(go.Scatter(
        x=models['X_test'].index,
        y=predictions,
        name=f'Predicted ({model_name})',
        line=dict(color='#FF6B6B', width=1.5)
    ))

    fig_pred.update_layout(
        title=f'Actual vs Predicted Price — {model_name}',
        xaxis_title='Date',
        yaxis_title='Price (₹)',
        template='plotly_dark',
        height=400,
        hovermode='x unified'
    )

    st.plotly_chart(fig_pred, use_container_width=True)

    st.markdown("---")

    # ============================================
    # NEXT DAY PREDICTION
    # ============================================
    st.subheader("🔮 Next Trading Day Prediction")

    # Latest data se predict karo
    latest_features = df[models['feature_names']].iloc[-1].values.reshape(1, -1)
    latest_scaled   = models['scaler_X'].transform(latest_features)
    next_day_pred   = models['scaler_y'].inverse_transform(
        selected_model.predict(latest_scaled).reshape(-1, 1)
    )[0][0]

    current_price = raw_df['Close'].iloc[-1]
    price_diff    = next_day_pred - current_price
    price_diff_pct = (price_diff / current_price) * 100

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "📅 Current Price",
            f"₹{current_price:.2f}"
        )
    with col2:
        st.metric(
            "🔮 Next Day Prediction",
            f"₹{next_day_pred:.2f}",
            f"{price_diff:+.2f} ({price_diff_pct:+.2f}%)"
        )
    with col3:
        if price_diff > 0:
            st.success("📈 Signal: BUY — Price Expected to Rise!")
        else:
            st.error("📉 Signal: SELL — Price Expected to Fall!")

    st.markdown("---")


# ============================================
    # BUY/SELL SIGNALS
    # ============================================
    st.subheader("🚦 Technical Buy/Sell Signals")

    latest_data = df.iloc[-1]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### RSI Signal")
        rsi_val = latest_data['RSI']
        if rsi_val > 70:
            st.error(f"🔴 OVERBOUGHT — RSI: {rsi_val:.2f}")
            st.markdown("Price may fall soon. Consider selling.")
        elif rsi_val < 30:
            st.success(f"🟢 OVERSOLD — RSI: {rsi_val:.2f}")
            st.markdown("Price may rise soon. Consider buying.")
        else:
            st.info(f"🟡 NEUTRAL — RSI: {rsi_val:.2f}")
            st.markdown("No clear signal. Hold position.")

    with col2:
        st.markdown("### MACD Signal")
        macd_val = latest_data['MACD']
        signal_val = latest_data['MACD_Signal']
        if macd_val > signal_val:
            st.success("🟢 BULLISH — MACD above Signal")
            st.markdown("Positive momentum. Uptrend expected.")
        else:
            st.error("🔴 BEARISH — MACD below Signal")
            st.markdown("Negative momentum. Downtrend expected.")

    with col3:
        st.markdown("### Moving Average Signal")
        ma50  = latest_data['MA_50']
        ma200 = latest_data['MA_200']
        close = latest_data['Close']
        if close > ma50 > ma200:
            st.success("🟢 STRONG BULLISH — Golden Cross")
            st.markdown("Price above MA50 & MA200. Strong uptrend.")
        elif close < ma50 < ma200:
            st.error("🔴 STRONG BEARISH — Death Cross")
            st.markdown("Price below MA50 & MA200. Strong downtrend.")
        else:
            st.info("🟡 MIXED SIGNALS")
            st.markdown("No clear trend. Monitor closely.")

    st.markdown("---")

    # ============================================
    # VOLUME ANALYSIS
    # ============================================
    st.subheader("📊 Volume Analysis")

    fig_vol = go.Figure()

    fig_vol.add_trace(go.Bar(
        x=df_filtered.index,
        y=df_filtered['Volume'],
        name='Volume',
        marker_color='#7C83FD',
        opacity=0.8
    ))

    fig_vol.update_layout(
        title='Daily Trading Volume',
        xaxis_title='Date',
        yaxis_title='Volume (Shares)',
        template='plotly_dark',
        height=300
    )

    st.plotly_chart(fig_vol, use_container_width=True)

    st.markdown("---")

    # ============================================
    # FOOTER
    # ============================================
    st.markdown("""
    <div style='text-align: center; color: gray; padding: 20px;'>
        <p>📈 Reliance Industries Stock Prediction Dashboard</p>
        <p>Built with Python | Streamlit | Machine Learning | Yahoo Finance API</p>
        <p>⚠️ Disclaimer: This dashboard is for educational purposes only.
        Not financial advice. Always do your own research before investing.</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================
# RUN APP
# ============================================
if __name__ == "__main__":
    main()


