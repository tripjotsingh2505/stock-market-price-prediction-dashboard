# 📈 Reliance Industries Stock Price Prediction Dashboard

![Python](https://img.shields.io/badge/Python-3.11+-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red?style=for-the-badge&logo=streamlit)
![ML](https://img.shields.io/badge/Machine%20Learning-Ridge%20|%20RF%20|%20XGBoost-green?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Live-brightgreen?style=for-the-badge)
![Data](https://img.shields.io/badge/Data-Yahoo%20Finance%20API-purple?style=for-the-badge)

---

## 🌐 Live Demo
> 🚀 **[Click here to view the Live Dashboard](#)** ← Deploy hone ke baad link daalna

---

## 📌 Project Overview

> An **end-to-end industry-level Data Science project** that predicts Reliance Industries (NSE: RELIANCE) stock prices using advanced Machine Learning models trained on **16 years of live market data** (2010-2026) fetched in real-time from Yahoo Finance API.

### 🎯 Problem Statement
> Stock markets are highly volatile and unpredictable. This project aims to build a robust ML pipeline that can accurately predict the next trading day's closing price of Reliance Industries using historical price data and technical indicators.

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| **Language** | Python 3.11+ |
| **Dashboard** | Streamlit |
| **ML Models** | Ridge Regression, Random Forest, XGBoost |
| **Data Source** | Yahoo Finance API (yfinance) |
| **Visualization** | Plotly |
| **Data Processing** | Pandas, NumPy |
| **ML Library** | Scikit-learn |
| **Deployment** | Render |

---

## 📊 Project Architecture

```
📦 Stock Price Prediction Pipeline
│
├── 📥 Data Collection (Yahoo Finance API)
│   └── Live OHLCV data — 2010 to Today
│
├── 🧹 Data Cleaning
│   └── Missing values, duplicates, timezone fix
│
├── 🔍 Exploratory Data Analysis
│   └── Business insights, patterns, trends
│
├── 📈 Visualization
│   └── Candlestick, RSI, MACD, Volume charts
│
├── ⚙️ Feature Engineering (30+ features)
│   ├── Moving Averages (MA7, MA21, MA50, MA200)
│   ├── RSI, MACD, Bollinger Bands
│   ├── Lag Features, Date Features
│   └── ATR, OBV, Price Range
│
├── 🔄 Preprocessing
│   └── MinMax Scaling, Train/Test Split (80/20)
│
├── 🤖 Model Building & Training
│   ├── Ridge Regression ← Best Model
│   ├── Random Forest
│   └── XGBoost
│
├── 📊 Model Evaluation
│   └── RMSE, MAE, R², MAPE, Cross Validation
│
└── 🚀 Deployment
    └── Live Streamlit Dashboard on Render
```

---

## 🤖 ML Models Performance

| Model | R² Score | RMSE | MAE | MAPE |
|-------|----------|------|-----|------|
| 🥇 **Ridge Regression** | **0.9567** | **₹18.87** | **₹14.10** | **1.00%** |
| 🥈 XGBoost (Tuned) | 0.9489 | ₹20.50 | ₹16.20 | 1.14% |
| 🥉 Random Forest | 0.9248 | ₹24.86 | ₹19.63 | 1.39% |

> 💡 **Ridge Regression** outperformed complex models because Reliance stock follows a predominantly **linear trend** over time.

---

## 📈 Key Features of Dashboard

| Feature | Description |
|---------|-------------|
| 📊 **Live Market Data** | Real-time price fetched from Yahoo Finance |
| 🕯️ **Candlestick Chart** | Interactive chart with MA50 & MA200 |
| 📉 **RSI Indicator** | Overbought/Oversold signals |
| 📈 **MACD Indicator** | Momentum & trend signals |
| 🤖 **Model Selection** | Choose between 3 ML models |
| 🔮 **Next Day Prediction** | Predict tomorrow's closing price |
| 🚦 **Buy/Sell Signals** | Automated trading signals |
| 🔄 **Auto Retrain** | One-click model retraining on latest data |

---

## 💡 Key Business Insights

- 📈 Reliance stock grew **4x** from 2020 to 2024 (₹395 → ₹1600)
- 📉 **COVID crash** (March 2020): Price hit ₹395 — all-time low in dataset
- 📅 **Best months**: July-September (historically highest average prices)
- 📅 **Worst months**: March-April (historically lowest average prices)
- 📊 **April 22, 2020**: Highest ever trading volume — 14.26 Crore shares
- 🎯 Ridge Regression achieved **1% MAPE** — only ₹14 error on ₹1400 stock!

---

## 📁 Project Structure

```
stock_prediction_project/
│
├── 📂 data/
│   └── reliance_featured_data.csv
│
├── 📂 notebooks/
│   ├── day1_setup.ipynb
│   ├── day2_cleaning.ipynb
│   ├── day3_eda.ipynb
│   ├── day4_visualization.ipynb
│   ├── day5_feature_engineering.ipynb
│   ├── day6_preprocessing.ipynb
│   ├── day7_model_building.ipynb
│   ├── day8_model_evaluation.ipynb
│   └── day9_model_improvement.ipynb
│
├── 📂 src/
│   └── app.py                 ← Streamlit Dashboard
│
├── 📂 models/
│   ├── ridge_model.pkl
│   ├── rf_model.pkl
│   ├── xgb_model.pkl
│   ├── scaler_X.pkl
│   └── scaler_y.pkl
│
├── 📂 reports/
│   └── charts/                ← Visualizations
│
├── requirements.txt
└── README.md
```

---

## 🔧 How to Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/stock-prediction.git
cd stock-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the dashboard
streamlit run src/app.py
```

---

## 📸 Dashboard Screenshots

> *(Add screenshots after deployment)*

---

## ⚠️ Disclaimer

> This project is built for **educational purposes only**. The predictions made by this dashboard should **not** be considered as financial advice. Always consult a certified financial advisor before making investment decisions.

---

## 👨‍💻 Author

**Your Name**
- 🔗 LinkedIn: [Your LinkedIn](#)
- 🐙 GitHub: [Your GitHub](#)
- 📧 Email: your.email@gmail.com

---

<div align="center">
  <b>⭐ If you found this project helpful, please give it a star! ⭐</b>
</div>
