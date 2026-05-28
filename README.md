<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=IBM+Plex+Mono&weight=600&size=28&pause=1000&color=00D4FF&center=true&vCenter=true&width=700&lines=Reliance+Stock+Intelligence+Terminal;Live+ML-Powered+Price+Prediction;End-to-End+Data+Science+Pipeline" alt="Typing SVG" />

<br/>

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-189AB4?style=for-the-badge)
![Plotly](https://img.shields.io/badge/Plotly-5.18+-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![Yahoo Finance](https://img.shields.io/badge/Yahoo%20Finance-Live%20API-6001D2?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Live-00C853?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-FFD600?style=for-the-badge)

<br/>

> **An end-to-end industry-level Data Science project** — live stock price prediction dashboard for Reliance Industries (NSE: RELIANCE), powered by Machine Learning, real-time Yahoo Finance data, and advanced technical analysis across 16 years of market history.

<br/>

🚀 **[View Live Dashboard](#)** &nbsp;&nbsp;|&nbsp;&nbsp; 📓 **[Explore Notebooks](notebooks/)** &nbsp;&nbsp;|&nbsp;&nbsp; ⭐ **Star this repo if you found it useful!**

</div>

---

## 📸 Dashboard Preview

<div align="center">

| 📊 Overview — Live Market Snapshot | 🤖 ML Predictions — Actual vs Predicted |
|:---:|:---:|
| ![Price Chart](reports/charts/priceChart.png) | ![Actual vs Predicted](reports/charts/actualVsPredicted.png) |

| 📈 Model Comparison — RMSE · MAE · R² | 📉 RSI — Momentum Indicator |
|:---:|:---:|
| ![Model Comparison](reports/charts/allModelComparison.png) | ![RSI](reports/charts/RSI14Period.png) |

</div>

---

## 🎯 Problem Statement

> Stock markets are inherently volatile and difficult to forecast. This project builds a **robust, scalable ML pipeline** that accurately predicts the next trading day's closing price of Reliance Industries — trained on 16 years of live market data with 30+ engineered technical features, and served through a beautiful real-time dashboard.

---

## 🏗️ Project Architecture

```mermaid
flowchart TD
    A[📥 Yahoo Finance API\nLive OHLCV — 2010 to Today] --> B[🧹 Data Cleaning\nMissing values · Duplicates · Timezone fix]
    B --> C[🔍 EDA\nTrends · Seasonality · Volume · Return Analysis]
    C --> D[⚙️ Feature Engineering\n30+ Technical Indicators]
    D --> E[🔄 Preprocessing\nMinMax Scaling · Train/Test Split 80/20]
    E --> F1[🔵 Ridge Regression\nR² 0.9567 · MAPE 1.00%]
    E --> F2[🟠 Random Forest\nR² 0.9248 · MAPE 1.39%]
    E --> F3[🟣 XGBoost Tuned\nR² 0.9489 · MAPE 1.14%]
    F1 --> G[📊 Model Evaluation\nRMSE · MAE · R² · MAPE · Residuals]
    F2 --> G
    F3 --> G
    G --> H[🚀 Live Streamlit Dashboard\nDeployed on Render]

    style A fill:#0A0E1A,color:#00D4FF,stroke:#00D4FF
    style H fill:#0A0E1A,color:#00FF88,stroke:#00FF88
    style F1 fill:#0A0E1A,color:#00D4FF,stroke:#00D4FF
    style F2 fill:#0A0E1A,color:#FF6B35,stroke:#FF6B35
    style F3 fill:#0A0E1A,color:#A855F7,stroke:#A855F7
```

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Language** | Python 3.11+ | Core development |
| **Dashboard & UI** | Streamlit + Custom CSS | Midnight Blue terminal theme |
| **Data Source** | Yahoo Finance API (`yfinance`) | Live OHLCV market data |
| **Visualisation** | Plotly (interactive) | All charts & indicators |
| **ML Models** | Ridge · Random Forest · XGBoost | Price prediction |
| **Data Processing** | Pandas · NumPy | Feature engineering & preprocessing |
| **ML Library** | Scikit-learn · XGBoost | Model training & evaluation |
| **Deployment** | Render | Cloud hosting |

---

## 🤖 ML Model Performance

| Rank | Model | R² Score | RMSE | MAE | MAPE |
|:----:|-------|:--------:|:----:|:---:|:----:|
| 🥇 | **Ridge Regression** | **0.9567** | **₹18.87** | **₹14.10** | **1.00%** |
| 🥈 | XGBoost (Tuned) | 0.9489 | ₹20.50 | ₹16.20 | 1.14% |
| 🥉 | Random Forest | 0.9248 | ₹24.86 | ₹19.63 | 1.39% |

> 💡 **Why Ridge Regression won?** Reliance stock follows a predominantly linear long-term trend. Ridge's L2 regularisation prevents overfitting while maintaining high accuracy — outperforming complex tree-based models on this dataset.

---

## ⚙️ Feature Engineering — 30+ Technical Indicators

```
Moving Averages    →  MA7 · MA21 · MA50 · MA200
Momentum           →  RSI (14-period)
Trend              →  MACD · Signal Line · Histogram
Volatility         →  Bollinger Bands (Upper · Middle · Lower) · ATR · BB Width
Volume             →  OBV (On-Balance Volume)
Lag Features       →  Close Lag 1 · 2 · 3 · 5
Date Features      →  Year · Month · Day · Day-of-Week · Quarter
Price Features     →  Daily Return · Price Range · Price Change
```

---

## 🖥️ Dashboard — 4 Tabs

| Tab | Features |
|-----|---------|
| **◈ Overview** | Live KPIs · Candlestick / Line chart · Volume & OBV · Buy/Sell Signals |
| **◈ Technical Analysis** | RSI · MACD · Bollinger Bands — all interactive with date range selector |
| **◈ ML Predictions** | Model selector · Actual vs Predicted · Residual analysis · Feature importance · Next-day forecast |
| **◈ Advanced Analytics** | Risk metrics (Sharpe · VaR · CVaR) · Drawdown · Return distribution · Seasonality heatmap |

---

## 💡 Key Business Insights

| # | Insight |
|---|---------|
| 📈 | Reliance grew **4×** from ₹395 to ₹1,600+ between 2020 and 2024 |
| 📉 | COVID crash (March 2020) — all-time low of **₹395**, a ~45% drawdown from peak |
| 📅 | **July–September** are historically the strongest months for Reliance |
| 📅 | **March–April** are historically the weakest months |
| 📊 | **22 April 2020** — peak trading volume of 14.26 Crore shares in a single day |
| 🎯 | Ridge Regression achieves **1% MAPE** — just ₹14 error on a ₹1,400 stock |
| 📐 | Yesterday's closing price accounts for **~75% of feature importance** |

---

## 📁 Project Structure

```
Stock Market Price Prediction & Analysis System/
│
├── 📂 src/
│   └── app.py                          ← Streamlit dashboard (Midnight Blue Edition)
│
├── 📂 notebooks/
│   ├── day1_setup.ipynb                ← Data collection & environment setup
│   ├── day2_cleaning.ipynb             ← Data cleaning & validation
│   ├── day3_eda.ipynb                  ← Exploratory data analysis
│   ├── day4_visualisation.ipynb        ← Interactive Plotly charts
│   ├── day5_feature_engineering.ipynb  ← 30+ technical indicators
│   ├── day6_preprocessing.ipynb        ← Scaling & train/test split
│   ├── day7_model_building.ipynb       ← Model training & comparison
│   ├── day8_model_evaluation.ipynb     ← RMSE · MAE · R² · MAPE · residuals
│   └── day9_model_improvement.ipynb    ← Cross-validation & hyperparameter tuning
│
├── 📂 models/
│   ├── ridge_model.pkl                 ← Best model (R² 0.9567)
│   ├── rf_model.pkl
│   ├── xgb_model.pkl
│   ├── scaler_X.pkl
│   └── scaler_y.pkl
│
├── 📂 data/
│   └── reliance_featured_data.csv      ← Engineered feature dataset
│
├── 📂 reports/charts/                  ← All exported visualisation charts
│
├── requirements.txt
├── Procfile                            ← Render deployment config
├── runtime.txt
└── README.md
```

---

## 🚀 How to Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/tripjotsingh2505/stock-market-price-prediction.git
cd "Stock Market Price Prediction & Analysis System"

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the dashboard
streamlit run src/app.py
```

> Opens at `http://localhost:8501` — live data fetched automatically from Yahoo Finance.

---

## 🌐 Deployment

Deployed on **Render** as a persistent web service — no sleep mode, always live.

| File | Purpose |
|------|---------|
| `Procfile` | Tells Render how to start the Streamlit app |
| `runtime.txt` | Pins the Python version |
| `requirements.txt` | All package dependencies with minimum versions |

---

## ⚠️ Disclaimer

> This project is built entirely for **educational and portfolio purposes**. The predictions and signals generated by this dashboard do **not** constitute financial advice. Always conduct your own research and consult a certified financial advisor before making any investment decisions.

---

<div align="center">

## 👨‍💻 Author

**Tripjot Singh**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Tripjot%20Singh-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/tripjot-singh-7a75a0284)
[![GitHub](https://img.shields.io/badge/GitHub-tripjotsingh2505-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/tripjotsingh2505)
[![Email](https://img.shields.io/badge/Email-tripjotsingh25%40gmail.com-EA4335?style=for-the-badge&logo=gmail&logoColor=white)](mailto:tripjotsingh25@gmail.com)

<br/>

---

*Built with 🤍 using Python · Streamlit · Machine Learning · Yahoo Finance API*

⭐ **If this project helped you, please consider giving it a star!** ⭐

</div>
