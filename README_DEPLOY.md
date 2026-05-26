# Deployment Notes

## Required files in GitHub repo root
- app.py
- requirements.txt
- Procfile
- runtime.txt

## Render settings
Build Command:
pip install -r requirements.txt

Start Command:
streamlit run app.py --server.address 0.0.0.0 --server.port $PORT

## Important
The app uses Yahoo Finance live data. For extra safety, also upload:
data/reliance_featured_data.csv

This file is used only if Yahoo Finance is unavailable.
