# RetailPulse — Retail Sales Forecasting Website

## Run locally
1. Install Python 3.10 or newer.
2. Extract the ZIP and open the folder in VS Code.
3. In the terminal run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed in the terminal, usually `http://localhost:8501`.

## Features
- Overview dashboard: sales KPIs, trend, category mix, monthly sales.
- Forecast lab: select store/category and forecast 7–90 days.
- Random Forest regression with lag and calendar features.
- Chronological holdout evaluation using MAE, RMSE and WAPE.
- Actual-vs-predicted chart and forecast CSV download.
- Data explorer with filters and CSV export.

## Dataset
`data/retail_sales.csv` is synthetic demonstration data, not real business records. Replace it with appropriately licensed historical data before drawing real-world conclusions. Forecasts do not account for promotions, price changes, stockouts or special-event calendars; multi-step error can accumulate.
