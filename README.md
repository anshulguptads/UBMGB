# Universal Bank - Personal Loan Prediction (Streamlit App)

This repo contains a Streamlit dashboard to:
1. Explore **top 5 marketing insights** with charts.
2. Train & evaluate **Decision Tree**, **Random Forest**, and **Gradient Boosted Trees** (80/20 split) and show **accuracy, precision, recall, F1** plus a **combined ROC** curve.
3. **Predict** on new customer data, view predictions in the app, and **download** the dataset with predicted labels.

## Quick Start (Streamlit Community Cloud)
1. Push these files to a **new public GitHub repo**.
2. On Streamlit Cloud, create a new app and point it to `app.py` on your repo's main branch.
3. The app will install requirements and launch automatically.

## Local Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Data
- CSV at `data/UniversalBank.csv` is included.
- Target: `Personal Loan` (1 = yes, 0 = no).
- We drop **ID** by default. Use the sidebar to optionally drop `Zip code` from features.
