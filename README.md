# House Price Predictor

A small machine learning project: predict house prices with **linear regression** (scikit-learn) and a minimal dark UI (Streamlit).

## Run it
```bash
pip install -r requirements.txt
streamlit run app.py
```
`houses.csv` is generated automatically on first run. To use your own data, replace it with a CSV that has the columns:
`area_sqft, bedrooms, bathrooms, age_years, distance_km, garage, price`.

## Files
- `generate_data.py`: builds the synthetic dataset
- `model.py`: loads data, trains and evaluates the model (run `python model.py` for a text report)
- `app.py`: the Streamlit interface
