import pandas as pd
import streamlit as st

from model import FEATURES, load_data, predict_price, train_model

CURRENCY = "$"

st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="centered")

# ---------- Minimal dark theme ----------
st.markdown("""
<style>
#MainMenu, footer, header {visibility: hidden;}
.stApp {background: #0d0f14; color: #e8eaf0;}
.block-container {max-width: 720px; padding-top: 3rem;}
h1 {font-weight: 700; letter-spacing: -0.5px; margin-bottom: 0;}
.sub {color: #8a90a2; margin-bottom: 2rem;}
.result {
    background: linear-gradient(135deg, #6d5dfc 0%, #c850c0 100%);
    border-radius: 20px; padding: 2rem; text-align: center; margin: 1.5rem 0;
    box-shadow: 0 10px 40px rgba(109, 93, 252, 0.25);
}
.result .label {font-size: 0.8rem; letter-spacing: 2px; opacity: 0.8; text-transform: uppercase;}
.result .price {font-size: 3rem; font-weight: 800; margin-top: 0.25rem;}
.chip {
    background: #161a23; border: 1px solid #232838; border-radius: 14px;
    padding: 0.9rem; text-align: center;
}
.chip .k {color: #8a90a2; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1px;}
.chip .v {font-size: 1.25rem; font-weight: 700; margin-top: 2px;}
div[data-testid="stSlider"] label, div[data-testid="stNumberInput"] label {color: #b7bccb;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_model():
    df = load_data()
    model, metrics = train_model(df)
    return df, model, metrics


df, model, metrics = get_model()

# ---------- Header ----------
st.markdown("# 🏠 House Price Predictor")
st.markdown('<div class="sub">Linear regression, trained on '
            f'{len(df):,} homes. Adjust the details to estimate a price.</div>',
            unsafe_allow_html=True)

# ---------- Inputs ----------
c1, c2 = st.columns(2)
with c1:
    area = st.slider("Area (sq ft)", 500, 4000, 1800, step=50)
    bedrooms = st.slider("Bedrooms", 1, 6, 3)
    bathrooms = st.slider("Bathrooms", 1, 5, 2)
with c2:
    age = st.slider("Age of house (years)", 0, 50, 10)
    distance = st.slider("Distance to city centre (km)", 1.0, 30.0, 8.0, step=0.5)
    garage = st.slider("Garage spaces", 0, 2, 1)

# ---------- Prediction ----------
price = predict_price(
    model, area_sqft=area, bedrooms=bedrooms, bathrooms=bathrooms,
    age_years=age, distance_km=distance, garage=garage,
)

st.markdown(
    f'<div class="result"><div class="label">Estimated price</div>'
    f'<div class="price">{CURRENCY}{price:,.0f}</div></div>',
    unsafe_allow_html=True,
)

# ---------- Model quality ----------
m1, m2, m3 = st.columns(3)
for col, (name, val) in zip(
    (m1, m2, m3),
    [("R² score", f"{metrics['R2']:.3f}"),
     ("Avg. error", f"{CURRENCY}{metrics['MAE']:,.0f}"),
     ("RMSE", f"{CURRENCY}{metrics['RMSE']:,.0f}")],
):
    col.markdown(f'<div class="chip"><div class="k">{name}</div><div class="v">{val}</div></div>',
                 unsafe_allow_html=True)

# ---------- What drives the price ----------
with st.expander("What drives the price?"):
    coefs = pd.DataFrame({"feature": FEATURES, "price impact per unit": model.coef_}).set_index("feature")
    st.bar_chart(coefs, color="#8b7bff")
    st.caption("Each bar shows how much the price changes when that feature increases by one unit.")

with st.expander("Peek at the dataset"):
    st.dataframe(df.head(10), width="stretch")
