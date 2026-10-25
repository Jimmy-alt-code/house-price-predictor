from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

DATASET = Path(__file__).parent / "houses.csv"
FEATURES = ["area_sqft", "bedrooms", "bathrooms", "age_years", "distance_km", "garage"]
TARGET = "price"


def load_data():
    if not DATASET.exists():
        from generate_data import generate
        generate(DATASET)
    return pd.read_csv(DATASET).dropna(subset=FEATURES + [TARGET])


def train_model(df):
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    metrics = {
        "MAE": mean_absolute_error(y_test, preds),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, preds))),
        "R2": r2_score(y_test, preds),
    }
    return model, metrics


def predict_price(model, **inputs):
    row = pd.DataFrame([inputs])[FEATURES]
    return max(float(model.predict(row)[0]), 0.0)


if __name__ == "__main__":
    df = load_data()
    _, m = train_model(df)
    print(f"Records: {len(df)}")
    print(f"MAE:  {m['MAE']:,.0f}\nRMSE: {m['RMSE']:,.0f}\nR2:   {m['R2']:.3f}")
