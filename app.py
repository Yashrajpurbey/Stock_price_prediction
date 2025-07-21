import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Page config
st.set_page_config(page_title="Stock Price Predictor", layout="wide")

st.title("📈 Stock Price Prediction App")
st.write("Predict next day closing price using Linear Regression & Random Forest.")

# Upload CSV file
uploaded_file = st.file_uploader("Upload stock CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, parse_dates=["Date"], index_col="Date")

    # Feature engineering
    df["Prev_Close"] = df["Close"].shift(1)
    df["5_day_avg"] = df["Close"].rolling(window=5).mean().shift(1)
    df["10_day_avg"] = df["Close"].rolling(window=10).mean().shift(1)
    df = df.dropna()

    # Train-test split
    X = df[["Prev_Close", "5_day_avg", "10_day_avg", "Volume"]]
    y = df["Close"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Model training
    lr_model = LinearRegression()
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    lr_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)

    # Predictions
    lr_preds = lr_model.predict(X_test)
    rf_preds = rf_model.predict(X_test)

    # Evaluation
    lr_rmse = mean_squared_error(y_test, lr_preds, squared=False)
    rf_rmse = mean_squared_error(y_test, rf_preds, squared=False)
    lr_r2 = r2_score(y_test, lr_preds)
    rf_r2 = r2_score(y_test, rf_preds)

    st.subheader("📊 Model Evaluation")
    st.write(f"**Linear Regression** RMSE: {lr_rmse:.2f}, R²: {lr_r2:.2f}")
    st.write(f"**Random Forest** RMSE: {rf_rmse:.2f}, R²: {rf_r2:.2f}")

    st.subheader("🔮 Predictions vs Actual")

    results = pd.DataFrame({
        "Actual": y_test.values,
        "LR_Predicted": lr_preds,
        "RF_Predicted": rf_preds
    }, index=y_test.index)

    st.line_chart(results)

    st.subheader("📅 Predict Next Closing Price")
    latest_row = df.iloc[-1]
    latest_features = np.array([
        latest_row["Close"],
        df["Close"].rolling(window=5).mean().iloc[-1],
        df["Close"].rolling(window=10).mean().iloc[-1],
        latest_row["Volume"]
    ]).reshape(1, -1)

    lr_next = lr_model.predict(latest_features)[0]
    rf_next = rf_model.predict(latest_features)[0]

    st.write(f"🔵 **Linear Regression Prediction:** {lr_next:.2f}")
    st.write(f"🟢 **Random Forest Prediction:** {rf_next:.2f}")
