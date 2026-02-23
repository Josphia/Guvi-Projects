import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
#Future_Price Final and Clear

st.title("🏡 Real Estate Investment Advisor")
st.subheader("Predicting the Future Price in 5 Years")

@st.cache_resource
def load_assets():
    models = joblib.load(r"E:\VS Code Projects\Guvi-Projects\Real Estate Investment Advisor Project\models_r.pkl")
    scaler = joblib.load(r"E:\VS Code Projects\Guvi-Projects\Real Estate Investment Advisor Project\scaler_r.pkl")
    features = joblib.load(r"E:\VS Code Projects\Guvi-Projects\Real Estate Investment Advisor Project\features_r.pkl")
    le_city = joblib.load(r"E:\VS Code Projects\Guvi-Projects\Real Estate Investment Advisor Project\le_city.pkl")
    return models, scaler, features, le_city

models, scaler, features, le_city = load_assets()

model_name = st.selectbox("Choose Model", list(models.keys()))
model = models[model_name]

bhk = st.number_input("BHK (between 1–10)", 1, 10, 2)
sqft = st.number_input("Size in SqFt", 100, 10000, 1200)
price = st.number_input("Current Price (in Lakhs)", 1.0, 1000.0, 50.0)
year_built = st.number_input("Year Built", 1950, 2026, 2015)
city = st.selectbox("City", le_city.classes_)

X_test = pd.read_csv(r"E:\VS Code Projects\Guvi-Projects\Real Estate Investment Advisor Project\X_test_r.csv")
y_test = pd.read_csv(r"E:\VS Code Projects\Guvi-Projects\Real Estate Investment Advisor Project\y_test_r.csv")

X_test_scaled = scaler.transform(X_test)
y_pred = model.predict(X_test_scaled)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)

if st.button("Predict Future Price and Evaluate Model Metrics"):

    city_enc = le_city.transform([city])[0]

    user_df = pd.DataFrame([[bhk, sqft, price, year_built, city_enc]], columns=features) 

    user_scaled = scaler.transform(user_df)
    future_price = model.predict(user_scaled)[0]

    st.success(f"💰 Estimated Future Price (5 Years): ₹{future_price:.2f} Lakhs")

    st.write(f"RMSE: {rmse:.4f}")
    st.write(f"MAE: {mae:.4f}")
    st.write(f"R² Score: {r2:.4f}")