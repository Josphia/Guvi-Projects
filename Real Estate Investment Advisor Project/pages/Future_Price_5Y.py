(Frontend)

import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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

city_growth_rates = {
    'Ahmedabad': 1.45, 'Amritsar': 1.25, 'Bangalore': 1.70, 'Bhopal': 1.30,
    'Bhubaneswar': 1.35, 'Bilaspur': 1.20, 'Chennai': 1.48, 'Coimbatore': 1.40,
    'Cuttack': 1.28, 'Dehradun': 1.35, 'Durgapur': 1.22, 'Dwarka': 1.55,
    'Faridabad': 1.42, 'Gaya': 1.20, 'Gurgaon': 1.65, 'Guwahati': 1.32,
    'Haridwar': 1.25, 'Hyderabad': 1.62, 'Indore': 1.44, 'Jaipur': 1.38,
    'Jamshedpur': 1.28, 'Jodhpur': 1.25, 'Kochi': 1.35, 'Kolkata': 1.30,
    'Lucknow': 1.45, 'Ludhiana': 1.32, 'Mangalore': 1.30, 'Mumbai': 1.55,
    'Mysore': 1.35, 'Nagpur': 1.38, 'New Delhi': 1.58, 'Noida': 1.60,
    'Patna': 1.35, 'Pune': 1.52, 'Raipur': 1.28, 'Ranchi': 1.30,
    'Silchar': 1.18, 'Surat': 1.45, 'Trivandrum': 1.35, 'Vijayawada': 1.32,
    'Vishakhapatnam': 1.40, 'Warangal': 1.25
}

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

