import streamlit as st
import pandas as pd
import joblib

st.title("🏡 Real Estate Investment Advisor")

@st.cache_resource
def load_assets():
    models = joblib.load(r"E:\VS Code Projects\Guvi-Projects\Real Estate Investment Advisor Project\models.pkl")
    scaler = joblib.load(r"E:\VS Code Projects\Guvi-Projects\Real Estate Investment Advisor Project\scaler.pkl")
    le_property = joblib.load(r"E:\VS Code Projects\Guvi-Projects\Real Estate Investment Advisor Project\le_property.pkl")
    le_furnished = joblib.load(r"E:\VS Code Projects\Guvi-Projects\Real Estate Investment Advisor Project\le_furnished.pkl")
    features = joblib.load(r"E:\VS Code Projects\Guvi-Projects\Real Estate Investment Advisor Project\features.pkl")
    return models, scaler, le_property, le_furnished, features

models, scaler, le_property, le_furnished, features = load_assets()

model_name = st.selectbox("Choose Model", list(models.keys()))
model = models[model_name]

bhk = st.number_input("BHK (between 1-10)", 1, 10, 2)
sqft = st.number_input("Size in SqFt", 100, 10000, 1200)
price = st.number_input("Price (Lakhs)", 1.0, 1000.0, 50.0)
age = st.number_input("Age of Property", 0, 50, 5)

schools = st.number_input("Nearby Schools", 0, 20, 2)
hospitals = st.number_input("Nearby Hospitals", 0, 20, 1)

property_type = st.selectbox("Property Type", le_property.classes_)
furnished_status = st.selectbox("Furnished Status", le_furnished.classes_)

if st.button("Predict"):

    user_df = pd.DataFrame([[ bhk, sqft, price, age, schools, hospitals, 
                            le_property.transform([property_type])[0], 
                            le_furnished.transform([furnished_status])[0]
    ]], columns=features)

    user_scaled = scaler.transform(user_df)
    prediction = model.predict(user_scaled)[0]

    if prediction == 1:
        st.success("Yes, GOOD Investment ✅")
    else:
        st.error("No, NOT a Good Investment ❌")