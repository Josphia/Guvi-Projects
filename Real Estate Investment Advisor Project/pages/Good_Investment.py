import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier

df = pd.read_csv(r"E:\VS Code Projects\Guvi-Projects\Real Estate Investment Advisor Project\india_housing_prices.csv")

st.title("🏡 Real Estate Investment Advisor")
st.subheader("Predict whether a property is a Good Investment")

city_median = df.groupby('City')['Price_per_SqFt'].transform('median')
df['Good_Investment'] = (df['Price_per_SqFt'] < city_median).astype(int)

features = ['BHK', 'Size_in_SqFt', 'Price_in_Lakhs', 'Age_of_Property',
    'Nearby_Schools', 'Nearby_Hospitals', 'Property_Type', 'Furnished_Status']

X = df[features].copy()
y = df['Good_Investment']

le_property = LabelEncoder()
le_furnished = LabelEncoder()

X['Property_Type'] = le_property.fit_transform(X['Property_Type'])
X['Furnished_Status'] = le_furnished.fit_transform(X['Furnished_Status'])

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42 )

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42)
}

model_accuracy = {}
trained_models = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, preds)

    model_accuracy[name] = acc
    trained_models[name] = model

acc_df = pd.DataFrame({
    "Model": model_accuracy.keys(),
    "Accuracy": model_accuracy.values()
}).sort_values(by="Accuracy", ascending=False)

st.markdown("### 📊 Model Performance")
st.dataframe(acc_df, use_container_width=True)

selected_model_name = st.selectbox(
    "Choose a model for prediction",
    acc_df["Model"]
)

selected_model = trained_models[selected_model_name]

st.markdown("### 📝 Enter Property Details")

bhk = st.number_input("Number of BHK", min_value=1, max_value=10, value=2)
sqft = st.number_input("Size in SqFt", min_value=100, value=1200)
price = st.number_input("Price (in Lakhs)", min_value=1.0, value=50.0)
age = st.number_input("Age of Property (Years)", min_value=0, value=5)
nearby_schools = st.number_input("Nearby Schools", min_value=0, value=2)
nearby_hospitals = st.number_input("Nearby Hospitals", min_value=0, value=1)
property_type = st.selectbox("Property Type", df['Property_Type'].unique())
furnished_status = st.selectbox("Furnished Status", df['Furnished_Status'].unique())

if st.button("Is this a Good Investment? 🔍"):

    property_type_encoded = le_property.transform([property_type])[0]
    furnished_status_encoded = le_furnished.transform([furnished_status])[0]

    user_data = pd.DataFrame([[
        bhk,
        sqft,
        price,
        age,
        nearby_schools,
        nearby_hospitals,
        property_type_encoded,
        furnished_status_encoded
    ]], columns=features)

    user_data_scaled = scaler.transform(user_data)

    # Predict
    prediction = selected_model.predict(user_data_scaled)
    probability = selected_model.predict_proba(user_data_scaled)[0][1]

    # Output
    if prediction[0] == 1:
        st.success(f"✅ YES! Good Investment\n\nConfidence: {probability:.2%}")
    else:
        st.error(f"❌ NOT a Good Investment\n\nConfidence: {1 - probability:.2%}")