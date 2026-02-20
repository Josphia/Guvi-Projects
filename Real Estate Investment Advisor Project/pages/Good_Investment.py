import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# 1. Load the dataset
df = pd.read_csv(r"E:\VS Code Projects\Guvi-Projects\Real Estate Investment Advisor Project\india_housing_prices.csv")

city_median = df.groupby('City')['Price_per_SqFt'].transform('median')
df['Good_Investment'] = (df['Price_per_SqFt'] < city_median).astype(int)

features = ['BHK', 'Size_in_SqFt', 'Price_in_Lakhs', 'Age_of_Property', 
            'Nearby_Schools', 'Nearby_Hospitals', 'Property_Type', 'Furnished_Status']

X = df[features].copy()
y = df['Good_Investment']

le = LabelEncoder()
X['Property_Type'] = le.fit_transform(X['Property_Type'])
X['Furnished_Status'] = le.fit_transform(X['Furnished_Status'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

predictions = model.predict(X_test_scaled)
print("\n--- Model Performance ---")
print(f"Accuracy: {accuracy_score(y_test, predictions):.2f}")
print("\nClassification Report:\n", classification_report(y_test, predictions))