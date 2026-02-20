import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# 1. Load the dataset
df = pd.read_csv('india_housing_prices.csv')

# 2. STEP 1: CREATE THE TARGET (Good_Investment)
# Rule: If the price per sqft is lower than the median price in that city, it's a "Good Investment"
city_median = df.groupby('City')['Price_per_SqFt'].transform('median')
df['Good_Investment'] = (df['Price_per_SqFt'] < city_median).astype(int)

print(f"Target Created! Good Investments: {df['Good_Investment'].sum()} out of {len(df)}")

# 3. PREPROCESSING
# Select features to use for prediction
features = ['BHK', 'Size_in_SqFt', 'Price_in_Lakhs', 'Age_of_Property', 
            'Nearby_Schools', 'Nearby_Hospitals', 'Property_Type', 'Furnished_Status']

X = df[features].copy()
y = df['Good_Investment']

# Encode categorical text data into numbers
le = LabelEncoder()
X['Property_Type'] = le.fit_transform(X['Property_Type'])
X['Furnished_Status'] = le.fit_transform(X['Furnished_Status'])

# 4. TRAIN-TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. FEATURE SCALING
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. MODEL TRAINING (Using Random Forest as required)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 7. EVALUATION
predictions = model.predict(X_test_scaled)
print("\n--- Model Performance ---")
print(f"Accuracy: {accuracy_score(y_test, predictions):.2f}")
print("\nClassification Report:\n", classification_report(y_test, predictions))