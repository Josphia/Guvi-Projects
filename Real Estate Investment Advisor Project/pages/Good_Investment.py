import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression 
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

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


models_list = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42),
    "SVM": SVC(probability=True, random_state=42)
}

for name, model in models_list.items():
    print(f"Training {name}...")
    
    model.fit(X_train_scaled, y_train)    # Training the actual object
    preds = model.predict(X_test_scaled)  # Making predictions
    acc = accuracy_score(y_test, preds)   # Checking accuracy
    
    print(f"{name} Accuracy: {acc:.4f}")
    print("-" * 30)









