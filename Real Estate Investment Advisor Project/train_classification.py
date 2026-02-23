import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

df = pd.read_csv(r"E:\VS Code Projects\Guvi-Projects\Real Estate Investment Advisor Project\india_housing_prices.csv")

city_median = df.groupby("City")["Price_per_SqFt"].transform("median")
df["Good_Investment"] = (df["Price_per_SqFt"] < city_median).astype(int)

features = [ "BHK", "Size_in_SqFt", "Price_in_Lakhs", "Age_of_Property", "Nearby_Schools",
    "Nearby_Hospitals", "Property_Type", "Furnished_Status" ]

X = df[features].copy()
y = df["Good_Investment"]

le_property = LabelEncoder()
le_furnished = LabelEncoder()
X["Property_Type"] = le_property.fit_transform(X["Property_Type"])
X["Furnished_Status"] = le_furnished.fit_transform(X["Furnished_Status"])

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42 )

X_test.to_csv(r"E:\VS Code Projects\Guvi-Projects\Real Estate Investment Advisor Project\X_test_c.csv", index=False)
y_test.to_csv(r"E:\VS Code Projects\Guvi-Projects\Real Estate Investment Advisor Project\y_test_c.csv", index=False)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

models = {
    "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=30, random_state=42),
    "XGBoost": XGBClassifier( n_estimators=30, max_depth=3, eval_metric="logloss", random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5, weights="distance")
    
}

trained_models = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    trained_models[name] = model

joblib.dump(features, r"E:\VS Code Projects\Guvi-Projects\Real Estate Investment Advisor Project\features.pkl")
joblib.dump(le_property, r"E:\VS Code Projects\Guvi-Projects\Real Estate Investment Advisor Project\le_property.pkl")
joblib.dump(le_furnished, r"E:\VS Code Projects\Guvi-Projects\Real Estate Investment Advisor Project\le_furnished.pkl")
joblib.dump(scaler, r"E:\VS Code Projects\Guvi-Projects\Real Estate Investment Advisor Project\scaler_c.pkl")
joblib.dump(trained_models, r"E:\VS Code Projects\Guvi-Projects\Real Estate Investment Advisor Project\models_c.pkl")
 
print("Training completed")