import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVR
from xgboost import XGBRegressor

df = pd.read_csv(r"E:\VS Code Projects\Guvi-Projects\Real Estate Investment Advisor Project\india_housing_prices.csv")

df["Future_Price_5Y"] = df["Price_in_Lakhs"] * (1.08 ** 5)

features = ["BHK", "Size_in_SqFt", "Price_in_Lakhs", "Year_Built", "Floor_No", "Total_Floors"]
X = df[features]
y = df["Future_Price_5Y"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_test.to_csv(r"E:\VS Code Projects\Guvi-Projects\Real Estate Investment Advisor Project\X_test_r.csv", index=False)
y_test.to_csv(r"E:\VS Code Projects\Guvi-Projects\Real Estate Investment Advisor Project\y_test_r.csv", index=False)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1),
    "LinearSVR": LinearSVR(random_state=42, max_iter=10000),
    "XGBoost": XGBRegressor(objective="reg:squarederror", n_estimators=50, max_depth=4, learning_rate=0.1, random_state=42, n_jobs=-1)
}

trained_models = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    trained_models[name] = model

joblib.dump(trained_models, r"E:\VS Code Projects\Guvi-Projects\Real Estate Investment Advisor Project\models_r.pkl", compress=3)
joblib.dump(scaler, r"E:\VS Code Projects\Guvi-Projects\Real Estate Investment Advisor Project\scaler_r.pkl")
joblib.dump(features, r"E:\VS Code Projects\Guvi-Projects\Real Estate Investment Advisor Project\features_r.pkl")

print("Training Done")