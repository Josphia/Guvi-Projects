import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

import mlflow
import mlflow.sklearn


# =========================
# 1. Load Dataset
# =========================
df = pd.read_csv("india_housing_prices.csv")


# =========================
# 2. Create Target Variable
# =========================
city_median = df.groupby("City")["Price_per_SqFt"].transform("median")
df["Good_Investment"] = (df["Price_per_SqFt"] < city_median).astype(int)


# =========================
# 3. Feature Selection
# =========================
features = [
    "BHK",
    "Size_in_SqFt",
    "Price_in_Lakhs",
    "Age_of_Property",
    "Nearby_Schools",
    "Nearby_Hospitals",
    "Property_Type",
    "Furnished_Status"
]

X = df[features].copy()
y = df["Good_Investment"]


# =========================
# 4. Encoding
# =========================
le_property = LabelEncoder()
le_furnished = LabelEncoder()

X["Property_Type"] = le_property.fit_transform(X["Property_Type"])
X["Furnished_Status"] = le_furnished.fit_transform(X["Furnished_Status"])


# =========================
# 5. Train-Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_test.to_csv("X_test_c.csv", index=False)
y_test.to_csv("y_test_c.csv", index=False)


# =========================
# 6. Scaling
# =========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# =========================
# 7. Models
# =========================
models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Random Forest": RandomForestClassifier(n_estimators=30, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5, weights="distance"),
    "XGBoost": XGBClassifier(
        n_estimators=30,
        max_depth=3,
        eval_metric="logloss",
        random_state=42
    )
}


# =========================
# 8. MLflow Experiment
# =========================
mlflow.set_experiment("Good_Investment_Classification")

trained_models = {}

for name, model in models.items():

    with mlflow.start_run(run_name=name):

        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)

        # Log params & metrics
        mlflow.log_param("model_name", name)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("roc_auc", roc_auc)

        # Log model
        mlflow.sklearn.log_model(model, "model")

        trained_models[name] = model


# =========================
# 9. Save Artifacts
# =========================
joblib.dump(trained_models, "models_c.pkl")
joblib.dump(scaler, "scaler_c.pkl")
joblib.dump(features, "features.pkl")
joblib.dump(le_property, "le_property.pkl")
joblib.dump(le_furnished, "le_furnished.pkl")

print("✅ Classification Training Completed Successfully")