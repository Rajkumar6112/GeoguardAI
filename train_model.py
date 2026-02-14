import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load dataset
df = pd.read_csv("landslide_data.csv")

# Remove any bad rows
df = df.dropna()

# Features and label
X = df[["Soil Moisture", "Rain Sensor"]]
y = df["Landslide Occurrence"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# Random Forest Model
# -----------------------------
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)
rf_pred = rf_model.predict(X_test_scaled)

rf_accuracy = accuracy_score(y_test, rf_pred)
print("Random Forest Accuracy:", rf_accuracy)
print("Random Forest Report:\n", classification_report(y_test, rf_pred))

# -----------------------------
# Decision Tree Model
# -----------------------------
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train_scaled, y_train)
dt_pred = dt_model.predict(X_test_scaled)

dt_accuracy = accuracy_score(y_test, dt_pred)
print("Decision Tree Accuracy:", dt_accuracy)
print("Decision Tree Report:\n", classification_report(y_test, dt_pred))

# Save models and scaler
joblib.dump(rf_model, "random_forest_model.pkl")
joblib.dump(dt_model, "decision_tree_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ Models and scaler saved successfully!")
