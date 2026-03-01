import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE

# 📂 Load dataset
df = pd.read_csv("prediction_log.csv")

print("Total samples:", len(df))
print(df["Landslide Occurrence"].value_counts())

# 🎯 Features (Smart Version 2.0)
X = df[["Soil Moisture", "Rain Sensor", "Temperature", "Humidity"]]
y = df["Landslide Occurrence"]

# 📊 Train/Test Split FIRST (Very Important)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y   # preserves class distribution
)

# 🔄 Apply SMOTE ONLY on training data
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print("\nAfter SMOTE balancing (Train only):")
print(pd.Series(y_train_resampled).value_counts())

# 📏 Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

# 🌲 Random Forest
rf_model = RandomForestClassifier(
    n_estimators=150,
    random_state=42
)
rf_model.fit(X_train_scaled, y_train_resampled)

# 🌳 Decision Tree
dt_model = DecisionTreeClassifier(
    random_state=42
)
dt_model.fit(X_train_scaled, y_train_resampled)

# 📈 Evaluation
rf_pred = rf_model.predict(X_test_scaled)
dt_pred = dt_model.predict(X_test_scaled)

print("\n🔵 Random Forest Report:")
print(classification_report(y_test, rf_pred))
print("Confusion Matrix (RF):")
print(confusion_matrix(y_test, rf_pred))
print("ROC-AUC (RF):", roc_auc_score(y_test, rf_pred))

print("\n🟢 Decision Tree Report:")
print(classification_report(y_test, dt_pred))
print("Confusion Matrix (DT):")
print(confusion_matrix(y_test, dt_pred))
print("ROC-AUC (DT):", roc_auc_score(y_test, dt_pred))

# 📊 Feature Importance (Very important for presentation)
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "RF Importance": rf_model.feature_importances_
}).sort_values(by="RF Importance", ascending=False)

print("\n📌 Feature Importance (Random Forest):")
print(feature_importance)

# 💾 Save models
joblib.dump(rf_model, "random_forest_model.pkl")
joblib.dump(dt_model, "decision_tree_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\n✅ Smart Model v3.1 trained successfully!")