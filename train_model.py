import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib

df = pd.read_csv("prediction_log.csv")
print(df.columns)
exit()

# Features upgraded
X = df[["Soil Moisture", "Rain Sensor", "Temperature", "Humidity"]]
y = df["Landslide Occurrence"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

print("RF Accuracy:", accuracy_score(y_test, rf_model.predict(X_test)))
print("DT Accuracy:", accuracy_score(y_test, dt_model.predict(X_test)))

joblib.dump(rf_model, "random_forest_model.pkl")
joblib.dump(dt_model, "decision_tree_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ Smart Model 2.0 Trained Successfully")
