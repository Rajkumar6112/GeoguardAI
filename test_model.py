import numpy as np
import joblib

# Load trained models and scaler
rf_model = joblib.load("landslide_rf_model.pkl") 
dt_model = joblib.load("landslide_dt_model.pkl")  
scaler = joblib.load("scaler.pkl")

# Example test data (Soil Moisture: 400, Rain Sensor: 250)
test_data = np.array([[50,50]])

# Scale input data
test_data_scaled = scaler.transform(test_data)

# Make predictions
rf_prediction = rf_model.predict(test_data_scaled)
dt_prediction = dt_model.predict(test_data_scaled)

# Print results
print(f"🌱 Soil Moisture: {test_data[0][0]}")
print(f"🌧️ Rain Sensor: {test_data[0][1]}")
print(f"🌍 Random Forest Prediction (1=Landslide, 0=No Landslide): {rf_prediction[0]}")
print(f"🌍 Decision Tree Prediction (1=Landslide, 0=No Landslide): {dt_prediction[0]}")
