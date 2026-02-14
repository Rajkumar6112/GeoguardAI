import serial
import time
import joblib
import numpy as np

# Load trained models and scaler
rf_model = joblib.load("random_forest_model.pkl")
dt_model = joblib.load("decision_tree_model.pkl")
scaler = joblib.load("scaler.pkl")

# Connect to Arduino
try:
    arduino = serial.Serial('COM12', 9600, timeout=2)  # Change COM if needed
    time.sleep(2)
except:
    print("❌ Could not connect to Arduino. Check COM port.")
    exit()

print("✅ Real-time landslide prediction started...")

soil_value = None
rain_value = None

try:
    while True:
        if arduino.in_waiting > 0:
            line = arduino.readline().decode('utf-8', errors='ignore').strip()
            print("🔹 Raw Data:", line)

            if "Soil" in line:
                soil_value = int(line.split(":")[1].strip())
                print("   ↳ Soil captured:", soil_value)

            elif "Rain" in line:
                rain_value = int(line.split(":")[1].strip())
                print("   ↳ Rain captured:", rain_value)

            # Predict only when both values are received
            if soil_value is not None and rain_value is not None:
                features = np.array([[soil_value, rain_value]])
                features_scaled = scaler.transform(features)

                rf_pred = rf_model.predict(features_scaled)[0]
                dt_pred = dt_model.predict(features_scaled)[0]

                print("📊 Predictions → RF:", rf_pred, "DT:", dt_pred)

                # After capturing soil_value and rain_value

                if soil_value < 400 and rain_value < 300:
                    risk_level = "HIGH"
                elif soil_value < 700 and rain_value < 700:
                    risk_level = "MEDIUM"
                else:
                    risk_level = "LOW"

                print("⚠️ Risk Level:", risk_level)

                if rf_pred == 1 or dt_pred == 1:
                    print("⚠️  Landslide Risk Detected!")
                else:
                    print("✅ No Landslide Risk.")

                print("-----------------------------")

                # Reset values for next reading
                soil_value = None
                rain_value = None

        time.sleep(0.5)

except KeyboardInterrupt:
    print("❌ Real-time prediction stopped.")
    arduino.close()
