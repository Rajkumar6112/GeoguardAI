import serial
import pandas as pd
import time
import os

csv_file = "landslide_data.csv"

# Auto-create CSV with header
if not os.path.exists(csv_file):
    df = pd.DataFrame(columns=["Timestamp", "Soil Moisture", "Rain Sensor", "Landslide Occurrence"])
    df.to_csv(csv_file, index=False)
    print("📁 CSV file created")

try:
    arduino = serial.Serial('COM12', 9600, timeout=2)  # Change COM if needed
    time.sleep(2)
except:
    print("❌ Could not connect to Arduino. Check COM port.")
    exit()

print("✅ Collecting data... Press CTRL+C to stop.")

soil_value = None
rain_value = None

try:
    while True:
        if arduino.in_waiting > 0:
            line = arduino.readline().decode('utf-8', errors='ignore').strip()
            print("🔹 Raw Data:", line)

            # Parse Soil line
            if "Soil" in line:
                soil_value = int(line.split(":")[1].strip())
                print("   ↳ Soil captured:", soil_value)

            # Parse Rain line
            elif "Rain" in line:
                rain_value = int(line.split(":")[1].strip())
                print("   ↳ Rain captured:", rain_value)

            # Save only when both values exist
            if soil_value is not None and rain_value is not None:
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                landslide = 1 if (soil_value < 500 and rain_value < 300) else 0

                row = pd.DataFrame([[timestamp, soil_value, rain_value, landslide]],
                                   columns=["Timestamp", "Soil Moisture", "Rain Sensor", "Landslide Occurrence"])

                row.to_csv(csv_file, mode='a', header=False, index=False)

                print(f"✅ Saved → {timestamp}, Soil={soil_value}, Rain={rain_value}, Landslide={landslide}")

                # Reset for next reading
                soil_value = None
                rain_value = None

        time.sleep(0.5)

except KeyboardInterrupt:
    print("❌ Data collection stopped.")
    arduino.close()
