import serial
import time

try:
    arduino = serial.Serial('COM12', 9600, timeout=2)
    time.sleep(2)  # Allow connection to stabilize
except serial.SerialException:
    print("Error: Could not connect to Arduino. Check COM port.")
    exit()

print("Collecting data...")

try:
    while True:
        if arduino.in_waiting > 0:
            line = arduino.readline().decode('utf-8', errors='ignore').strip()
            print(f"🔹 Raw Data Received: {line}")

            # Split the line correctly if both values are on the same line
            if "Soil Moisture" in line and "Rain Sensor" in line:
                try:
                    parts = line.replace("Soil Moisture:", "").replace("Rain Sensor:", "").split("|")
                    soil_moisture = int(parts[0].strip())
                    rain_value = int(parts[1].strip())

                    print(f"✅ Soil Moisture: {soil_moisture}, Rain Sensor: {rain_value}")

                    # Predict landslide occurrence
                    landslide = 1 if (soil_moisture < 500 and rain_value < 300) else 0
                    print(f"✅ Data Processed: {soil_moisture}, {rain_value}, {landslide}")

                except ValueError:
                    print("❌ Error: Data format issue, skipping this reading.")

        time.sleep(1)

except KeyboardInterrupt:
    print("❌ Data collection stopped.")
    arduino.close()
