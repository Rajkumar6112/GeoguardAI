import serial
import requests
import time

# Initialize Serial Communication (Change COM3 if needed)
arduino = serial.Serial('COM12', 9600)
time.sleep(2)  # Allow time for connection

# Flask API URL (Ensure Flask is running) 
API_URL = "http://127.0.0.1:5000/predict"

while True:
    try:
        # Read data from Arduino
        data = arduino.readline().decode().strip()
        values = data.split(',')
        
        if len(values) == 2:  # Ensure both sensor values are received
            soil_moisture = int(values[0])
            rain_sensor = int(values[1])

            # Prepare JSON payload
            payload = {
                "soil_moisture": soil_moisture,
                "rain_sensor": rain_sensor
            }

            # Send data to API
            response = requests.post(API_URL, json=payload)
            print("API Response:", response.json())

        time.sleep(2)  # Send data every 2 seconds

    except Exception as e:
        print("Error:", e)
        break
