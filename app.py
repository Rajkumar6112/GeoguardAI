from flask import Flask, jsonify, render_template
import serial
import time
import joblib
import numpy as np
import csv 
from datetime import datetime
import requests
import os
import time
import pandas as pd
from flask import request
app = Flask(__name__)

# Load trained models and scaler
rf_model = joblib.load("random_forest_model.pkl")
dt_model = joblib.load("decision_tree_model.pkl")
scaler = joblib.load("scaler.pkl")

arduino = None
last_soil = None
last_rain = None
# Connect to Arduino (change COM port if needed)
print("🔧 Starting Flask backend...")

if os.environ.get("RENDER") == "true":
    print("🌐 Running in cloud mode (Arduino disabled)")
else:
    try:
        arduino = serial.Serial('COM12',  9600, timeout = 2)
        time.sleep(2)
        print("✅ Arduino connected on COM12")
    except Exception as e:
        arduino = None
        print("⚠️ Arduino not connected:", e)

latest_data = {}

last_alert_time = 0

if not os.path.exists("prediction_log.csv"):
    with open("prediction_log.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "Timestamp",
            "Soil Moisture",
            "Rain Sensor",
            "Temperature",
            "Humidity",
            "Landslide Occurrence"
        ])

@app.route("/data")
def get_data():
    global last_soil, last_rain, last_alert_time

    rf_pred = None
    dt_pred = None
    risk_level = "LOW"

    city = request.args.get("city", "Coimbatore")

    # 🌦 ALWAYS fetch weather
    api_key = "f72f8c6c200cfb972d6a5c234a1f9a65"
    weather_url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"

    temperature = None
    humidity = None
    weather_desc = "Unavailable"
    city_name = city
    lat = None
    lon = None
    try:
        response = requests.get(weather_url)
        weather_json = response.json()

        if response.status_code == 200 and "main" in weather_json:
            temperature = weather_json["main"]["temp"]
            humidity = weather_json["main"]["humidity"]
            weather_desc = weather_json["weather"][0]["description"]
            city_name = weather_json["name"]
            lat = weather_json["coord"]["lat"]
            lon = weather_json["coord"]["lon"]
        
        else:
            print("Weather API error", weather_json)

        
    except Exception as e:
        print("Weather request failed", e)
    
    # 🔹 SENSOR READING
    if arduino and arduino.in_waiting > 0:
        line = arduino.readline().decode('utf-8', errors='ignore').strip()
        print("🔹 Serial:", line)

        if "Soil" in line:
            last_soil = int(line.split(":")[1].strip())
            
        elif "Rain" in line:
            last_rain = int(line.split(":")[1].strip())
            
    # 🔹 Prediction only if both values exist
    if last_soil is not None and last_rain is not None:
        features = pd.DataFrame([[last_soil, last_rain]], columns = ["Soil Moisture", "Rain Sensor"])
        features_scaled = scaler.transform(features)

        rf_pred = rf_model.predict(features_scaled)[0]
        dt_pred = dt_model.predict(features_scaled)[0]

        if last_soil < 400 and last_rain < 300:
            risk_level = "HIGH"
        elif last_soil < 700 and last_rain < 700:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        if risk_level == "HIGH":
            current_time = time.time()
            if current_time - last_alert_time > 60:
                send_telegram_alert(f"⚠️ HIGH Landslide Risk!\nSoil: {last_soil}\nRain: {last_rain}")
                last_alert_time =current_time

        # 📝 Log
        with open("prediction_log.csv", "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                last_soil,
                last_rain,
                temperature,
                humidity,
                1 if risk_level == "HIGH" else 0
            ])

    return jsonify({
        "soil_moisture": int(last_soil) if last_soil is not None else None,
        "rain_sensor": int(last_rain) if last_rain is not None else None,
        "rf_prediction": int(rf_pred) if rf_pred is not None else None,
        "dt_prediction": int(dt_pred) if dt_pred is not None else None,
        "risk_level": risk_level,
        "city": city_name,
        "temperature": float(temperature) if temperature is not None else None,
        "humidity": int(humidity) if humidity is not None else None,
        "weather": weather_desc,
        "lat": float(lat) if lat is not None else None,
        "lon": float(lon) if lon is not None else None
    })

@app.route("/")
def index():
    return render_template("index.html")

def send_telegram_alert(message):
    bot_token = "8496227214:AAGoPNqauJywQjbX8orHQXPXd6AxgBuc9TM"
    chat_id = "1727206518"

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    
    payload = {
        "chat_id": chat_id,
        "text": message
    }
    try:
        requests.post(url, data=payload)
        print("📩 Telegram alert sent!")
    except Exception as e:
        print("Telegram error:", e)
        
@app.route("/analytics")
def analytics():
    try:
        import pandas as pd

        df = pd.read_csv("prediction_log.csv")

        # Print columns for debugging
        print("Columns:", df.columns)

        total_records = len(df)

        high_count = len(df[df.iloc[:, -1] == "HIGH"])
        medium_count = len(df[df.iloc[:, -1] == "MEDIUM"])
        low_count = len(df[df.iloc[:, -1] == "LOW"])

        return render_template(
            "analytics.html",
            total=total_records,
            high=high_count,
            medium=medium_count,
            low=low_count
        )

    except Exception as e:
        return f"Error loading analytics: {e}"

if __name__ == "__main__":
    app.run(host = "0.0.0.0", port = int(os.environ.get("PORT",5000)))
