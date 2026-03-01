from flask import Flask, jsonify, render_template, request
import serial
import time
import joblib
import numpy as np
import csv
from datetime import datetime
import requests
import os
import pandas as pd

app = Flask(__name__)

# ==============================
# LOAD MODELS
# ==============================
rf_model = joblib.load("random_forest_model.pkl")
dt_model = joblib.load("decision_tree_model.pkl")
scaler = joblib.load("scaler.pkl")

arduino = None
last_soil = None
last_rain = None
last_alert_time = 0

print("🔧 Starting Flask backend...")

# ==============================
# ARDUINO CONNECTION (LOCAL ONLY)
# ==============================
if os.environ.get("RENDER") == "true":
    print("🌐 Running in cloud mode (Arduino disabled)")
else:
    try:
        arduino = serial.Serial('COM12', 9600, timeout=2)
        time.sleep(2)
        print("✅ Arduino connected on COM12")
    except Exception as e:
        arduino = None
        print("⚠️ Arduino not connected:", e)

# ==============================
# CREATE LOG FILE IF NOT EXISTS
# ==============================
if not os.path.exists("prediction_log.csv"):
    with open("prediction_log.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "Timestamp",
            "Soil Moisture",
            "Rain Sensor",
            "Temperature",
            "Humidity",
            "Risk Score"
        ])

# ==============================
# MAIN DATA ROUTE
# ==============================
@app.route("/data")
def get_data():
    global last_soil, last_rain, last_alert_time

    city = request.args.get("city", "Coimbatore")

    temperature = 0
    humidity = 0
    weather_desc = "Unavailable"
    city_name = city
    lat = None
    lon = None

    # ==============================
    # WEATHER FETCH
    # ==============================
    api_key = os.environ.get("OPENWEATHER_API_KEY")
    weather_url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"

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

    except Exception as e:
        print("Weather Error:", e)

    # ==============================
    # SENSOR READING
    # ==============================
    if arduino and arduino.in_waiting > 0:
        line = arduino.readline().decode('utf-8', errors='ignore').strip()
        print("🔹 Serial:", line)

        if "Soil" in line:
            last_soil = int(line.split(":")[1].strip())

        elif "Rain" in line:
            last_rain = int(line.split(":")[1].strip())

    risk_score = 0
    risk_level = "LOW"
    rf_pred = None
    dt_pred = None
    rf_confidence = None
    dt_confidence = None

    if last_soil is not None and last_rain is not None:

        features = pd.DataFrame(
            [[last_soil, last_rain, temperature, humidity]],
            columns=["Soil Moisture", "Rain Sensor", "Temperature", "Humidity"]
        )

        features_scaled = scaler.transform(features)

        # ✅ Proper Predictions
        rf_pred = rf_model.predict(features_scaled)[0]
        dt_pred = dt_model.predict(features_scaled)[0]

        # ✅ Proper Probabilities
        rf_prob = rf_model.predict_proba(features_scaled)[0][1]
        dt_prob = dt_model.predict_proba(features_scaled)[0][1]

        # ✅ Risk Score (0–100)
        avg_prob = (rf_prob + dt_prob) / 2
        risk_score = round(avg_prob * 100, 2)

        rf_confidence = round(rf_prob * 100, 2)
        dt_confidence = round(dt_prob * 100, 2)

        # ✅ Risk Level
        if risk_score >= 70:
            risk_level = "HIGH"
        elif risk_score >= 40:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        # ==============================
        # TELEGRAM ALERT (Cooldown 60s)
        # ==============================
        if risk_level == "HIGH":
            current_time = time.time()
            if current_time - last_alert_time > 60:
                send_telegram_alert(
                    f"⚠️ HIGH Landslide Risk!\nCity: {city_name}\nRisk Score: {risk_score}%"
                )
                last_alert_time = current_time

        # ==============================
        # LOG DATA
        # ==============================
        with open("prediction_log.csv", "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                last_soil,
                last_rain,
                temperature,
                humidity,
                risk_score
            ])

    return jsonify({
        "soil_moisture": last_soil,
        "rain_sensor": last_rain,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "rf_prediction":int(rf_pred)if rf_pred is not None else None,
        "dt_prediction": int(dt_pred) if dt_pred is not None else None,
        "rf_confidence": rf_confidence,
        "dt_confidence": dt_confidence,
        "city": city_name,
        "temperature": temperature,
        "humidity": humidity,
        "weather": weather_desc,
        "lat": lat,
        "lon": lon
    })


# ==============================
# TELEGRAM FUNCTION
# ==============================
def send_telegram_alert(message):
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")

    if not bot_token or not chat_id:
        return

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"

    try:
        requests.post(url, data={"chat_id": chat_id, "text": message})
        print("📩 Telegram alert sent!")
    except Exception as e:
        print("Telegram Error:", e)


# ==============================
# ANALYTICS ROUTE
# ==============================
@app.route("/analytics")
def analytics():
    try:
        df = pd.read_csv("prediction_log.csv")

        total_records = len(df)

        high_count = len(df[df["Risk Score"] >= 60])
        medium_count = len(df[(df["Risk Score"] >= 30) & (df["Risk Score"] < 60)])
        low_count = len(df[df["Risk Score"] < 30])

        return render_template(
            "analytics.html",
            total=total_records,
            high=high_count,
            medium=medium_count,
            low=low_count
        )

    except Exception as e:
        return f"Error loading analytics: {e}"


@app.route("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))