from flask import Flask, jsonify, render_template, request
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

last_soil = None
last_rain = None
last_alert_time = 0

print("🔧 Starting Flask backend...")

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
@app.route("/sensor-data", methods=["POST"])
def receive_sensor_data():
    global last_soil, last_rain

    data = request.get_json()

    if not data:
        return jsonify({"error": "No data received"}), 400

    last_soil = data.get("soil")
    last_rain = data.get("rain")

    print("📡 Received from ESP8266:", last_soil, last_rain)

    return jsonify({"status": "success"}), 200

# ==============================
# WEATHER CACHE
# ==============================
last_weather_time = 0
cached_weather = {}

@app.route("/data")
def get_data():
    global last_soil, last_rain, last_alert_time
    global last_weather_time, cached_weather

    # ==============================
    # GET LOCATION INPUT
    # ==============================
    lat = request.args.get("lat")
    lon = request.args.get("lon")
    city = request.args.get("city")

    api_key = os.environ.get("OPENWEATHER_API_KEY")

    # Default location (Coimbatore)
    if not lat and not lon and not city:
        city = "Coimbatore"

    # ==============================
    # WEATHER FETCH (CACHED 5 MIN)
    # ==============================
    temperature = 0
    humidity = 0
    weather_desc = "Unavailable"
    city_name = city if city else "Unknown"
    forecast_rain = 0
    forecast_temp = 0
    forecast_humidity = 0

    current_time = time.time()

    if current_time - last_weather_time > 300:  # 5 minutes cache

        try:
            if lat and lon:
                lat = float(lat)
                lon = float(lon)
                weather_url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
                forecast_url = f"http://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={api_key}&units=metric"
            else:
                weather_url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
                forecast_url = f"http://api.openweathermap.org/data/2.5/forecast?q={city}&appid={api_key}&units=metric"

            weather_response = requests.get(weather_url, timeout = 5)
            weather_json = weather_response.json()

            if weather_response.status_code == 200:
                temperature = weather_json["main"]["temp"]
                humidity = weather_json["main"]["humidity"]
                weather_desc = weather_json["weather"][0]["description"]
                city_name = weather_json["name"]
                lat = weather_json["coord"]["lat"]
                lon = weather_json["coord"]["lon"]

            forecast_response = requests.get(forecast_url, timeout = 5)
            forecast_json = forecast_response.json()

            future_risk_series = []

            if forecast_response.status_code == 200:
                forecast_list = forecast_json["list"]

            # 🔹 Keep next immediate slot for display
            next_slot = forecast_list[0]

            forecast_temp = next_slot["main"]["temp"]
            forecast_humidity = next_slot["main"]["humidity"]
            forecast_rain = next_slot.get("rain", {}).get("3h", 0)

            # 🔹 Now generate 5-step future prediction
            for slot in forecast_list[:5]:
                f_rain = slot.get("rain", {}).get("3h", 0)

                # Start from current risk score
                future_score = risk_score

                if f_rain > 5:
                    future_score += 20
                elif f_rain > 2:
                    future_score += 10

                if future_score >= 70:
                    future_risk_series.append(2)
                elif future_score >= 40:
                    future_risk_series.append(1)
                else:
                    future_risk_series.append(0)
            
            # Save to cache
            cached_weather = {
                "temperature": temperature,
                "humidity": humidity,
                "weather_desc": weather_desc,
                "city_name": city_name,
                "lat": lat,
                "lon": lon,
                "forecast_temp": forecast_temp,
                "forecast_humidity": forecast_humidity,
                "forecast_rain": forecast_rain
            }

            last_weather_time = current_time

        except Exception as e:
            print("Weather API Error:", e)

    else:
        # Use cached data
        temperature = cached_weather.get("temperature", 0)
        humidity = cached_weather.get("humidity", 0)
        weather_desc = cached_weather.get("weather_desc", "Unavailable")
        city_name = cached_weather.get("city_name", city_name)
        lat = cached_weather.get("lat", lat)
        lon = cached_weather.get("lon", lon)
        forecast_temp = cached_weather.get("forecast_temp", 0)
        forecast_humidity = cached_weather.get("forecast_humidity", 0)
        forecast_rain = cached_weather.get("forecast_rain", 0)

    # ==============================
    # ML PREDICTION
    # ==============================
    risk_score = 0
    risk_level = "LOW"
    rf_pred = None
    dt_pred = None
    rf_confidence = None
    dt_confidence = None
    future_risk = "LOW"

    if last_soil is not None and last_rain is not None:

        features = pd.DataFrame(
            [[last_soil, last_rain, temperature, humidity]],
            columns=["Soil Moisture", "Rain Sensor", "Temperature", "Humidity"]
        )

        features_scaled = scaler.transform(features)

        rf_pred = rf_model.predict(features_scaled)[0]
        dt_pred = dt_model.predict(features_scaled)[0]

        rf_prob = rf_model.predict_proba(features_scaled)[0][1]
        dt_prob = dt_model.predict_proba(features_scaled)[0][1]

        rf_confidence = round(rf_prob * 100, 2)
        dt_confidence = round(dt_prob * 100, 2)

        risk_score = round(((rf_prob + dt_prob) / 2) * 100, 2)

        if risk_score >= 70:
            risk_level = "HIGH"
        elif risk_score >= 40:
            risk_level = "MEDIUM"

        # Future risk logic
        if forecast_rain > 5 and risk_score >= 40:
            future_risk = "HIGH"
        elif forecast_rain > 2:
            future_risk = "MEDIUM"

        # Telegram alert
        if risk_level == "HIGH":
            if current_time - last_alert_time > 60:
                send_telegram_alert(
                    f"⚠️ HIGH Landslide Risk!\nCity: {city_name}\nRisk Score: {risk_score}%"
                )
                last_alert_time = current_time

        # Log data
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
        "rf_prediction": int(rf_pred) if rf_pred is not None else None,
        "dt_prediction": int(dt_pred) if dt_pred is not None else None,
        "rf_confidence": rf_confidence,
        "dt_confidence": dt_confidence,
        "city": city_name,
        "temperature": temperature,
        "humidity": humidity,
        "weather": weather_desc,
        "lat": lat,
        "lon": lon,
        "forecast_temp": forecast_temp,
        "forecast_humidity": forecast_humidity,
        "forecast_rain": forecast_rain,
        "future_risk": future_risk
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