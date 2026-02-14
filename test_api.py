import requests

url = "http://127.0.0.1:5000/predict"
data = {"soil_moisture": 600, "rain_sensor": 400}

response = requests.post(url, json=data)
print("Response:", response.json())  # Expected output: {'landslide_risk': 1}
