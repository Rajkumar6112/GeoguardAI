#include <ESP8266WiFi.h>
#include <ESP8266HTTPClient.h>

const char* ssid = "YOUR_WIFI";
const char* password = "YOUR_PASSWORD";
const char* serverName = "https://geoguardai.onrender.com/sensor-data";

void setup() {
  Serial.begin(115200);
  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
  }
}

void loop() {
  if (WiFi.status() == WL_CONNECTED) {

    HTTPClient http;
    WiFiClientSecure client;
    client.setInsecure();  // ignore SSL certificate

    http.begin(client, serverName);
    http.addHeader("Content-Type", "application/json");

    int soil = analogRead(A0);
    int rain = analogRead(D1);

    String jsonData = "{\"soil\":" + String(soil) +
                      ",\"rain\":" + String(rain) + "}";

    int httpResponseCode = http.POST(jsonData);

    http.end();
  }

  delay(5000);
}