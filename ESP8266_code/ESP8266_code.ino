#include <ESP8266WiFi.h>
#include <ESP8266HTTPClient.h>
#include <WiFiClientSecure.h>

const char* ssid = "Alone viber";
const char* password = "1223334444";

const char* serverName = "https://geoguardai.onrender.com/sensor-data";

#define SOIL_PIN A0
#define RAIN_PIN D2
int soilValue = analogRead(A0);
int rainValue = digitalRead(D2);

void setup() {
  Serial.begin(115200);

  WiFi.begin(ssid, password);
  Serial.print("Connecting");

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("\nWiFi connected");
}

void loop() {
  if (WiFi.status() == WL_CONNECTED) {

    WiFiClientSecure client;
    client.setInsecure();  // ignore SSL certificate

    HTTPClient http;

    int soilValue = analogRead(SOIL_PIN);
    int rainValue = digitalRead(RAIN_PIN);

    String jsonData = "{\"soil\":" + String(soilValue) +
                      ",\"rain\":" + String(rainValue) + "}";

    http.begin(client, serverName);
    http.addHeader("Content-Type", "application/json");

    int httpResponseCode = http.POST(jsonData);

    Serial.print("Response: ");
    Serial.println(httpResponseCode);

    http.end();
  }

  delay(5000);
}