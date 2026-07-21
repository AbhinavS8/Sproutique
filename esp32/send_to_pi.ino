#include <ESP8266WiFi.h>
#include <ESP8266HTTPClient.h>
#include <DHT.h>

// Wi-Fi credentials
const char* ssid = ".."; //same wifi as RPi
const char* password = "..";

// Server URL
const char* serverUrl = "../data"; //configure based on hostname -I

// DHT sensor configuration
#define DHTPIN D2      
#define DHTTYPE DHT22  
DHT dht(DHTPIN, DHTTYPE);

#define PH_ANALOG_PIN A0 
#define PH_DIGITAL_PIN D1    // pH sensor digital pin (optional)
 // Configure pH digital pin as input (if used)
 //pinMode(PH_DIGITAL_PIN, INPUT);


// Timing
const unsigned long interval = 10000; 
unsigned long previousMillis = 0;

void setup() {
    Serial.begin(115200);
    dht.begin();

    // Connect to Wi-Fi
    WiFi.begin(ssid, password);
    Serial.print("Connecting to WiFi");
    while (WiFi.status() != WL_CONNECTED) {
        delay(1000);
        Serial.print(".");
    }
    Serial.println(WiFi.localIP());
    Serial.println(" Connected!");
}

void loop() {
    unsigned long currentMillis = millis();

    if (currentMillis - previousMillis >= interval) {
        previousMillis = currentMillis;  // Update the timer

        // Read sensor data
        float temperature = dht.readTemperature();
        float humidity = dht.readHumidity();
        int analogValue = analogRead(PH_ANALOG_PIN);

        float voltage = analogValue * (1.0 / 1024.0); // Scale for 1V ADC on ESP8266
        float ph = 5;//(voltage - 0.5) * 3.5 + 7.0;  
        
        // Check if sensor readings are valid
        if (isnan(temperature) || isnan(humidity)) {
            Serial.println("Failed to read from sensor");
            return;
        }

        // Send data to the server
        sendData(temperature, humidity, ph);
    }
    delay(1000);
}

void sendData(float temperature, float humidity, float ph) {
    if (WiFi.status() == WL_CONNECTED) {
        HTTPClient http;
        WiFiClient client;
        http.begin(client, serverUrl);

        // Create JSON payload
        String jsonData = "{\"temperature\": " + String(temperature, 2) + 
                          ", \"humidity\": " + String(humidity, 2) + 
                          ", \"ph\": " + String(ph, 2) +"}";
        http.addHeader("Content-Type", "application/json");

        // Send the POST request
        int httpResponseCode = http.POST(jsonData);
        if (httpResponseCode > 0) {
            String response = http.getString();
            Serial.println("Response: " + response);
            Serial.println(String(temperature)+" "+String(humidity)+" "+String(ph));
        } else {
            Serial.printf("Error sending data: %s\n", http.errorToString(httpResponseCode).c_str());
        }

        http.end();  // Close the connection
    } else {
        Serial.println("WiFi not connected");
    }
}

