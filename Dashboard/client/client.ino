#include <WiFi.h>
#include <ESPmDNS.h>

const char* ssid = "wifi_ssid";
const char* password = "wifi_password";

WiFiServer server(5000);

void setup() {
    Serial.begin(115200);
    WiFi.begin(ssid, password);
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    Serial.println("\nWi-Fi connected!");

    if (!MDNS.begin("esp32")) {
        Serial.println("Error starting mDNS");
        return;
    }
    Serial.println("mDNS responder started as esp32.local");
    server.begin();
}

void loop() {
    WiFiClient client = server.available();
    if (client) {
        // Serial.println("Client connected!");
        String command = "";
        
        // Set a timeout to wait for data and read the full command
        client.setTimeout(100); // Wait up to 100ms for data
        command = client.readString();
        command.trim(); // Remove any extra whitespace/newlines

        if (command.length() > 0) {
            Serial.print("Received command: ");
            Serial.println(command);
            
            if (command == "LED_ON") {
                client.println("LED turned ON");
            } else if (command == "LED_OFF") {
                client.println("LED turned OFF");
            } else {
                client.println("Unknown command");
            }
            client.flush(); // Ensure response is sent before closing
        }
        
        client.stop();
    }
}