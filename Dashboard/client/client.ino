#include <WiFi.h>

const char* ssid = "Damamax Fiber_72AF";
const char* password = "atout4wifi";
const char* serverIP = "192.168.100.25"; // Replace with your PC's local IP address
const int serverPort = 12345;

WiFiClient client;

void setup() {
  Serial.begin(115200);
  
  // Connect to Wi-Fi
  Serial.println("Connecting to Wi-Fi...");
  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.println("Connecting...");
  }

  Serial.println("Connected to Wi-Fi");
  Serial.print("ESP32 IP Address: ");
  Serial.println(WiFi.localIP());

  // Connect to the server (PC)
  if (client.connect(serverIP, serverPort)) {
    Serial.println("Connected to server");
  } else {
    Serial.println("Connection to server failed");
  }

  // Seed the random number generator
  randomSeed(analogRead(0));  
}

void loop() {
  if (!client.connected()) {
    Serial.println("Lost connection to server. Reconnecting...");
    if (client.connect(serverIP, serverPort)) {
      Serial.println("Reconnected to server");
    } else {
      Serial.println("Reconnection failed");
      delay(1000);
      return;
    }
  }

  String message = String(random(0, 2));
  for (int i=0; i<14; i++){
    // Generate a random number
    int randomNumber = random(0, 2); // Generate a random number between 0 and 99
    message += "," + String(randomNumber);
  }
  // Send data to the server (PC)
  client.print(message);
  Serial.println("Sent to server: " + message);

  // Check for incoming data from the server
  if (client.available()) {
    String incomingData = client.readStringUntil('\n');
    Serial.print("Received from server: ");
    Serial.println(incomingData);
  }

  delay(10);  // Wait before sending next data
}
