// #include "BluetoothSerial.h"
// BluetoothSerial SerialBT;

// Motor 1
#define MOTOR1_DIR1 13 // 100% 
#define MOTOR1_DIR2 14 // 100%
#define MOTOR1_PWM  33 // 100%

// Motor 2
#define MOTOR2_DIR1 27 // 100
#define MOTOR2_DIR2 26 // 100
#define MOTOR2_PWM  25 // 100%

// #define BUTTON_PIN 32
#define LED_PIN 2 // Built-in LED on most ESP32 boards

// int speed1 = 0;
// int speed2 = 0;

// bool motorsEnabled = true;
// bool lastButtonState = HIGH;

void setup() {
  // Set motor direction pins as outputs:
  pinMode(MOTOR1_DIR1, OUTPUT);
  pinMode(MOTOR1_DIR2, OUTPUT);
  pinMode(MOTOR2_DIR1, OUTPUT);
  pinMode(MOTOR2_DIR2, OUTPUT);

  // Set the PWM pins as outputs (for analogWrite):
  pinMode(MOTOR1_PWM, OUTPUT);
  pinMode(MOTOR2_PWM, OUTPUT);
}

void loop() {
  digitalWrite(LED_PIN, HIGH);
  // Example: Run Motor 1 forward at 50% speed
  digitalWrite(MOTOR1_DIR1, HIGH);
  digitalWrite(MOTOR1_DIR2, LOW);
  analogWrite(MOTOR1_PWM, 50); // 128 out of 255 ~50% speed

  // Example: Run Motor 2 in reverse at full speed
  digitalWrite(MOTOR2_DIR1, LOW);
  digitalWrite(MOTOR2_DIR2, HIGH);
  analogWrite(MOTOR2_PWM, 50); // Full speed

  delay(2000);

  digitalWrite(MOTOR1_DIR1, LOW);
  digitalWrite(MOTOR1_DIR2, HIGH);
  analogWrite(MOTOR1_PWM, 50); // 128 out of 255 ~50% speed

  // Example: Run Motor 2 in reverse at full speed
  digitalWrite(MOTOR2_DIR1, HIGH);
  digitalWrite(MOTOR2_DIR2, LOW);
  analogWrite(MOTOR2_PWM, 50); // Full speed

  delay(2000);
  
  // Stop both motors
  analogWrite(MOTOR1_PWM, 0); // Stop Motor 1
  analogWrite(MOTOR2_PWM, 0); // Stop Motor 2
  digitalWrite(LED_PIN, LOW);
  delay(2000);
}
