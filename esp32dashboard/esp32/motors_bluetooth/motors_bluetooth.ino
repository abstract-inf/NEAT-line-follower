#include "BluetoothSerial.h"
BluetoothSerial SerialBT;

// Motor 1
#define MOTOR1_DIR1 13
#define MOTOR1_DIR2 14
#define MOTOR1_PWM  33

// Motor 2
#define MOTOR2_DIR1 27
#define MOTOR2_DIR2 26
#define MOTOR2_PWM  25

// #define LED_PIN 2

void setup() {
  pinMode(MOTOR1_DIR1, OUTPUT);
  pinMode(MOTOR1_DIR2, OUTPUT);
  pinMode(MOTOR2_DIR1, OUTPUT);
  pinMode(MOTOR2_DIR2, OUTPUT);
  pinMode(MOTOR1_PWM, OUTPUT);
  pinMode(MOTOR2_PWM, OUTPUT);
  // pinMode(LED_PIN, OUTPUT);

  SerialBT.begin("ESP32-Motors");
  // Serial.begin(115200);
}

void loop() {
  if (SerialBT.available()) {
    char input[20]; // Buffer for input
    int len = SerialBT.readBytesUntil('\n', input, sizeof(input)-1);
    input[len] = '\0'; // Null-terminate

    int speed1, speed2;
    if (sscanf(input, "%d,%d", &speed1, &speed2) == 2) {
      speed1 = constrain(speed1, -150, 150); // Limit speed
      speed2 = constrain(speed2, -150, 150);
      
      setMotor(MOTOR1_DIR1, MOTOR1_DIR2, MOTOR1_PWM, speed1);
      setMotor(MOTOR2_DIR1, MOTOR2_DIR2, MOTOR2_PWM, speed2);
    }
  }
}


void setMotor(int dir1, int dir2, int pwm, int speed) {
  if (speed > 0) {
    digitalWrite(dir1, HIGH);
    digitalWrite(dir2, LOW);
  } else if (speed < 0) {
    digitalWrite(dir1, LOW);
    digitalWrite(dir2, HIGH);
  } else {
    digitalWrite(dir1, LOW);
    digitalWrite(dir2, LOW);
  }
  analogWrite(pwm, abs(constrain(speed, -255, 255)));
}
