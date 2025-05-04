#include "../esp32_neat_rnn.h"
#include "I2Cdev.h"
#include "MPU6050_6Axis_MotionApps20.h"
#include "Wire.h"
#include "BluetoothSerial.h"

// ======= Hardware Configuration =======
// MPU6050
#define I2C_SDA 21
#define I2C_SCL 22
MPU6050 mpu;

// Motors
#define MOTOR1_DIR1 13
#define MOTOR1_DIR2 14
#define MOTOR1_PWM  33
#define MOTOR2_DIR1 27
#define MOTOR2_DIR2 26
#define MOTOR2_PWM  25

// Line Sensors
#define s0 2
#define s1 15
#define s2 4
#define s3 16
#define Sens 17

// ======= Global Variables =======
NEATRNN net;
BluetoothSerial SerialBT;

// MPU6050
bool dmpReady = false;
uint8_t fifoBuffer[64];
Quaternion q;
VectorFloat gravity;
float ypr[3];
float ypr_offset[3] = {0};
volatile bool mpuInterrupt = false;

// Performance Tracking
unsigned long loop_count = 0;
unsigned long last_report_time = 0;
const unsigned long report_interval = 1000;
unsigned long total_time = 0;
float min_time = 1000000.0, max_time = 0.0;


// ======= Interrupt Handlers =======
void IRAM_ATTR dmpDataReady() {
    mpuInterrupt = true;
}

// ======= Setup =======
void setup() {
    Serial.begin(115200);
    while (!Serial);
    
    // Initialize all systems
    initMPU6050();
    initMotors();
    initLineSensors();
    initBluetooth();
    
    Serial.println("\nSystem Initialization Complete");
    Serial.println("----------------------------");
}

void initMPU6050() {
    Wire.begin(I2C_SDA, I2C_SCL, 400000);
    mpu.initialize();
    pinMode(2, INPUT);
    
    if (mpu.dmpInitialize() == 0) {
        mpu.setDMPEnabled(true);
        dmpReady = true;
        attachInterrupt(digitalPinToInterrupt(2), dmpDataReady, RISING); // Moved here
        calibrateAngles();
    }
}

void initMotors() {
    pinMode(MOTOR1_DIR1, OUTPUT);
    pinMode(MOTOR1_DIR2, OUTPUT);
    pinMode(MOTOR2_DIR1, OUTPUT);
    pinMode(MOTOR2_DIR2, OUTPUT);
    pinMode(MOTOR1_PWM, OUTPUT);
    pinMode(MOTOR2_PWM, OUTPUT);
}

void initLineSensors() {
    pinMode(s0, OUTPUT);
    pinMode(s1, OUTPUT);
    pinMode(s2, OUTPUT);
    pinMode(s3, OUTPUT);
    pinMode(Sens, INPUT);
}

void initBluetooth() {
    SerialBT.begin("ESP32-Robot");
}

// ======= Main Loop =======
void loop() {
    unsigned long start_time = micros();
    
    // 1. Read all sensors
    readLineSensors();
    readIMU();
    
    // 2. Process through NEAT network
    processNEAT();
    
    // 3. Handle motor control
    controlMotors();
    
    // 4. Handle Bluetooth commands
    checkBluetooth();
    
    // Performance measurement
    unsigned long elapsed = micros() - start_time;
    updatePerformanceStats(elapsed);
    reportPerformance();
}

// ======= Sensor Functions =======
void readLineSensors() {
    for (int i = 0; i <= 15; i++) {
        digitalWrite(s0, bitRead(i, 0));
        digitalWrite(s1, bitRead(i, 1));
        digitalWrite(s2, bitRead(i, 2));
        digitalWrite(s3, bitRead(i, 3));
        delayMicroseconds(100); // Reduced from 1ms for faster reading
        net.inputs[i+3] = digitalRead(Sens); // Sensors start at input 3
    }
    // Print sensor values for debugging
    SerialBT.print("Sensor 0: "); SerialBT.print(net.inputs[3]);
    SerialBT.print(", Sensor 1: "); SerialBT.print(net.inputs[4]);
    SerialBT.print(", Sensor 2: "); SerialBT.print(net.inputs[5]);
    SerialBT.print(", Sensor 3: "); SerialBT.println(net.inputs[6]);
    SerialBT.print("Sensor 4: "); SerialBT.print(net.inputs[7]);
    SerialBT.print(", Sensor 5: "); SerialBT.print(net.inputs[8]);
    SerialBT.print(", Sensor 6: "); SerialBT.print(net.inputs[9]);
    SerialBT.print(", Sensor 7: "); SerialBT.println(net.inputs[10]);
    SerialBT.print("Sensor 8: "); SerialBT.print(net.inputs[11]);
    SerialBT.print(", Sensor 9: "); SerialBT.print(net.inputs[12]);
    SerialBT.print(", Sensor 10: "); SerialBT.print(net.inputs[13]);
    SerialBT.print(", Sensor 11: "); SerialBT.println(net.inputs[14]);
    SerialBT.print("Sensor 12: "); SerialBT.print(net.inputs[15]);
    SerialBT.print(", Sensor 13: "); SerialBT.print(net.inputs[16]);
    SerialBT.print(", Sensor 14: "); SerialBT.print(net.inputs[17]);
    SerialBT.print(", Sensor 15: "); SerialBT.println(net.inputs[18]);
    SerialBT.print("Sensor 16: "); SerialBT.print(net.inputs[19]);
    SerialBT.print(", Sensor 17: "); SerialBT.print(net.inputs[20]);
    SerialBT.println();
}

void readIMU() {
    if (mpuInterrupt && dmpReady && mpu.dmpGetCurrentFIFOPacket(fifoBuffer)) {
        mpu.dmpGetQuaternion(&q, fifoBuffer);
        mpu.dmpGetGravity(&gravity, &q);
        mpu.dmpGetYawPitchRoll(ypr, &q, &gravity);
        
        // Store IMU data in network inputs (adjust indices as needed)
        net.inputs[18] = ypr[0]; // Yaw
        net.inputs[19] = ypr[1]; // Pitch
        net.inputs[20] = ypr[2]; // Roll
        mpuInterrupt = false;
    }

    // print IMU data for debugging
    SerialBT.print("Yaw: "); SerialBT.print(ypr[0]);
    SerialBT.print(", Pitch: "); SerialBT.print(ypr[1]);
    SerialBT.print(", Roll: "); SerialBT.println(ypr[2]);
    SerialBT.println();

}

// ======= NEAT Processing =======
void processNEAT() {
    // First two inputs are previous outputs
    net.inputs[1] = net.get_output_0();
    net.inputs[2] = net.get_output_1();
    
    // Step the network
    net.step();
}

// ======= Motor Control =======
void controlMotors() {
    // Get outputs from network
    float left_motor = net.get_output_0();
    float right_motor = net.get_output_1();
    
    // Map to PWM values
    int pwm_left = (left_motor + 1.0f) * 127.5f;
    int pwm_right = (right_motor + 1.0f) * 127.5f;
    
    // Apply to motors
    setMotor(MOTOR1_DIR1, MOTOR1_DIR2, MOTOR1_PWM, pwm_left);
    setMotor(MOTOR2_DIR1, MOTOR2_DIR2, MOTOR2_PWM, pwm_right);

    // Debugging output
    SerialBT.print("Left Motor Speed: ");
    SerialBT.print(pwm_left);
    SerialBT.print(", Right Motor Speed: ");
    SerialBT.println(pwm_right);
}

void setMotor(int dir1, int dir2, int pwm, int speed) {
    speed = constrain(speed, -255, 255);
    digitalWrite(dir1, speed > 0 ? HIGH : LOW);
    digitalWrite(dir2, speed > 0 ? LOW : HIGH);
    analogWrite(pwm, abs(speed));
}

// ======= Bluetooth =======
void checkBluetooth() {
    if (SerialBT.available()) {
        char input[64];  // Larger buffer
        int len = SerialBT.readBytesUntil('\n', input, sizeof(input)-1);
        input[len] = '\0';
        
        int speed1, speed2;
        if (sscanf(input, "%d,%d", &speed1, &speed2) == 2) {
            setMotor(MOTOR1_DIR1, MOTOR1_DIR2, MOTOR1_PWM, speed1);
            setMotor(MOTOR2_DIR1, MOTOR2_DIR2, MOTOR2_PWM, speed2);
        }
    }
}

// ======= Performance Monitoring =======
void updatePerformanceStats(unsigned long elapsed) {
    loop_count++;
    total_time += elapsed;
    if (elapsed < min_time) min_time = elapsed;
    if (elapsed > max_time) max_time = elapsed;
}

void reportPerformance() {
    if (millis() - last_report_time >= report_interval) {
        float avg_time = (float)total_time / loop_count;
        float loops_per_second = 1000000.0 / avg_time;
        
        Serial.println("\n===== Performance Report =====");
        Serial.print("Average: "); Serial.print(avg_time); Serial.println(" μs/loop");
        Serial.print("Min: "); Serial.print(min_time); Serial.println(" μs/loop");
        Serial.print("Max: "); Serial.print(max_time); Serial.println(" μs/loop");
        Serial.print("Frequency: "); Serial.print(loops_per_second); Serial.println(" Hz");
        Serial.print("CPU Usage: "); Serial.print((avg_time/10000.0)*100); Serial.println("%");
        Serial.println("============================");

        SerialBT.println("\n===== Performance Report =====");
        SerialBT.print("Average: "); SerialBT.print(avg_time); SerialBT.println(" μs/loop");
        SerialBT.print("Min: "); SerialBT.print(min_time); SerialBT.println(" μs/loop");
        SerialBT.print("Max: "); SerialBT.print(max_time); SerialBT.println(" μs/loop");
        SerialBT.print("Frequency: "); SerialBT.print(loops_per_second); SerialBT.println(" Hz");
        SerialBT.print("CPU Usage: "); SerialBT.print((avg_time/10000.0)*100); SerialBT.println("%");
        SerialBT.println("============================");
        
        // Reset counters
        loop_count = 0;
        total_time = 0;
        min_time = 1000000.0;
        max_time = 0.0;
        last_report_time = millis();
    }
}

// ======= IMU Calibration =======
void calibrateAngles() {
    float sum[3] = {0};
    int samples = 100;

    for (int i = 0; i < samples; i++) {
        if (mpu.dmpGetCurrentFIFOPacket(fifoBuffer)) {
            mpu.dmpGetQuaternion(&q, fifoBuffer);
            mpu.dmpGetGravity(&gravity, &q);
            mpu.dmpGetYawPitchRoll(ypr, &q, &gravity);
            sum[0] += ypr[0];
            sum[1] += ypr[1];
            sum[2] += ypr[2];
            delay(10);
        }
    }

    for (int i = 0; i < 3; i++) ypr_offset[i] = sum[i] / samples;
}