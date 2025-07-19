#include <Wire.h>
#include <I2Cdev.h>
#include <MPU6050_6Axis_MotionApps20.h>
#include <BluetoothSerial.h>

// Forward declaration for interrupt handler
void dmpDataReady();

// ======= Hardware Configuration =======
// MPU6050
#define I2C_SDA 21
#define I2C_SCL 22
MPU6050 mpu;

// Motors
#define MOTOR1_DIR1 14
#define MOTOR1_DIR2 13
#define MOTOR1_PWM 33
#define MOTOR2_DIR1 26
#define MOTOR2_DIR2 27
#define MOTOR2_PWM 25

// Line Sensors
#define s0 18 // was 2
#define s1 19 // was 15
#define s2 4
#define s3 16
#define Sens 17

// ======= Global Variables =======
BluetoothSerial SerialBT;

// MPU6050
bool dmpReady = false;
uint8_t fifoBuffer[64];
Quaternion q;
VectorFloat gravity;
float ypr[3];
float ypr_offset[3] = {0};
volatile bool mpuInterrupt = false;

// PID Control
float kp = 4.0, ki = 0.0, kd = 0.2;
float prev_error = 0;
float integral = 0;
unsigned long last_pid_time = 0;

int BASE_SPEED = 150;
int MAX_SPEED = 255;
int MIN_SPEED = -255;

bool calibrate_flag = false;
bool running = false;

// Current Motor Speeds
int currentLeftSpeed = 0;
int currentRightSpeed = 0;

// Manual Control
bool manualMode = false;
int manualLeftSpeed = 0;
int manualRightSpeed = 0;

// Angle Control
bool angleControlMode = false;
float targetYaw = 0.0;

// Timing
unsigned long last_send_time = 0;
const unsigned long send_interval = 100; // 100ms

void setup() {
    Serial.begin(115200);
    initMPU6050();
    initMotors();
    initLineSensors();
    initBluetooth();
}

void initMPU6050() {
    Wire.begin(I2C_SDA, I2C_SCL, 400000);
    mpu.initialize();
    // pinMode(2, INPUT);
    
    if (mpu.dmpInitialize() == 0) {
        mpu.setDMPEnabled(true);
        dmpReady = true;
        // attachInterrupt(digitalPinToInterrupt(2), dmpDataReady, RISING);
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

void loop() {
    // Read sensors
    readIMU();
    int16_t sensor_values[15];
    readLineSensors(sensor_values);

    // Handle Bluetooth commands
    checkBluetooth();

    // Send data to dashboard
    if (millis() - last_send_time >= send_interval) {
        sendIMUData();
        sendLineSensorData(sensor_values);
        sendMotorData();
        last_send_time = millis();
    }

    // Motor control
    if (manualMode) {
        setMotor(MOTOR1_DIR1, MOTOR1_DIR2, MOTOR1_PWM, currentLeftSpeed);
        setMotor(MOTOR2_DIR1, MOTOR2_DIR2, MOTOR2_PWM, currentRightSpeed);
    }
    else if (angleControlMode) {
        handleAngleControl();
    }
    else if (running) { // PID Mode
       // ======= PID Control Logic =======
        // 1. Calculate error from line sensors
        float error = calculateLineError(sensor_values); // Implement this
        
        // 2. Compute time delta (dt) in seconds
        unsigned long now = millis();
        float dt = (now - last_pid_time) / 1000.0;
        last_pid_time = now;

        // 3. Compute PID terms
        float proportional = kp * error;
        integral += ki * error * dt;
        float derivative = kd * (error - prev_error) / dt;
        prev_error = error;

        // Prevent motor jitter when the error is near zero
        if (abs(error) < 0.5) { // Adjust threshold based on testing
            error = 0;
            integral = 0; // Reset integral when centered
        }

        // 4. Calculate total PID output
        float pid_output = proportional + integral + derivative;

        // 5. Apply to motors with differential steering
        currentLeftSpeed  = constrain(BASE_SPEED - pid_output, MIN_SPEED, MAX_SPEED);
        currentRightSpeed = constrain(BASE_SPEED + pid_output, MIN_SPEED, MAX_SPEED);
        

        setMotor(MOTOR1_DIR1, MOTOR1_DIR2, MOTOR1_PWM, currentLeftSpeed);
        setMotor(MOTOR2_DIR1, MOTOR2_DIR2, MOTOR2_PWM, currentRightSpeed);
    }
    else {
        // Stop motors when not running
        prev_error = 0;  // Reset PID history
        integral = 0;

        currentLeftSpeed = 0;
        currentRightSpeed = 0;
        setMotor(MOTOR1_DIR1, MOTOR1_DIR2, MOTOR1_PWM, 0);
        setMotor(MOTOR2_DIR1, MOTOR2_DIR2, MOTOR2_PWM, 0);
    }

    if (calibrate_flag) {
        calibrateAngles();
        calibrate_flag = false;
    }
}

// function to calculate line error for digital sensor
float calculateLineError(int16_t* sensor_values) {
    // Weighted error calculation for digital sensors
    float weights[] = {-3, -2.5, -2, -1.5, -1, -0.5, 0, 0, 0.5, 1, 1.5, 2, 2.5, 3}; // Centered around sensor 7
    float error = 0;
    int active_sensors = 0;

    for (int i = 0; i < 15; i++) {
        if (sensor_values[i]) {
            error += weights[i];
            active_sensors++;
        }
    }

    if (active_sensors == 0) {
        // No line detected - use last known error
        return (prev_error > 0) ? 3.0 : -3.0;
    }

    return error / active_sensors; // Normalized error
}

// ======= Sensor Functions =======
void readIMU() {
    if (dmpReady && mpu.dmpGetCurrentFIFOPacket(fifoBuffer)) {
        mpu.dmpGetQuaternion(&q, fifoBuffer);
        mpu.dmpGetGravity(&gravity, &q);
        mpu.dmpGetYawPitchRoll(ypr, &q, &gravity);
        // Apply calibration offsets
        for (int i = 0; i < 3; i++) {
            ypr[i] -= ypr_offset[i];
        }
    }
}

void readLineSensors(int16_t* values) {
    for (int i = 0; i < 15; i++) {
        digitalWrite(s0, bitRead(i, 0));
        digitalWrite(s1, bitRead(i, 1));
        digitalWrite(s2, bitRead(i, 2));
        digitalWrite(s3, bitRead(i, 3));
        delayMicroseconds(100);
        values[i] = digitalRead(Sens);
    }
}

// ======= Bluetooth Communication =======
void checkBluetooth() {
    if (SerialBT.available()) {
        String command = SerialBT.readStringUntil('\n');
        command.trim();
        
        if (command.startsWith("PID:")) {
            parsePID(command);
        }
        else if (command == "RESET") {
            running = false;
            manualMode = false;
            angleControlMode = false;
            SerialBT.println("STATUS: Reset complete");
        }
        else if (command == "START") {
            running = true;
            manualMode = false;
            angleControlMode = false;
            SerialBT.println("STATUS: Started");
        }
        else if (command == "STOP") {
            running = false;
            manualMode = false;
            angleControlMode = false;
            SerialBT.println("STATUS: Stopped");
        }
        else if (command == "CALIBRATE") {
            calibrate_flag = true;
            SerialBT.println("STATUS: Calibrating...");
        }
        else if (command.indexOf(',') != -1) {
            // Manual speed command
            int commaIndex = command.indexOf(',');
            int r_speed = command.substring(0, commaIndex).toInt();
            int l_speed = command.substring(commaIndex+1).toInt();
            
            currentLeftSpeed = l_speed;
            currentRightSpeed = r_speed;
            manualMode = true;
            running = false;
            angleControlMode = false;
            SerialBT.println("STATUS: Manual speed set");
            sendMotorData();
        }
        else if (command.startsWith("SET_ANGLE ")) {
            // Angle control command
            int spaceIndex = command.indexOf(' ');
            if (spaceIndex != -1) {
                targetYaw = command.substring(spaceIndex+1).toFloat();
                angleControlMode = true;
                manualMode = false;
                running = false;
                SerialBT.println("STATUS: Rotating to target angle");
            }
        }
    }
}

void parsePID(String command) {
    command = command.substring(4);
    int comma1 = command.indexOf(',');
    int comma2 = command.lastIndexOf(',');
    
    if (comma1 != -1 && comma2 != -1) {
        kp = command.substring(0, comma1).toFloat();
        ki = command.substring(comma1+1, comma2).toFloat();
        kd = command.substring(comma2+1).toFloat();
        SerialBT.print("STATUS: PID Updated - KP:");
        SerialBT.print(kp);
        SerialBT.print(" KI:");
        SerialBT.print(ki);
        SerialBT.print(" KD:");
        SerialBT.println(kd);
    }
}

// ======= Angle Control =======
void handleAngleControl() {
    float currentYawRad = ypr[0]; // Already calibrated and in radians
    float currentYawDeg = currentYawRad * 180.0 / M_PI; // Convert to degrees
    float error = targetYaw - currentYawDeg; // Both in degrees now

    // Normalize error to [-180, 180]
    error = fmod(error, 360.0);
    if (error > 180.0) error -= 360.0;
    else if (error < -180.0) error += 360.0;

    float tolerance = 4.0; // Degrees
    if (fabs(error) <= tolerance) {
        setMotor(MOTOR1_DIR1, MOTOR1_DIR2, MOTOR1_PWM, 0);
        setMotor(MOTOR2_DIR1, MOTOR2_DIR2, MOTOR2_PWM, 0);
        angleControlMode = false;
        SerialBT.println("STATUS: Target angle reached");
    } else {
        int speed = 125;
        if (error > 0) { // Clockwise
            setMotor(MOTOR1_DIR1, MOTOR1_DIR2, MOTOR1_PWM, -speed);
            setMotor(MOTOR2_DIR1, MOTOR2_DIR2, MOTOR2_PWM, speed);
        } else { // Counter-clockwise
            setMotor(MOTOR1_DIR1, MOTOR1_DIR2, MOTOR1_PWM, speed);
            setMotor(MOTOR2_DIR1, MOTOR2_DIR2, MOTOR2_PWM, -speed);
        }
    }
}

// ======= Data Sending Functions =======
void sendIMUData() {
    // Send Yaw, Pitch, Roll
    SerialBT.print("YPR:");
    SerialBT.print(ypr[0] * 180/M_PI);
    SerialBT.print(",");
    SerialBT.print(ypr[1] * 180/M_PI);
    SerialBT.print(",");
    SerialBT.println(ypr[2] * 180/M_PI);

    // Send Acceleration
    VectorInt16 accel;
    mpu.getAcceleration(&accel.x, &accel.y, &accel.z);
    SerialBT.print("ACCEL:");
    SerialBT.print(accel.x);
    SerialBT.print(",");
    SerialBT.print(accel.y);
    SerialBT.print(",");
    SerialBT.println(accel.z);
}

void sendLineSensorData(int16_t* values) {
    SerialBT.print("LINE:");
    for (int i = 0; i < 15; i++) {
        SerialBT.print(values[i]);
        if (i < 14) SerialBT.print(",");
    }
    SerialBT.println();
}

void sendMotorData() {
    SerialBT.print("MOTORS:");
    SerialBT.print(currentLeftSpeed);
    SerialBT.print(",");
    SerialBT.println(currentRightSpeed);
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
    SerialBT.println("STATUS: Calibration complete");
}

// ======= Motor Control =======
void setMotor(int dir1, int dir2, int pwmPin, int speed) {
    speed = constrain(speed, MIN_SPEED, MAX_SPEED);
    digitalWrite(dir1, speed > 0 ? HIGH : LOW);
    digitalWrite(dir2, speed > 0 ? LOW : HIGH);
    analogWrite(pwmPin, abs(speed));
}

// ======= Interrupt Handler =======
void IRAM_ATTR dmpDataReady() {
    mpuInterrupt = true;
}