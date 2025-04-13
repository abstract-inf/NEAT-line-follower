#include "I2Cdev.h"
#include "MPU6050_6Axis_MotionApps20.h"
#include "Wire.h"

#define I2C_SDA 21
#define I2C_SCL 22

MPU6050 mpu;

// DMP and orientation variables
bool dmpReady = false;
uint8_t fifoBuffer[64];
Quaternion q;
VectorFloat gravity;
float ypr[3];           // [yaw, pitch, roll]
float ypr_offset[3] = {0};

// Interrupt flag
volatile bool mpuInterrupt = false;
void IRAM_ATTR dmpDataReady() {
    mpuInterrupt = true;
}

void setup() {
    Serial.begin(115200);
    while (!Serial);

    Wire.begin(I2C_SDA, I2C_SCL, 400000);
    mpu.initialize();
    pinMode(2, INPUT);
    attachInterrupt(digitalPinToInterrupt(2), dmpDataReady, RISING);

    Serial.println(mpu.testConnection() ? "MPU6050 connected" : "MPU6050 connection failed");

    mpu.CalibrateAccel(10);
    mpu.CalibrateGyro(10);

    if (mpu.dmpInitialize() == 0) {
        mpu.setDMPEnabled(true);
        dmpReady = true;
        delay(1000);
        calibrateAngles(); // Grab offset after stabilization
    } else {
        Serial.println("DMP init failed");
    }
}

void calibrateAngles() {
    Serial.println("Calibrating angles...");
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
    Serial.println("Calibration done.");
}

void loop() {
    if (!dmpReady) return;

    if (mpu.dmpGetCurrentFIFOPacket(fifoBuffer)) {
        mpu.dmpGetQuaternion(&q, fifoBuffer);
        mpu.dmpGetGravity(&gravity, &q);
        mpu.dmpGetYawPitchRoll(ypr, &q, &gravity);

        float yaw = (ypr[0] - ypr_offset[0]) * 180.0 / M_PI;
        float pitch = (ypr[1] - ypr_offset[1]) * 180.0 / M_PI;
        float roll = (ypr[2] - ypr_offset[2]) * 180.0 / M_PI;

        // For Arduino Serial Plotter (CSV format or tab-separated)
        Serial.print(yaw); Serial.print("\t");
        Serial.print(pitch); Serial.print("\t");
        Serial.println(roll);
    }
}
