import pygame
import math
import sys
import random

# -------------------------
# Simulation and Motor Parameters
# -------------------------
dt = 0.025  # time step in seconds

# Motor physical parameters
max_rpm = 13000           # Maximum motor RPM (unloaded)
gear_ratio = 30.0         # Reduction ratio (motor to wheel)
wheel_diameter = 0.05     # Wheel diameter in meters (e.g., 5 cm)
wheel_radius = wheel_diameter / 2.0

# Calculate effective maximum wheel speed (in m/s)
effective_rps = (max_rpm / gear_ratio) / 60.0
max_linear_speed = effective_rps * (2 * math.pi * wheel_radius)  # in m/s

# Base motor response parameters
tau = 0.2                 # Time constant for motor speed response (seconds)
base_friction_coef = 0.5  # Base damping coefficient for friction effects

# Differential drive parameters
wheel_base = 0.1          # Distance between the wheels in meters

# For rendering: conversion from meters to pixels
meter_to_pixel = 100      # 1 meter = 100 pixels

# Additional realistic parameters
static_friction_threshold = 0.05   # Minimum difference to overcome static friction (m/s)
max_possible_acceleration = 2.0      # Maximum allowed acceleration (m/s²)

# -------------------------
# Robot Class
# -------------------------
class Robot:
    def __init__(self, x, y, theta):
        # Position in meters and orientation in radians
        self.x = x
        self.y = y
        self.theta = theta
        # Current speeds of left and right wheels in m/s
        self.v_left = 0.0
        self.v_right = 0.0
        # Accelerations (for visualization)
        self.a_left = 0.0
        self.a_right = 0.0

    def update_motor_speed(self, current_speed, target_pwm):
        """
        Update motor speed with additional realism.
          1. Convert PWM command to target speed using motor specs.
          2. Compute a nonlinear acceleration: the closer to max speed, the lower the acceleration.
          3. Account for friction.
          4. Clamp acceleration to a maximum value to simulate wheel slip.
          5. Apply a static friction threshold to prevent movement if force is too low.
        """
        target_speed = (target_pwm / 255.0) * max_linear_speed

        # Nonlinear acceleration: reduce acceleration as speed approaches max speed
        accel_raw = (target_speed - current_speed) / tau
        nonlinear_factor = (1 - current_speed / (max_linear_speed + 1e-6))**2
        effective_accel = accel_raw * nonlinear_factor

        # Use base friction coefficient
        friction_coef = base_friction_coef

        # Friction term (damping) proportional to current speed
        friction_term = friction_coef * current_speed

        net_acceleration = effective_accel - friction_term

        # Limit the net acceleration to simulate traction limitations (wheel slip)
        net_acceleration = max(min(net_acceleration, max_possible_acceleration), -max_possible_acceleration)

        # Apply static friction: if nearly stopped and force is too low, do not move
        if abs(current_speed) < 0.01 and abs(target_speed - current_speed) < static_friction_threshold:
            new_speed = 0.0
            net_acceleration = 0.0
        else:
            new_speed = current_speed + net_acceleration * dt

        return new_speed, net_acceleration

    def update(self, target_pwm_left, target_pwm_right):
        # Update each motor's speed and get acceleration values
        self.v_left, self.a_left = self.update_motor_speed(self.v_left, target_pwm_left)
        self.v_right, self.a_right = self.update_motor_speed(self.v_right, target_pwm_right)

        # -------------------------
        # Differential Drive Kinematics
        # -------------------------
        # Compute linear speed (v) and angular velocity (omega)
        v = (self.v_left + self.v_right) / 2.0
        omega = (self.v_right - self.v_left) / wheel_base

        # Update robot's pose (x, y in meters and theta in radians)
        self.x += v * math.cos(self.theta) * dt
        self.y += v * math.sin(self.theta) * dt
        self.theta += omega * dt

    def draw(self, screen):
        # Convert robot position from meters to pixels for display.
        x_pix = int(self.x * meter_to_pixel)
        y_pix = int(self.y * meter_to_pixel)
        # Define robot's visual size (e.g., a circle with 5cm radius scaled to pixels)
        robot_radius = int(0.05 * meter_to_pixel)
        pygame.draw.circle(screen, (0, 255, 0), (x_pix, y_pix), robot_radius)
        # Draw a heading line to indicate orientation.
        end_x = x_pix + int(robot_radius * math.cos(self.theta))
        end_y = y_pix + int(robot_radius * math.sin(self.theta))
        pygame.draw.line(screen, (255, 0, 0), (x_pix, y_pix), (end_x, end_y), 2)

# -------------------------
# Main Simulation Loop
# -------------------------
def main():
    pygame.init()

    WIDTH, HEIGHT = 1800, 900

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Enhanced Realistic Differential Drive Simulation")
    clock = pygame.time.Clock()

    # Create a font for on-screen text
    font = pygame.font.SysFont("Arial", 16)

    # Initialize the robot at the center of the screen (in meters)
    start_x = (WIDTH / meter_to_pixel) / 2
    start_y = (HEIGHT / meter_to_pixel) / 2
    robot = Robot(start_x, start_y, 0)

    # Initial PWM command values for left and right motors.
    target_pwm_left = 200
    target_pwm_right = 200

    running = True
    while running:
        clock.tick(100)  # Simulation running at 50 FPS

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_ESCAPE]:
            running = False
        
        # For demonstration: adjust PWM commands using arrow keys.
        if keys[pygame.K_UP]:
            target_pwm_left = 255
            target_pwm_right = 255
        elif keys[pygame.K_DOWN]:
            target_pwm_left = 100
            target_pwm_right = 100
        elif keys[pygame.K_LEFT]:
            target_pwm_left = 100
            target_pwm_right = 255
        elif keys[pygame.K_RIGHT]:
            target_pwm_left = 255
            target_pwm_right = 100
        else:
            target_pwm_left = 200
            target_pwm_right = 200

        # Update the robot's state (speeds and position)
        robot.update(target_pwm_left, target_pwm_right)

        # Clear the screen and draw the robot
        screen.fill((0, 0, 0))
        robot.draw(screen)

        # Compute overall robot speed (linear)
        robot_speed = (robot.v_left + robot.v_right) / 2.0

        # Prepare text surfaces for visualization
        text_left_speed = font.render(f"Left Speed: {robot.v_left:.2f} m/s", True, (255, 255, 255))
        text_right_speed = font.render(f"Right Speed: {robot.v_right:.2f} m/s", True, (255, 255, 255))
        text_robot_speed = font.render(f"Robot Speed: {robot_speed:.2f} m/s", True, (255, 255, 255))
        text_left_accel = font.render(f"Left Accel: {robot.a_left:.2f} m/s²", True, (255, 255, 255))
        text_right_accel = font.render(f"Right Accel: {robot.a_right:.2f} m/s²", True, (255, 255, 255))

        # Blit text surfaces onto the screen
        screen.blit(text_left_speed, (10, 10))
        screen.blit(text_right_speed, (10, 30))
        screen.blit(text_robot_speed, (10, 50))
        screen.blit(text_left_accel, (10, 70))
        screen.blit(text_right_accel, (10, 90))

        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
