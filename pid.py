# this code is a test on how to build a line follower AI agent using NEAT algorithm in a simulation using pygame

import pygame
import math
import random
import json
from agent import LineFollowerPID

# Initialize Pygame
pygame.init()

# Screen setup
# Load the line path image
line_path = pygame.image.load("imgs/line_paths/line_path0.png")
# set the screen size to the image size
screen = pygame.display.set_mode(line_path.get_size())
pygame.display.set_caption("PID Line Follower")


def applyPID(sensor_readings) -> list[float, float]:
    # PID Constants
    Kp = 1.0
    Ki = 0.0
    Kd = 0.0

    # PID Variables
    integral = 0
    prev_error = 0

    # PID Loop
    # calculate error
    sensor_count = len(sensor_readings)
    center_index = (sensor_count - 1) / 2.0
    weighted_sum = 0
    total_readings = 0
    for idx, reading in enumerate(sensor_readings):
        weight = idx - center_index
        weighted_sum += reading * weight
        total_readings += reading

    # Calculate PID
    error = weighted_sum / total_readings if total_readings != 0 else 0
    integral += error
    derivative = error - prev_error

    action = Kp * error + Ki * integral + Kd * derivative

    left_motor = 0.5 + action
    right_motor = 0.5 - action

    return [left_motor, right_motor]


def run_pid():
    with open("agent/config_line_follower.json", "r") as f:
        robot_config = json.load(f)

    line_follower = LineFollowerPID(robot_config, screen)

    running = True
    clock = pygame.time.Clock()
    while running:
        clock.tick(60)

        dt = clock.get_time() / 1000.0  # multiply by 1000 to convert to ms (e.g 16 to 0.016)
        # dt = min(dt, 1/30.0)  # capping for maintaing stability if the simulation becomes too intensive
        
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Quit game when pressing ESC
        keys = pygame.key.get_pressed()
        if keys[pygame.K_ESCAPE]:
            running = False
        
        # Fill the background with white
        # screen.fill((255, 255, 255))
        screen.blit(line_path, (0, 0))  # Draw the background line path once

        sensor_readings = line_follower.get_line_sensor_readings()

        action = applyPID(sensor_readings)

        line_follower.apply_action(action)

        line_follower.step(dt)
        line_follower.draw()


        # Update the display
        pygame.display.update()


if __name__ == "__main__":
    run_pid()