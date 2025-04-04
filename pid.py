import time
import pygame
import math
import json
import csv
from agent import LineFollowerPID

# Initialize Pygame
pygame.init()

# Load track image
line_path = pygame.image.load("imgs/line_paths/test.png")
screen = pygame.display.set_mode(line_path.get_size())
pygame.display.set_caption("PID Line Follower - Optimized")

def run_pid():
    global attempt_id

    # Specify the file name
    file_name = 'PID_logged_data.csv'

    # Define the header (if creating a new file)
    header = ['attempt_id', 'step', 'sensor0', 'sensor1', 'sensor2', 'sensor3', 'sensor4',
             'sensor5', 'sensor6', 'sensor7', 'sensor8', 'sensor9',
             'sensor10', 'sensor11', 'sensor12', 'sensor13', 'sensor14',
             'action_L', 'action_R']

    # Open the CSV file in append mode ('a') to avoid overwriting
    with open(file_name, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        # Write header only if the file is empty (initial run)
        if file.tell() == 0:  # check if the file is empty
            writer.writerow(header)

    # Load full configuration
    with open("agent/config_line_follower.json", "r") as f:
        config = json.load(f)

    # Initialize robot with full config
    line_follower = LineFollowerPID(config, screen)
    
    total_error = 0
    running = True
    clock = pygame.time.Clock()
    
    # Record start time in milliseconds
    start_time = pygame.time.get_ticks()
    
    while running:
        dt = clock.tick(60) / 1000.0
        
        # Check if 10 seconds have passed (15000 milliseconds)
        if (pygame.time.get_ticks() - start_time) >= 15000:
            running = False
            break


        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
        if pygame.key.get_pressed()[pygame.K_ESCAPE]:
            time.sleep(0.5)  # Prevent immediate exit
            running = False
            break

        # Simulation update
        screen.blit(line_path, (0, 0))
        try:
            # reached the end 
            if line_follower.get_color() == "green": # Green color detected
                running = False
                break
            sensor_readings = line_follower.get_line_sensor_readings()

        except IndexError:
            print("Robot out of bounds!")
            running = False
            break
        action = line_follower.compute_pid_action(sensor_readings)
        line_follower.apply_action(action)
        line_follower.step(dt)
        line_follower.draw(img_path="agent/robot.png", draw_robot=True)

        # Debug display
        font = pygame.font.SysFont(None, 24)
        debug_text = [
            f"Kp: {line_follower.Kp:.2f}  Ki: {line_follower.Ki:.2f}  Kd: {line_follower.Kd:.2f}",
            f"Motors: L={action[0]:.2f}  R={action[1]:.2f}",
            f"Error: {line_follower.prev_error:.2f}  Integral: {line_follower.integral:.2f}"
        ]
        y_offset = 10
        for text in debug_text:
            surf = font.render(text, True, (255, 0, 0))
            screen.blit(surf, (10, y_offset))
            y_offset += 25

        total_error += abs(line_follower.prev_error)

        # Log data to CSV
        with open(file_name, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([attempt_id] + [line_follower.step_count] + sensor_readings.tolist() + action)

        line_follower.step_count += 1

        # Update the display
        pygame.display.flip()


    print(f"Kp: {line_follower.Kp:.2f}  Ki: {line_follower.Ki:.2f}  Kd: {line_follower.Kd:.2f}")
    print(f"Total error: {total_error:.2f}")
    print("-"*40)

    attempt_id += 1
    line_follower.step_count = 0

    run_pid()
    pygame.quit()

if __name__ == "__main__":
    attempt_id = 0
    run_pid()