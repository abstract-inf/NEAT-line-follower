# pid.py
import pygame
import json
import csv
import os
from environment.track import VirtualTrack, Viewport
from agent import LineFollowerPID

# Initialize Pygame
DISPLAY_TRAINING_WINDOW = True  # Set this to False to disable rendering

# Set environment variables FIRST
if not DISPLAY_TRAINING_WINDOW:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

# Initialize Pygame ONCE regardless of display mode
pygame.init()

# Screen setup
if DISPLAY_TRAINING_WINDOW:
    pygame.display.set_caption("PID Line Follower")
    viewport = Viewport(1200, 875, DISPLAY_TRAINING_WINDOW)
else:
    viewport = Viewport(1, 1, DISPLAY_TRAINING_WINDOW)

current_track = None

def run_pid():
    # File setup for logging
    file_name = 'PID_logged_data.csv'
    header = ['attempt_id', 'step', 'sensor0', 'sensor1', 'sensor2', 'sensor3', 'sensor4',
             'sensor5', 'sensor6', 'sensor7', 'sensor8', 'sensor9',
             'sensor10', 'sensor11', 'sensor12', 'sensor13', 'sensor14',
             'action_L', 'action_R']

    if not os.path.exists(file_name):
        with open(file_name, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)

    # Load configurations
    with open("agent/config_line_follower.json", "r") as f:
        config = json.load(f)

    # Initialize track and surfaces
    track_path = "environment/tracks/track for the paper.png"
    current_track = VirtualTrack(track_path)
    viewport.update_world_size(current_track.width, current_track.height)
    temp_surface = pygame.Surface((current_track.width, current_track.height))
    
    # Initialize robot
    line_follower = LineFollowerPID(config, temp_surface)
    total_error = 0
    running = True
    clock = pygame.time.Clock()
    start_time = pygame.time.get_ticks()

    while running:
        dt = clock.tick(60) / 1000.0
        temp_surface.blit(current_track.surface, (0, 0))

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif DISPLAY_TRAINING_WINDOW:
                viewport.handle_events(event)
        
        if line_follower.check_middle_sensor_color() == "green":
            running = False

        # Update viewport controls
        if DISPLAY_TRAINING_WINDOW:
            viewport.handle_viewport_controls()

        # Simulation logic
        if (pygame.time.get_ticks() - start_time) >= 600000:
            running = False

        try:
            if line_follower.check_middle_sensor_color() == "green":
                running = False
            sensor_readings = line_follower.get_line_sensor_readings()
        except IndexError:
            print("Robot out of bounds!")
            running = False

        # Compute and apply action
        action = line_follower.compute_pid_action(sensor_readings, dt)
        line_follower.apply_action(action, dt)
        
        # Move robot FIRST
        line_follower.step(dt)
        
        # Get POST-movement sensor positions for accurate drawing
        post_step_readings = line_follower.get_line_sensor_readings()  # Updates sensor coordinates
        
        # Draw with updated positions
        line_follower.draw(img_path="agent/robot.png", draw_robot=True)

        # Debug display
        font = pygame.font.SysFont("Arial", 24)
        debug_text = [
            f"Kp: {line_follower.Kp:.2f}  Ki: {line_follower.Ki:.2f}  Kd: {line_follower.Kd:.2f}",
            f"Motors: L={action[0]:.2f}  R={action[1]:.2f}",
            f"Error: {line_follower.prev_error:.2f}  Integral: {line_follower.integral:.2f}"
        ]
        
        # Add FPS display
        fps = int(clock.get_fps())
        debug_text.append(f"FPS: {fps}")

        y_offset = 10
        for text in debug_text:
            surf = font.render(text, True, (255, 0, 0))
            temp_surface.blit(surf, (10, y_offset))
            y_offset += 25

        # Apply viewport transformations
        if DISPLAY_TRAINING_WINDOW:
            viewport.apply(temp_surface)
            pygame.display.flip()

        # Logging
        total_error += abs(line_follower.prev_error)
        with open(file_name, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([attempt_id] + [line_follower.step_count] + 
                           sensor_readings.tolist() + action)

        line_follower.step_count += 1

    print(f"Kp: {line_follower.Kp:.2f}  Ki: {line_follower.Ki:.2f}  Kd: {line_follower.Kd:.2f}")
    print(f"Total error: {total_error:.2f}")
    print("-"*40)

if __name__ == "__main__":
    attempt_id = 10
    run_pid()
    pygame.quit()