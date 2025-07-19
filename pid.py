# pid.py 
import pygame 
import json 
import csv 
import os 
from environment.track import VirtualTrack, Viewport 
from agent import LineFollowerPID 

# Initialize Pygame 
DISPLAY_TRAINING_WINDOW = True # Set to False for headless mode (no GUI)

if not DISPLAY_TRAINING_WINDOW: 
    os.environ["SDL_VIDEODRIVER"] = "dummy" 

pygame.init() 

if DISPLAY_TRAINING_WINDOW: 
    pygame.display.set_caption("PID Line Follower") 
    viewport = Viewport(1200, 875, DISPLAY_TRAINING_WINDOW) 
else: 
    viewport = Viewport(1, 1, DISPLAY_TRAINING_WINDOW) 

current_track = None 

def run_pid(speed, attempt_id): 
    file_name = 'PID_logged_data_non_headless.csv' 
    header = ['speed', 'attempt_id', 'step', 'sensor0', 'sensor1', 'sensor2', 'sensor3', 'sensor4', 
              'sensor5', 'sensor6', 'sensor7', 'sensor8', 'sensor9', 
              'sensor10', 'sensor11', 'sensor12', 'sensor13', 'sensor14', 
              'action_L', 'action_R'] 

    if not os.path.exists(file_name): 
        with open(file_name, 'w', newline='') as f: 
            writer = csv.writer(f) 
            writer.writerow(header) 

    with open("agent/config_line_follower.json", "r") as f: 
        config = json.load(f) 

    track_path = "environment/tracks/track for the paper.png" 
    current_track = VirtualTrack(track_path) 
    viewport.update_world_size(current_track.width, current_track.height) 
    temp_surface = pygame.Surface((current_track.width, current_track.height)) 
        
    line_follower = LineFollowerPID(config, temp_surface) 
    line_follower.max_speed = speed 
    total_error = 0 
    running = True 
    clock = pygame.time.Clock() 
    start_time = pygame.time.get_ticks() 
    
    # Initialize font only if displaying
    if DISPLAY_TRAINING_WINDOW:
        font = pygame.font.SysFont("Arial", 24)

    # --- OPTIMIZATION: Open file once before the loop ---
    with open(file_name, 'a', newline='') as file:
        writer = csv.writer(file)

        while running: 
            temp_surface.blit(current_track.surface, (0, 0))
            # --- OPTIMIZATION: Use fixed dt for headless mode, cap FPS only when displaying ---
            if DISPLAY_TRAINING_WINDOW:
                dt = clock.tick(60) / 1000.0 
            else:
                dt = 1/60.0 # Use a fixed, small timestep for simulation stability

            for event in pygame.event.get(): 
                if event.type == pygame.QUIT: 
                    running = False 
                elif DISPLAY_TRAINING_WINDOW: 
                    viewport.handle_events(event) 

            # Simulation logic
            if (pygame.time.get_ticks() - start_time) >= 100000: # 100 seconds
                running = False
                print(f"\033[91m[speed {speed}, attempt {attempt_id}] Robot took too long / is turning around itself.\033[0m") 
                break

            middle_sensor_color = line_follower.check_middle_sensor_color()
            if middle_sensor_color == "green": 
                running = False 
                print(f"\033[92m[speed {speed}, attempt {attempt_id}] Robot has reached the end of the track.\033[0m") 
                break 
            # elif middle_sensor_color == "red": 
            #     print(f"\033[91m[speed {speed}, attempt {attempt_id}] Robot has gone off the track. Stopping simulation.\033[0m") 
            #     running = False 
            #     break 

            # if line_follower.get_line_sensor_readings() == [0] * 15:
            #     print(f"\033[91m[speed {speed}, attempt {attempt_id}] Robot is out of bounds. Stopping simulation.\033[0m") 
            #     running = False 
            #     break

            try: 
                sensor_readings = line_follower.get_line_sensor_readings() 
            except IndexError: 
                print("Robot out of bounds!") 
                running = False 
                break # Exit loop immediately on error

            action = line_follower.compute_pid_action(sensor_readings, dt) 
            line_follower.apply_action(action, dt) 
            line_follower.step(dt) 
            post_step_readings = line_follower.get_line_sensor_readings()
            
            # --- OPTIMIZATION: Logging is now much faster ---
            total_error += abs(line_follower.prev_error) 
            # Correctly writes data matching the header
            writer.writerow([speed, attempt_id, line_follower.step_count] + sensor_readings.tolist() + action) 
            
            # --- OPTIMIZATION: All graphics operations are skipped in headless mode ---
            if DISPLAY_TRAINING_WINDOW:
                viewport.handle_viewport_controls()
                # Redraw background to clear previous frame's artifacts
                temp_surface.blit(current_track.surface, (0, 0))
                
                line_follower.draw(img_path="agent/robot.png", draw_robot=True) 

                debug_text = [ 
                    f"Kp: {line_follower.Kp:.2f}  Ki: {line_follower.Ki:.2f}  Kd: {line_follower.Kd:.2f}", 
                    f"Motors: L={action[0]:.2f}  R={action[1]:.2f}", 
                    f"Error: {line_follower.prev_error:.2f}  Integral: {line_follower.integral:.2f}",
                    f"FPS: {int(clock.get_fps())}"
                ] 

                y_offset = 10 
                for text in debug_text: 
                    surf = font.render(text, True, (255, 0, 0)) 
                    temp_surface.blit(surf, (10, y_offset)) 
                    y_offset += 25 

                viewport.apply(temp_surface) 
                pygame.display.flip() 

            line_follower.step_count += 1 

    # print(f"Kp: {line_follower.Kp:.2f}  Ki: {line_follower.Ki:.2f}  Kd: {line_follower.Kd:.2f}") 
    # print(f"Total error: {total_error:.2f}") 
    # print("-"*40) 

    # print(f"\033[91m[speed {speed}, attempt {attempt_id}] Robot has gone off the track. Stopping simulation.\033[0m") 

if __name__ == "__main__": 
    for speed in [100, 150, 200, 250, 300, 350, 400, 450, 500]: 
        for attempt_id in range(10): 
            # print(f"Running PID with attempt_id: {attempt_id} and speed: {speed}") 
            run_pid(speed, attempt_id) 
    pygame.quit()