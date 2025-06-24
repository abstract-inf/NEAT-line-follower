# Updated NEAT code with logging
import time
import pygame
import neat
import os
import pickle
import json
import csv  # Added import
from agent import LineFollowerNEAT
from environment.track import VirtualTrack, Viewport

def run_example(genome, config):
    # CSV logging setup
    file_name = 'NEAT_logged_data.csv'
    header = ['attempt_id', 'step', 'sensor0', 'sensor1', 'sensor2', 'sensor3', 'sensor4',
             'sensor5', 'sensor6', 'sensor7', 'sensor8', 'sensor9',
             'sensor10', 'sensor11', 'sensor12', 'sensor13', 'sensor14',
             'action_L', 'action_R', 'angular_velocity', 'angular_acceleration']
    
    if not os.path.exists(file_name):
        with open(file_name, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
    
    attempt_id = 0  # Set based on your needs

    # Pygame and track initialization
    viewport = Viewport(1200, 800)
    track_path = "environment/tracks/track for the paper.png"
    current_track = VirtualTrack(track_path)
    viewport.update_world_size(current_track.width, current_track.height)
    temp_surface = pygame.Surface((current_track.width, current_track.height))
    temp_surface.blit(current_track.surface, (0, 0))

    with open("agent/config_line_follower.json", "r") as f:
        robot_config = json.load(f)

    line_follower = LineFollowerNEAT(
        genome=genome,
        neat_config=config,
        robot_config=robot_config,
        screen=temp_surface,
        sensor_type="semi_circle",
        draw_robot=True,
        img_path="agent/robot.png"
    )
    line_follower.max_speed = 253

    # Logging variables
    step_count = 0
    previous_angular_velocity = 0.0

    # Sensor frequency control
    target_sensor_frequency = 30.0
    sensor_read_interval = 1.0 / target_sensor_frequency
    time_since_last_sensor_read = 0.0

    # Main loop
    clock = pygame.time.Clock()
    running = True
    accumulator = 0.0
    fixed_dt = 1/60.0
    last_time = pygame.time.get_ticks() / 1000.0
    
    has_finished = False # Flag to indicate the robot has reached the end of the track

    while running and not has_finished:
        current_time = pygame.time.get_ticks() / 1000.0
        frame_time = current_time - last_time
        last_time = current_time
        accumulator += frame_time

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            viewport.handle_events(event)

        while accumulator >= fixed_dt:
            pygame.event.pump() # Keep Pygame's event system alive
            dt = fixed_dt
            time_since_last_sensor_read += dt
            logged_sensors = False
            sensor_readings = None

            temp_surface.blit(current_track.full_image, (0, 0))

            print(line_follower.check_middle_sensor_color())
            if line_follower.check_middle_sensor_color() == "green":
                has_finished = True
                print("Robot has reached the end of the track.")
                break

            if time_since_last_sensor_read >= sensor_read_interval:
                sensor_readings = line_follower.get_line_sensor_readings()
                time_since_last_sensor_read -= sensor_read_interval
                logged_sensors = True

            accumulator -= fixed_dt
            
            line_follower.step(dt)

            if logged_sensors:
                # Get motor actions
                action_L = line_follower.left_wheel_velocity
                action_R = line_follower.right_wheel_velocity

                # Get velocity
                linear_velocity, angular_velocity = line_follower.get_velocity()

                # Calculate angular acceleration
                angular_acceleration = (angular_velocity - previous_angular_velocity) / dt
                previous_angular_velocity = angular_velocity

                # Log data
                with open(file_name, 'a', newline='') as f:
                    writer = csv.writer(f)
                    row_data = [attempt_id, step_count] + sensor_readings.tolist() + [action_L, action_R, angular_velocity, angular_acceleration]
                    writer.writerow(row_data)

                step_count += 1


        # Rendering (unchanged)
        temp_surface.blit(current_track.full_image, (0, 0))
        line_follower.draw()
        viewport.apply(temp_surface)
        pygame.display.flip()
        clock.tick(60)
        
    pygame.quit()

if __name__ == "__main__":
    pygame.init()
    local_dir = os.path.dirname(__file__)
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        os.path.join(local_dir, 'config.txt')
    )

    # Load best genome
    import glob
    genome_files = glob.glob("neat_results/models/best_genome_*.pkl")
    latest_genome = max(genome_files, key=os.path.getctime)

    with open(latest_genome, "rb") as f:
        winner = pickle.load(f)

    print(f"Loaded genome from {latest_genome}")
    run_example(winner, config)