# Updated code with display initialization fix and controlled sensor frequency
import time
import pygame
import neat
import os
import pickle
import json
from agent import LineFollowerNEAT
from environment.track import VirtualTrack, Viewport

def run_example(genome, config):
    # Initialize Pygame display FIRST
    viewport = Viewport(1200, 800)  # This initializes the display surface

    # Now load track AFTER display exists
    track_path = "environment/tracks/train/test.png"
    current_track = VirtualTrack(track_path)

    # Update viewport with track dimensions
    viewport.update_world_size(current_track.width, current_track.height)

    # Create temporary surface for sensor accuracy
    temp_surface = pygame.Surface((current_track.width, current_track.height))
    temp_surface.blit(current_track.surface, (0, 0))

    # Load robot config and create instance
    with open("agent/config_line_follower.json", "r") as f:
        robot_config = json.load(f)

    line_follower = LineFollowerNEAT(
        genome=genome,
        neat_config=config,
        robot_config=robot_config,
        screen=temp_surface,
        sensor_type="jsumo_xline_v2",
        draw_robot=True,
        img_path="agent/robot.png"
    )

    # Sensor frequency control variables (same as training)
    target_sensor_frequency = 30.0  # Example: ESP32 reads at 30 Hz
    sensor_read_interval = 1.0 / target_sensor_frequency
    time_since_last_sensor_read = 0.0

    # Main loop
    clock = pygame.time.Clock()
    running = True
    accumulator = 0.0
    fixed_dt = 1/60.0
    last_time = pygame.time.get_ticks() / 1000.0

    while running:
        current_time = pygame.time.get_ticks() / 1000.0
        frame_time = current_time - last_time
        last_time = current_time

        accumulator += frame_time

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            viewport.handle_events(event)

        while accumulator >= fixed_dt:
            dt = fixed_dt
            time_since_last_sensor_read += dt
            if time_since_last_sensor_read >= sensor_read_interval:
                line_follower.get_line_sensor_readings()
                time_since_last_sensor_read -= sensor_read_interval

            line_follower.step(dt)
            accumulator -= fixed_dt

        # Rendering
        temp_surface.blit(current_track.full_image, (0, 0))
        line_follower.draw()
        viewport.apply(temp_surface)
        pygame.display.flip()
        clock.tick(60)
        
    pygame.quit()

if __name__ == "__main__":
    # Initialize Pygame first
    pygame.init()

    # Load config and genome
    local_dir = os.path.dirname(__file__)
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        os.path.join(local_dir, 'config.txt')
    )

    # Find and load best genome
    import glob
    genome_files = glob.glob("neat_results/models/best_genome_*.pkl")
    latest_genome = max(genome_files, key=os.path.getctime)

    with open(latest_genome, "rb") as f:
        winner = pickle.load(f)

    print(f"Loaded genome from {latest_genome}")
    run_example(winner, config)