# Updated code with display initialization fix
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
        img_path="agent/robot_center.png"
        )

    # Main loop
    clock = pygame.time.Clock()
    running = True
    
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            viewport.handle_events(event)

        # Physics update
        line_follower.step(1/60.0)  # Fixed 60Hz physics
        
        # Rendering
        temp_surface.blit(current_track.full_image, (0, 0))
        line_follower.draw()
        viewport.apply(temp_surface)
        pygame.display.flip()
        clock.tick(60)
        # time.sleep(1)

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