# this code is a test on how to build a line follower AI agent using NEAT algorithm in a simulation using pygame

import pygame
import math
import neat
import os
import pickle
import visualize
import json
from agent import LineFollowerBot

# Initialize Pygame
pygame.init()

# Screen setup
# Load the line path image
line_path = pygame.image.load("imgs/line_paths/test.png")
# set the screen size to the image size
screen = pygame.display.set_mode(line_path.get_size())
pygame.display.set_caption("NEAT Line Follower")


def run_example(genome, config):
    with open("agent/config_line_follower.json", "r") as f:
        robot_config = json.load(f)

    line_follower = LineFollowerBot(genome, config, robot_config, screen)

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
        screen.fill((255, 255, 255))
        screen.blit(line_path, (0, 0))  # Draw the background line path once

        line_follower.step(dt)
        line_follower.draw()


        # Update the display
        pygame.display.update()


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')
    
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    
    import glob

    # Find the latest genome file
    genome_files = glob.glob("neat_results/models/best_genome_*.pkl")
    latest_genome = max(genome_files, key=os.path.getctime)  # Sort by creation time

    # Load the latest genome
    with open(latest_genome, "rb") as f:
        winner = pickle.load(f)

    print(f"Loaded genome from {latest_genome}")

    run_example(winner, config)