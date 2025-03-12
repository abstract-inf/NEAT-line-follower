# this code is a test on how to build a line follower AI agent using NEAT algorithm in a simulation using pygame

import pygame
import math
import neat
import os
import pickle
import visualize
from robot import LineFollower

pygame.init()

# Set up the drawing window
WIDTH, HEIGHT = 1500, 700
screen = pygame.display.set_mode([WIDTH, HEIGHT])
pygame.display.set_caption("NEAT Line Follower")

# load line path image
line_path = pygame.image.load("line_paths/test.png")
line_path = pygame.transform.scale(line_path, (WIDTH, HEIGHT))


def run_example(genome, config):
    line_follower = LineFollower(genome, config, screen)

    running = True
    clock = pygame.time.Clock()
    while running:
        clock.tick(60)

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

        line_follower.move()
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
    genome_files = glob.glob("models/best_genome_*.pkl")
    latest_genome = max(genome_files, key=os.path.getctime)  # Sort by creation time

    # Load the latest genome
    with open(latest_genome, "rb") as f:
        winner = pickle.load(f)

    print(f"Loaded genome from {latest_genome}")

    run_example(winner, config)