"""
Line Follower AI Agent using NEAT and Pygame
------------------------------------------------
This script trains a line follower using the NEAT algorithm, simulating the agent with Pygame.
It supports saving checkpoints in a folder and saving the best genome with a timestamp.
"""

import os
import glob
import math
import pickle
import random
import datetime

import pygame
import neat
import visualize
from robot import LineFollower

# Initialize Pygame
pygame.init()

# Screen setup
WIDTH, HEIGHT = 1000, 1000
screen = pygame.display.set_mode([WIDTH, HEIGHT])
pygame.display.set_caption("NEAT Line Follower")

# Load and scale the line path image
line_path = pygame.image.load("line_paths/line_path.png")
line_path = pygame.transform.scale(line_path, (WIDTH, HEIGHT))

# Simulation time before moving to next generation (in seconds, assuming 60fps)
GEN_MAX_TIME = 20

def draw_stop_button(screen):
    """Draws a stop button in pygame and returns its rectangle for click detection."""
    font = pygame.font.Font(None, 30)
    button_rect = pygame.Rect(10, 10, 120, 40)  # (x, y, width, height)
    pygame.draw.rect(screen, (200, 0, 0), button_rect)  # Red button
    text_surface = font.render("Stop", True, (255, 255, 255))
    screen.blit(text_surface, (button_rect.x + 30, button_rect.y + 10))
    return button_rect


def calculate_fitness(robots, genes):
    """Balanced fitness function with speed incentives and progressive rewards"""
    for i, robot in enumerate(robots):
        try:
            robot.draw()
            angular_change, left_motor, right_motor = robot.move()
            
            avg_speed = (left_motor + right_motor) / 2
            center_strength = sum(robot.sensors_data[len(robot.sensors_data)//2-1 : len(robot.sensors_data)//2+2])
            line_presence = any(robot.sensors_data)
            
            # Base Speed Reward (encourage movement)
            speed_reward = avg_speed * 1.5
            
            # Center Alignment Bonus (strong center preference)
            center_bonus = center_strength ** 1.5 * 2
            
            # Smooth Operation Bonus
            smooth_bonus = max(0, 2 - abs(left_motor - right_motor)) * 0.5
            
            # Progressive Off-Track Penalty
            off_track_penalty = 0
            if not line_presence:
                off_track_penalty = 500  # Faster off-track = worse
                
            # Stagnation Penalty (prevents parked robots)
            if avg_speed < 3.0:
                genes[i].fitness -= 1.5 * (3.0 - avg_speed)
                continue
           
            # Angular Change Penalty (proportional)
            steering_penalty = abs(angular_change) * 0.1  # 0.1 per degree
            
            # Total Fitness
            genes[i].fitness += (
                speed_reward +
                center_bonus +
                smooth_bonus -
                off_track_penalty -
                steering_penalty
            )
            

        except IndexError:
            genes[i].fitness -= 200
            robots.pop(i)
            genes.pop(i)
            break


def eval_genomes(genomes, config):
    """
    Evaluates each genome by simulating the robot's movement.
    Adjusts fitness based on sensor activity and the differential drive behavior.
    """
    genes = [] 
    robots = []
    
    for genome_id, genome in genomes:
        genome.fitness = 0  # Initialize fitness
        line_follower = LineFollower(genome, config, screen)
        genes.append(genome)
        robots.append(line_follower)

    running = True
    clock = pygame.time.Clock()
    ticks = 0

    # Run simulation loop for GEN_MAX_TIME seconds (assuming 60 ticks per second)
    while running and robots and ticks < 60 * GEN_MAX_TIME:
        ticks += 1
        clock.tick(0)

        # Draw background and line path.
        screen.fill((255, 255, 255))
        screen.blit(line_path, (0, 0))
        
        # Draw stop button.
        # stop_button = draw_stop_button(screen)
       
        # Check for events.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # if stop_button.collidepoint(event.pos):
                    pass  # Stop training if button is clicked.
        if pygame.key.get_pressed()[pygame.K_ESCAPE]:
            running = False

        calculate_fitness(robots, genes)
        pygame.display.flip()


def save_model(winner):
    """Saves the best genome to a file with a timestamp."""
    os.makedirs("models", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = os.path.join("models", f"best_genome_{timestamp}.pkl")
    with open(filename, "wb") as f:
        pickle.dump(winner, f)
    print("Genome saved as", filename)


def main():
    """Main function to handle NEAT training and checkpointing."""
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')
    checkpoint_dir = os.path.join(local_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Load NEAT configuration.
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_path)

    continue_learning = True

    # Restore from the latest checkpoint if continuing training.
    if continue_learning:
        print("Loading checkpoint from last model...")
        checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "neat-checkpoint-*"))
        if checkpoint_files:
            latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
            population = neat.Checkpointer.restore_checkpoint(latest_checkpoint)
            population.config = config
            print("Restored from checkpoint:", latest_checkpoint)
        else:
            print("No checkpoint found. Starting a new population.")
            population = neat.Population(config)
    else:
        print("Training a new model...")
        population = neat.Population(config)

    # Add NEAT reporters.
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    # Save a checkpoint every 10 generations into the checkpoint folder.
    checkpoint_prefix = os.path.join(checkpoint_dir, "neat-checkpoint-")
    population.add_reporter(neat.Checkpointer(generation_interval=10, filename_prefix=checkpoint_prefix))

    # Run the NEAT algorithm.
    generations = 50 
    winner = population.run(eval_genomes, generations)

    print("\nBest genome:\n{!s}".format(winner))
    save_model(winner)

    # Visualize training statistics and the best network.
    visualize.plot_stats(stats, ylog=False, view=True, filename="")
    visualize.plot_species(stats, view=True, filename="")
    visualize.draw_net(config, winner, view=True, filename="Net")


if __name__ == "__main__":
    main()
