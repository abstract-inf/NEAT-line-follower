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
import json

import pygame
import neat
import visualize
from agent import LineFollowerBot

# Initialize Pygame
pygame.init()

# Screen setup
WIDTH, HEIGHT = 1500, 700
screen = pygame.display.set_mode([WIDTH, HEIGHT])
pygame.display.set_caption("NEAT Line Follower")

# Load and scale the line path image
line_path = pygame.image.load("imgs/line_paths/line_path5.png")
line_path = pygame.transform.scale(line_path, (WIDTH, HEIGHT))

# Simulation time before moving to next generation (in seconds, assuming 60fps)
GEN_MAX_TIME = 40

# def draw_stop_button(screen):
#     """Draws a stop button in pygame and returns its rectangle for click detection."""
#     font = pygame.font.Font(None, 30)
#     button_rect = pygame.Rect(10, 10, 120, 40)  # (x, y, width, height)
#     pygame.draw.rect(screen, (200, 0, 0), button_rect)  # Red button
#     text_surface = font.render("Stop", True, (255, 255, 255))
#     screen.blit(text_surface, (button_rect.x + 30, button_rect.y + 10))
#     return button_rect

def draw_robots(robots:LineFollowerBot):
    for robot in robots:
        robot.draw()
    
def calculate_fitness(robots:LineFollowerBot, genes):
    for i, robot in enumerate(robots):
        try:
            robot.step(dt)
            
            linear_velocity, angular_velocity_rad = robot.get_velocity()
            angular_velocity_deg = angular_velocity_rad * 180/math.pi
            
            left_wheel_velocity, right_wheel_velocity = robot.left_wheel_velocity, robot.right_wheel_velocity
            
            center_strength = sum(robot.sensor_readings[len(robot.sensor_readings)//2-1 : len(robot.sensor_readings)//2+2])
            line_presence = any(robot.sensor_readings)
            
            # Base Speed Reward (encourage movement)
            speed_reward = linear_velocity * 1.5
            
            # Center Alignment Bonus (strong center preference)
            center_bonus = center_strength ** 1.5 * 2
            
            # Smooth Operation Bonus
            smooth_bonus = max(0, 2 - abs(left_wheel_velocity - right_wheel_velocity)) * 0.5
            
            # Sensor color detection
            sensor_color = robot.get_color()

            # 1. Progressive off-track penalty
            if not line_presence:
                off_track_penalty = 50 + (robot.off_track_time * 10)  # Ramping penalty
                robot.off_track_time += 1
            else:
                off_track_penalty = 0
                robot.off_track_time = 0

            # 2. Speed penalty adjustment
            if linear_velocity < 3.0 and center_strength < 2:  # Only penalize slow speeds when not centered
                genes[i].fitness -= 1.5 * (3.0 - linear_velocity)

            # 3. Steering penalty refinement
            steering_penalty = min(abs(angular_velocity_deg) * 0.8, 30)  # Cap max penalty

            # 4. Completion reward scaling
            if sensor_color == "green":
                genes[i].fitness += 5000 + (linear_velocity * 100)  # Reward speed AND completion
                robots.pop(i)
                genes.pop(i)
                continue
            elif sensor_color == "yellow":
                genes[i].fitness += 200
                continue
            elif sensor_color == "red":
                genes[i].fitness -= 500
                robots.pop(i)
                genes.pop(i)
                continue

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
    # global robot_config

    genes = [] 
    robots = []
    
    for genome_id, genome in genomes:
        genome.fitness = 0  # Initialize fitness
        line_follower = LineFollowerBot(genome=genome,
                                        neat_config=config,
                                        robot_config=robot_config,
                                        screen=screen,
                                        sensor_count=15)
        genes.append(genome)
        robots.append(line_follower)

    running = True
    clock = pygame.time.Clock()
    ticks = 0

    # Run simulation loop for GEN_MAX_TIME seconds (assuming 60 ticks per second)
    while running and robots and ticks < 60 * GEN_MAX_TIME:
        ticks += 1
        clock.tick(0)
        
        # calculate dt for consisten physics update
        global dt
        dt = clock.get_time() / 1000.0  # multiply by 1000 to convert to ms (e.g 16 to 0.016)
        dt = min(dt, 1/30.0)  # capping for maintaing stability if the simulation becomes too intensive
        
        # Draw background and line path.
        # screen.fill((255, 255, 255))
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
        # draw after calculating the fitness, so no robot sensor overlap
        draw_robots(robots=robots)
        
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

    # load robot config
    with open("agent/config_line_follower.json", "r") as f:
        global robot_config
        robot_config = json.load(f)

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
    generations = 12
    winner = population.run(eval_genomes, generations)

    print("\nBest genome:\n{!s}".format(winner))
    save_model(winner)

    # Visualize training statistics and the best network.
    visualize.plot_stats(stats, ylog=False, view=True, filename="")
    visualize.plot_species(stats, view=True, filename="")
    visualize.draw_net(config, winner, view=True, filename="Net")


if __name__ == "__main__":
    main()
