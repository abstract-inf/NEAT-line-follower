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
from agent import LineFollowerNEAT
from environment.track import VirtualTrack, Viewport

# Initialize Pygame
pygame.init()

DISPLAY_TRAINING_WINDOW = False  # Set this to False to disable rendering

# Screen setup
# Replace screen setup with:
viewport = Viewport(1200, 800, DISPLAY_TRAINING_WINDOW)
current_track = None

# Simulation time before moving to next generation (in seconds, assuming 60fps)
GEN_MAX_TIME = 40


def draw_robots(robots:LineFollowerNEAT):
    for robot in robots:
        robot.draw(draw_robot=True, opacity=200)
    
def calculate_fitness(robots:LineFollowerNEAT, genes):
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
    global current_track, viewport, dt
    
    # Load new random track 
    track_path = "environment/tracks/train/test.png" 
    current_track = VirtualTrack(track_path) 
    viewport.update_world_size(current_track.width, current_track.height)
    
    # Create temporary surface for accurate sensor readings 
    temp_surface = pygame.Surface((current_track.width, current_track.height)) 
    temp_surface.blit(current_track.surface, (0, 0)) 

    genes = [] 
    robots = []
    
    for genome_id, genome in genomes:
        genome.fitness = 0  # Initialize fitness
        line_follower = LineFollowerNEAT(
                        genome=genome,
                        neat_config=config,
                        robot_config=robot_config,
                        screen=temp_surface,
                        sensor_type="jsumo_xline_v2",
                        draw_robot=True,
                        img_path="agent/robot.png"
                        )
        genes.append(genome)
        robots.append(line_follower)

    running = True
    clock = pygame.time.Clock()
    ticks = 0
    accumulator = 0.0
    fixed_dt = 1/60.0  # Physics timestep (60Hz)
    last_time = pygame.time.get_ticks() / 1000.0

    # Run simulation loop for GEN_MAX_TIME seconds
    while running and robots and ticks < 60 * GEN_MAX_TIME:
        # Calculate delta time
        current_time = pygame.time.get_ticks() / 1000.0
        frame_time = current_time - last_time
        last_time = current_time
        
        # Prevent spiral of death on slow systems
        frame_time = min(frame_time, 0.25)
        
        # Accumulate time for fixed timestep
        accumulator += frame_time

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif DISPLAY_TRAINING_WINDOW:
                viewport.handle_events(event)

        # Fixed timestep physics updates
        while accumulator >= fixed_dt:
            # Update physics with consistent timestep
            dt = fixed_dt  # Use fixed timestep for physics
            
            # Reset surface for sensor accuracy (critical for both modes)
            temp_surface.blit(current_track.full_image, (0, 0))
            
            calculate_fitness(robots, genes)
            
            accumulator -= fixed_dt
            ticks += 1

        # Optional rendering (independent of physics updates)
        if DISPLAY_TRAINING_WINDOW:
            # Smooth viewport controls
            viewport.handle_viewport_controls()
            
            # Render at display framerate
            temp_surface.blit(current_track.full_image, (0, 0))  # Fresh background
            draw_robots(robots)
            viewport.apply(temp_surface)
            pygame.display.flip()
            
            # Cap rendering framerate
            clock.tick(60)
        else:
            # Run physics as fast as possible in headless mode
            pass

def save_model(winner):
    """Saves the best genome to a file with a timestamp."""
    os.makedirs("neat_results/models", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = os.path.join("neat_results/models", f"best_genome_{timestamp}.pkl")
    with open(filename, "wb") as f:
        pickle.dump(winner, f)
    print("Genome saved as", filename)


def main():
    """Main function to handle NEAT training and checkpointing."""
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')
    checkpoint_dir = os.path.join(local_dir, "neat_results/checkpoints")
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
    GENERATIONS = 500

    winner = None  # Explicit initialization
    
    try:
        winner = population.run(eval_genomes, GENERATIONS)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    finally:
        if winner:
            print(f"\nBest genome:\n{winner}")
            save_model(winner)
        else:
            print("\nNo winner genome available")

        # Visualize training statistics and the best network.
        visualize.plot_stats(stats, ylog=False, view=True, filename="stats")
        visualize.plot_species(stats, view=True, filename="species")
        visualize.draw_net(config, winner, view=True, filename="Net")


if __name__ == "__main__":
    main()
