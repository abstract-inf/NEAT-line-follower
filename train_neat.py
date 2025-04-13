# train_neat.py

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
DISPLAY_TRAINING_WINDOW = True  # Set this to False to disable rendering

# Set environment variables FIRST
if not DISPLAY_TRAINING_WINDOW:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

# Initialize Pygame ONCE regardless of display mode
pygame.init()

# Screen setup
if DISPLAY_TRAINING_WINDOW:
    pygame.display.set_caption("Line Follower NEAT")
    viewport = Viewport(1200, 800, DISPLAY_TRAINING_WINDOW)
else:
    # Create minimal viewport for headless mode
    viewport = Viewport(1, 1, DISPLAY_TRAINING_WINDOW)

current_track = None

# Simulation time before moving to next generation (in seconds, assuming 60fps)
GEN_MAX_TIME = 40

max_fitness_index = -1  # Global variable to track the index of the genome with the highest fitness
def find_highest_fitness_index():
    global max_fitness_index, genes
    """Find the index of the genome with the highest fitness."""
    max_fitness = -float('inf')
    max_fitness_index = -1
    for i, genome in enumerate(genes):
        if genome.fitness > max_fitness:
            max_fitness = genome.fitness
            max_fitness_index = i
    return max_fitness_index, max_fitness

def draw_robots(robots:LineFollowerNEAT):
    for i, robot in enumerate(robots):
        if i == find_highest_fitness_index()[0]:
            robot.draw(draw_robot=True, opacity=255)
        else:
            robot.draw(draw_robot=True, opacity=50)
    
def calculate_fitness(robots: LineFollowerNEAT, genes):
    for i, robot in enumerate(robots):
        try:
            robot.step(dt)
            
            linear_velocity, angular_velocity_rad = robot.get_velocity()
            angular_velocity_deg = angular_velocity_rad * 180 / math.pi
            left_wheel_velocity, right_wheel_velocity = robot.left_wheel_velocity, robot.right_wheel_velocity

            # Process sensor readings: use the center 3 sensors and middle sensor separately.
            center_indices = range(len(robot.sensor_readings)//2 - 1, len(robot.sensor_readings)//2 + 2)
            center_strength = sum(robot.sensor_readings[j] for j in center_indices)
            middle_sensor = robot.sensor_readings[len(robot.sensor_readings)//2]
            line_presence = any(robot.sensor_readings)
            
            # Use absolute speed so that fast reverse is not automatically penalized.
            abs_speed = abs(linear_velocity)
            # if i == find_highest_fitness_index()[0]:
            #     print(f"Robot {i} - Linear Velocity: {linear_velocity}, Angular Velocity: {angular_velocity_deg}, Left Wheel: {left_wheel_velocity}, Right Wheel: {right_wheel_velocity}, Center Strength: {center_strength}, Middle Sensor: {middle_sensor}")
            # Modulated Speed Reward:
            #   - If well-centered (center_strength high), reward speed more.
            #   - If the robot is turning sharply (angular_velocity high), reduce the reward.
            if center_strength >= 2:
                speed_reward = abs_speed * 2.0
            else:
                speed_reward = abs_speed * 0.5

            # If turning sharply, cut the speed reward to discourage excessive speed in curves.
            if abs(angular_velocity_deg) > 20:
                speed_reward *= 0.5

            # # Bonus for the middle sensor being fully active.
            # middle_bonus = 50 if middle_sensor == 1 else 0

            # Center alignment bonus remains.
            center_bonus = (center_strength ** 1.5) * 2 if linear_velocity > 10 and linear_velocity < 35 else 0

            # Smooth operation bonus rewards similar wheel speeds.
            smooth_bonus = max(0, 2 - abs(left_wheel_velocity - right_wheel_velocity)) * 0.5

            # Progressive off-track penalty if no sensor sees the line.
            if not line_presence:
                off_track_penalty = 50 + (robot.off_track_time * 50)
                robot.off_track_time += 1
            else:
                off_track_penalty = 0
                robot.off_track_time = 0

            # Reward optimal speed (10 to 37), penalize too slow or too fast.
            if linear_velocity <= 0:
                speed_penalty = 200  # Big penalty for no movement or reverse
            elif abs_speed < 6:
                speed_penalty = 20 * (6 - abs_speed)  # Penalize slow speed
            elif abs_speed > 27:
                speed_penalty = 20 * (abs_speed - 27)  # Penalize high speed
            else:
                speed_penalty = -150 * (abs_speed - 8)  # Reward moderate speed



            # Steering penalty: high angular velocities indicate jitter or unstable turning.
            steering_penalty = min(abs(angular_velocity_deg) * 0.8, 30)

            # Completion reward scaling (middle sensor zone detection).
            middle_sensor_zone_color = robot.check_middle_sensor_color()
            zone_action = False
            if middle_sensor_zone_color == "green":
                genes[i].fitness += 5000 * linear_velocity
                robots.pop(i)
                genes.pop(i)
                zone_action = True
            elif middle_sensor_zone_color == "red":
                genes[i].fitness -= 500
                robots.pop(i)
                genes.pop(i)
                zone_action = True
            elif middle_sensor_zone_color == "yellow":
                genes[i].fitness += 500 * linear_velocity

            if zone_action:
                continue

            # Total fitness update:
            genes[i].fitness += (
                speed_reward +
                # middle_bonus +
                center_bonus +
                smooth_bonus -
                off_track_penalty -
                speed_penalty -
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
    global current_track, viewport, dt, genes, robots

    # Load new random track
    track_path = "environment/tracks/train/track 1.png"
    current_track = VirtualTrack(track_path)
    viewport.update_world_size(current_track.width, current_track.height)

    # Create temporary surface for accurate sensor readings
    temp_surface = pygame.Surface((current_track.width, current_track.height))
    temp_surface.blit(current_track.surface, (0, 0))

    genes = []
    robots = []

    target_sensor_frequency = 30.0  # Example: ESP32 reads at 30 Hz
    sensor_read_interval = 1.0 / target_sensor_frequency
    time_since_last_sensor_read = 0.0

    for genome_id, genome in genomes:
        genome.fitness = 0  # Initialize fitness
        line_follower = LineFollowerNEAT(
            genome=genome,
            neat_config=config,
            robot_config=robot_config,
            screen=temp_surface,
            sensor_type="jsumo_xline_v2",
            draw_robot=True,
            img_path="agent/robot.png",
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

        if not DISPLAY_TRAINING_WINDOW:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif DISPLAY_TRAINING_WINDOW:
                    viewport.handle_events(event)

        # Reset surface for sensor accuracy (critical for both modes)
        temp_surface.blit(current_track.full_image, (0, 0))

        # Fixed timestep physics updates
        while accumulator >= fixed_dt:
            pygame.event.pump() # Keep Pygame's event system alive

            # Handle events ONCE per frame
            if DISPLAY_TRAINING_WINDOW:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif DISPLAY_TRAINING_WINDOW:
                        viewport.handle_events(event)

            # Update physics with consistent timestep
            dt = fixed_dt  # Use fixed timestep for physics

            # Update time since last sensor read
            time_since_last_sensor_read += dt

            # Check if it's time to read sensors for each robot
            for robot in robots:
                if time_since_last_sensor_read >= sensor_read_interval:
                    robot.get_line_sensor_readings()  # Assuming you've added this method to LineFollowerNEAT
                    time_since_last_sensor_read -= sensor_read_interval  # Reset the timer

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
            max_fitness = find_highest_fitness_index()[1]
            max_fitness_text = f"Max Fitness: {max_fitness:.2f}"
            font = pygame.font.SysFont("Arial", 120)
            text_surface = font.render(max_fitness_text, True, (0,0,0))
            temp_surface.blit(text_surface, (10, 10))
            viewport.apply(temp_surface)

            # viewport.apply(temp_surface)

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

    # Save a checkpoint every 1 generations into the checkpoint folder.
    checkpoint_prefix = os.path.join(checkpoint_dir, "neat-checkpoint-")
    population.add_reporter(neat.Checkpointer(generation_interval=1, filename_prefix=checkpoint_prefix))

    # Run the NEAT algorithm.
    GENERATIONS = 100

    winner = None  # Explicit initialization
    
    try:
        winner = population.run(eval_genomes, GENERATIONS)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nAn error occurred during training: {e}")
        raise e

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
