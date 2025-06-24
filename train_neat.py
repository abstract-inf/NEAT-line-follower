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

# best genome fitness, id, and highest fitness used for debugging
max_fitness = -float('inf')  # for the current alive genomes
max_fitness_key = -1
highest_fitness = -1  # for the best genome in the population
highest_fitness_key = -1
max_fitness_index = -1  # Global variable to track the index of the genome with the highest fitness

# Global variable to track the current generation
current_generation = 0 # Initialize current_generation

# Simulation time before moving to next generation (in seconds, assuming 60fps)
GEN_MAX_TIME = 60 * 0.75

def find_highest_fitness_index():
    global max_fitness_index, genes, max_fitness, highest_fitness, highest_fitness_key, max_fitness_key
    """Find the index of the genome with the highest fitness."""
    max_fitness = -float('inf')
    max_fitness_index = -1
    for i, genome in enumerate(genes):
        if genome.fitness > max_fitness:
            max_fitness = genome.fitness
            max_fitness_index = i
    
    # Update global variables
    max_fitness_key = genes[max_fitness_index].key

    if max_fitness > highest_fitness:
        highest_fitness = max_fitness
        highest_fitness_key = genes[max_fitness_index].key
    else:
        max_fitness_key = highest_fitness_key

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
                speed_reward = abs_speed * 10.0
            else:
                speed_reward = abs_speed * 0.5

            # If turning sharply, cut the speed reward to discourage excessive speed in curves.
            if abs(angular_velocity_deg) > 20:
                speed_reward *= 0.5

            # Bonus for the middle sensor being fully active.
            middle_bonus = 50 if middle_sensor == 1 else 0

            # Center alignment bonus remains.
            center_bonus = (center_strength ** 1.5) * 2 if linear_velocity > 3 else 0

            # Smooth operation bonus rewards similar wheel speeds.
            smooth_bonus = max(0, 2 - abs(left_wheel_velocity - right_wheel_velocity)) * 0.5

            # Progressive off-track penalty if no sensor sees the line.
            if not line_presence:
                off_track_penalty = 50 + (robot.off_track_time * 50)
                robot.off_track_time += 1
            else:
                off_track_penalty = 0
                robot.off_track_time = 0

            # Penalize very slow movement (absolute speed below threshold).
            if linear_velocity <= 0:
                speed_penalty = 100 * (abs_speed + 5.0)
            elif 0 < linear_velocity <= 3.0:
                speed_penalty = 100 * (3.0 - abs_speed)
            elif linear_velocity >= 25.0:
                speed_penalty = 100 * (abs_speed - 25.0)
            else:
                speed_penalty = 50 * (abs_speed - 25.0)   # Reward for moderate speed.

            

            # Steering penalty: high angular velocities indicate jitter or unstable turning.
            steering_penalty = min(abs(angular_velocity_deg) * 0.8, 30)

            # Completion reward scaling (middle sensor zone detection).
            middle_sensor_zone_color = robot.check_middle_sensor_color()
            zone_action = False
            if middle_sensor_zone_color == "green":
                genes[i].fitness += 5000 + (abs_speed * 100)
                robots.pop(i)
                genes.pop(i)
                zone_action = True
            elif middle_sensor_zone_color == "red":
                genes[i].fitness -= 500
                robots.pop(i)
                genes.pop(i)
                zone_action = True
            elif middle_sensor_zone_color == "yellow":
                genes[i].fitness += 500

            if zone_action:
                continue

            # Total fitness update:
            genes[i].fitness += (
                speed_reward +
                middle_bonus +
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
    global current_track, viewport, dt, genes, robots, current_generation # Add current_generation to globals

    # Increment the generation counter
    current_generation += 1

    # Load new random track
    track_path = "environment/tracks/track for the paper.png"
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

    # Calculate max_speed for the current generation
    # Start at 150 (initial value) and increment by 1, capping at 500
    initial_max_speed = 100
    max_allowed_speed = 500
    # The max_speed for the current generation will be:
    # initial_max_speed + (current_generation - 1)
    # We subtract 1 from current_generation because the first generation (current_generation = 1)
    # should have the initial_max_speed.
    # Use max() to ensure it doesn't go below initial_max_speed if for some reason current_generation is less than 1.
    # Use min() to cap it at max_allowed_speed.
    current_max_speed = min(max_allowed_speed, initial_max_speed + (current_generation - 1))
    
    print(f"Generation: {current_generation}, Max Speed set to: {current_max_speed}")

    for genome_id, genome in genomes:
        genome.fitness = 0  # Initialize fitness
        line_follower = LineFollowerNEAT(
            genome=genome,
            neat_config=config,
            robot_config=robot_config,
            screen=temp_surface,
            sensor_type="semi_circle",
            draw_robot=True,
            img_path="agent/robot.png",
        )
        line_follower.max_speed = current_max_speed  # Set maximum speed for the robot based on generation
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
                    robot.get_line_sensor_readings()  # Reads and saves readings in the robot object
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


            # Render fitness values and keys at the bottom left of the screen.
            # max_fitness = find_highest_fitness_index()[1]
            max_fitness_text = f"Max Fitness: {max_fitness:.2f}"
            max_fitness_key_text = f"Max Fitness Key: {max_fitness_key}"
            highest_fitness_text = f"Highest Fitness: {highest_fitness:.2f}"
            highest_fitness_key_text = f"Highest Fitness Key: {highest_fitness_key}"
            max_fitness_index_text = f"Max Fitness Index: {max_fitness_index}"
            current_generation_text = f"Generation: {current_generation}" # Display current generation
            current_max_speed_text = f"Current Max Speed: {current_max_speed}" # Display current max speed

            # Get the height of the surface to position text at the bottom.
            screen_height = temp_surface.get_height()
            padding = 2

            # Set up fonts.
            font = pygame.font.SysFont("Arial", 20)
            fps_font = pygame.font.SysFont("Arial", 20, bold=True)

            # Prepare text lines as tuples (text, color).
            lines = [
                (highest_fitness_text, (0, 255, 0)),      # highest fitness - green
                (highest_fitness_key_text, (0, 255, 0)),    # highest fitness key - green
                (max_fitness_text, (0, 0, 0)),          # max fitness - black
                (max_fitness_key_text, (0, 0, 0)),        # max fitness key - black
                (max_fitness_index_text, (0, 0, 0)),         # max fitness index - black
                (current_generation_text, (0, 0, 255)),     # Current generation - blue
                (current_max_speed_text, (255, 165, 0)),   # Current max speed - orange
            ]

            # Render each line from bottom upwards.
            # Adjust the starting y position to make space for the new lines
            initial_y_offset = 135 + (2 * font.get_linesize()) # Add space for 2 new lines
            for i, (line, color) in enumerate(reversed(lines)):
                text_surface = font.render(line, True, color)
                text_height = text_surface.get_height()
                # Position text with padding from the left and stacked upwards from the bottom.
                y = initial_y_offset - (i + 1) * text_height
                temp_surface.blit(text_surface, (padding, y))
                
            # Render the FPS indicator at the top-left corner in bold red.
            fps = int(clock.get_fps())
            fps_text = f"FPS: {fps}"
            fps_surface = fps_font.render(fps_text, True, (255, 0, 0))
            temp_surface.blit(fps_surface, (padding, 0))
                
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
            latest_checkpoint = "neat_results/checkpoints/neat-checkpoint-153"
            population = neat.Checkpointer.restore_checkpoint(latest_checkpoint)
            population.config = config
            print("Restored from checkpoint:", latest_checkpoint)
            # When restoring from a checkpoint, we need to try and infer the current_generation
            # The checkpoint filename format is "neat-checkpoint-X" where X is the generation number.
            try:
                global current_generation
                current_generation = int(latest_checkpoint.split('-')[-1])
                print(f"Inferred starting generation: {current_generation}")
            except ValueError:
                print("Could not infer generation from checkpoint filename. Starting generation count from 0.")
                current_generation = 0 # Default if filename format is unexpected
        else:
            print("No checkpoint found. Starting a new population.")
            population = neat.Population(config)
            current_generation = 0 # Reset generation count for a new population
    else:
        print("Training a new model...")
        population = neat.Population(config)
        current_generation = 0 # Reset generation count for a new population

    # Add NEAT reporters.
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    # Save a checkpoint every 1 generations into the checkpoint folder.
    checkpoint_prefix = os.path.join(checkpoint_dir, "neat-checkpoint-")
    population.add_reporter(neat.Checkpointer(generation_interval=1, filename_prefix=checkpoint_prefix))

    # Run the NEAT algorithm.
    GENERATIONS = 1

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