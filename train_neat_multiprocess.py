# train_neat_multiprocess.py
import os
import glob
import math
import pickle
import datetime
import json
import csv
import statistics
import argparse
import multiprocessing
from functools import partial

import pygame
import neat
import visualize
from agent import LineFollowerNEAT
from environment.track import VirtualTrack, Viewport


# Simulation time before moving to next generation (in seconds, assuming 60fps)
GEN_MAX_TIME = 60 * 0.75

# --- Global State (mostly for visual mode) ---
current_track = None
viewport = None
robot_config = None
genes = []
robots = []
max_fitness = -float('inf')
max_fitness_key = -1
highest_fitness = -1
highest_fitness_key = -1
max_fitness_index = -1
current_generation = 0
generation_fitness_data = []


# -----------------------------------------------------------------------------
# HEADLESS EVALUATION FUNCTION (FOR MULTIPROCESSING)
# -----------------------------------------------------------------------------
def eval_single_genome_headless(genome, config, robot_cfg, max_speed):
    # Ensure pygame display is initialized in each worker process
    if not pygame.display.get_init():
        pygame.display.set_mode((1, 1))
    """
    Evaluates a single genome in a headless environment. This function is designed
    to be called by a multiprocessing pool. It creates a robot, runs a simulation
    without rendering, calculates fitness, and returns the result.
    """
    genome.fitness = 0
    
    # Each process must load the track image itself (cannot share surfaces between processes)
    track_path = "environment/tracks/track for the paper.png"
    track_image = pygame.image.load(track_path).convert_alpha()
    line_follower = LineFollowerNEAT(
        genome=genome,
        neat_config=config,
        robot_config=robot_cfg,
        screen=track_image,
        sensor_type="semi_circle",
        draw_robot=False, # No drawing in headless mode
        img_path="agent/robot.png",
    )
    line_follower.max_speed = max_speed

    # --- Simulation Loop ---
    ticks = 0
    fixed_dt = 1/60.0
    sensor_read_interval = 1.0 / 30.0
    time_since_last_sensor_read = 0.0

    while ticks < 60 * GEN_MAX_TIME:
        # --- Physics and AI Update ---
        time_since_last_sensor_read += fixed_dt
        if time_since_last_sensor_read >= sensor_read_interval:
            line_follower.get_line_sensor_readings()
            time_since_last_sensor_read -= sensor_read_interval
        
        line_follower.step(fixed_dt)

        # --- Fitness Calculation ---
        linear_velocity, angular_velocity_rad = line_follower.get_velocity()
        angular_velocity_deg = angular_velocity_rad * 180 / math.pi
        left_wheel_velocity, right_wheel_velocity = line_follower.left_wheel_velocity, line_follower.right_wheel_velocity

        center_indices = range(len(line_follower.sensor_readings)//2 - 1, len(line_follower.sensor_readings)//2 + 2)
        center_strength = sum(line_follower.sensor_readings[j] for j in center_indices)
        middle_sensor = line_follower.sensor_readings[len(line_follower.sensor_readings)//2]
        line_presence = any(line_follower.sensor_readings)
        
        abs_speed = abs(linear_velocity)

        # Speed Reward
        if center_strength >= 2:
            speed_reward = abs_speed * 10.0
        else:
            speed_reward = abs_speed * 0.5
        if abs(angular_velocity_deg) > 20:
            speed_reward *= 0.5

        # Bonuses
        middle_bonus = 50 if middle_sensor == 1 else 0
        center_bonus = (center_strength ** 1.5) * 2 if linear_velocity > 3 else 0
        smooth_bonus = max(0, 2 - abs(left_wheel_velocity - right_wheel_velocity)) * 0.5

        # Penalties
        if not line_presence:
            off_track_penalty = 50 + (line_follower.off_track_time * 50)
            line_follower.off_track_time += 1
        else:
            off_track_penalty = 0
            line_follower.off_track_time = 0

        min_speed = (max_speed/1000)/2  # min speed is in m/s, e.g., 30 pixels/sec = 0.03 m/s

        if linear_velocity <= 0:
            speed_penalty = 100 * (abs_speed + 0.05)
        elif 0 < linear_velocity < min_speed:
            speed_penalty = 100 * (min_speed - abs_speed)
        elif linear_velocity > min_speed:
            speed_penalty = 100 * (abs_speed - min_speed)
        else:
            speed_penalty = 0  # No penalty in the optimal range
        
        steering_penalty = min(abs(angular_velocity_deg) * 0.8, 30)

        # Check for zone completion
        middle_sensor_zone_color = line_follower.check_middle_sensor_color()
        if middle_sensor_zone_color == "green":
            genome.fitness += 5000 + (abs_speed * 100)
            break # End simulation for this robot
        elif middle_sensor_zone_color == "red":
            genome.fitness -= 500
            break # End simulation for this robot
        elif middle_sensor_zone_color == "yellow":
            genome.fitness += 500

        # Update total fitness
        genome.fitness += (speed_reward + middle_bonus + center_bonus + smooth_bonus - 
                           off_track_penalty - speed_penalty - steering_penalty)
        
        ticks += 1

    return genome.fitness


# -----------------------------------------------------------------------------
# VISUAL MODE HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def find_highest_fitness_index():
    global max_fitness_index, genes, max_fitness, highest_fitness, highest_fitness_key, max_fitness_key
    max_fitness = -float('inf')
    max_fitness_index = -1
    for i, genome in enumerate(genes):
        if genome.fitness is not None and genome.fitness > max_fitness:
            max_fitness = genome.fitness
            max_fitness_index = i
    
    if genes and max_fitness_index != -1:
        max_fitness_key = genes[max_fitness_index].key
        if max_fitness > highest_fitness:
            highest_fitness = max_fitness
            highest_fitness_key = genes[max_fitness_index].key
        else:
            max_fitness_key = highest_fitness_key
    return max_fitness_index, max_fitness

def draw_robots(robots_to_draw):
    for i, robot in enumerate(robots_to_draw):
        if i == find_highest_fitness_index()[0]:
            robot.draw(draw_robot=True, opacity=255)
        else:
            robot.draw(draw_robot=True, opacity=50)

def calculate_fitness_visual(dt):
    # This function is coupled with the visual simulation loop
    global robots, genes
    robots_copy = list(robots)
    genes_copy = list(genes)

    for i, robot in enumerate(robots_copy):
        if robot not in robots:
            continue
        try:
            # The robot step is now done inside the main visual loop
            # Here we just calculate fitness based on its current state
            linear_velocity, angular_velocity_rad = robot.get_velocity()
            angular_velocity_deg = angular_velocity_rad * 180 / math.pi
            left_wheel_velocity, right_wheel_velocity = robot.left_wheel_velocity, robot.right_wheel_velocity

            center_indices = range(len(robot.sensor_readings)//2 - 1, len(robot.sensor_readings)//2 + 2)
            center_strength = sum(robot.sensor_readings[j] for j in center_indices)
            middle_sensor = robot.sensor_readings[len(robot.sensor_readings)//2]
            line_presence = any(robot.sensor_readings)
            
            abs_speed = abs(linear_velocity)

            if center_strength >= 2: speed_reward = abs_speed * 10.0
            else: speed_reward = abs_speed * 0.5
            if abs(angular_velocity_deg) > 20: speed_reward *= -0.5

            middle_bonus = 50 if middle_sensor == 1 else 0
            center_bonus = (center_strength ** 1.5) * 2 if linear_velocity > 3 else 0
            smooth_bonus = max(0, 2 - abs(left_wheel_velocity - right_wheel_velocity)) * 0.5

            if not line_presence:
                off_track_penalty = 50 + (robot.off_track_time * 50)
                robot.off_track_time += 1
            else:
                off_track_penalty = 0
                robot.off_track_time = 0

            min_speed = (MAX_SPEED/1000)/2  # min speed is in m/s, e.g., 30 pixels/sec = 0.03 m/s

            if linear_velocity <= 0:
                speed_penalty = 1000 * (abs_speed + 0.05)
            elif 0 < linear_velocity < min_speed:
                speed_penalty = -100 * (min_speed - abs_speed)
            elif linear_velocity >= min_speed:
                speed_penalty = -100 * (abs_speed - min_speed)
            else:
                speed_penalty = 0  # No penalty in the optimal range
            
            steering_penalty = min(abs(angular_velocity_deg) * 0.8, 30)

            middle_sensor_zone_color = robot.check_middle_sensor_color()
            zone_action = False
            if middle_sensor_zone_color == "green":
                genes[i].fitness += 5000 + (abs_speed * 100)
                robots.pop(robots.index(robot))
                genes.pop(genes.index(genes_copy[i]))
                zone_action = True
            elif middle_sensor_zone_color == "red":
                genes[i].fitness -= 500
                robots.pop(robots.index(robot))
                genes.pop(genes.index(genes_copy[i]))
                zone_action = True
            elif middle_sensor_zone_color == "yellow":
                if linear_velocity >= min_speed:
                    genes[i].fitness += 500
                else:
                    genes[i].fitness -= 1000
            
            if zone_action: continue

            genes[i].fitness += (speed_reward + middle_bonus + center_bonus + smooth_bonus - 
                               off_track_penalty - speed_penalty - steering_penalty)
        except (ValueError, IndexError):
            pass

# -----------------------------------------------------------------------------
# MAIN EVALUATION FUNCTION (CALLED BY NEAT)
# -----------------------------------------------------------------------------
def eval_genomes(genomes, config):
    """
    Evaluates genomes. Switches between parallel headless evaluation and
    single-threaded visual evaluation based on the DISPLAY_TRAINING_WINDOW flag.
    """
    global current_generation, generation_fitness_data, robot_config
    global current_track, viewport, genes, robots, MAX_SPEED

    current_generation += 1
    
    # --- Determine Max Speed for this Generation ---
    # initial_max_speed = 250
    # max_allowed_speed = 250
    # current_max_speed = min(max_allowed_speed, initial_max_speed + (current_generation - 1))
    
    print(f"Generation: {current_generation}, Max Speed set to: {MAX_SPEED}")

    # --- Load Track ---
    track_path = "environment/tracks/track for the paper.png"
    
    # =================================================
    # --- Headless Multiprocessing Mode ---
    # =================================================
    if not DISPLAY_TRAINING_WINDOW:
        # Use a multiprocessing pool to evaluate all genomes in parallel.
        with multiprocessing.Pool() as pool:
            # Prepare the worker function with arguments that don't change (no pygame.Surface passed!)
            worker_func = partial(eval_single_genome_headless, 
                                  config=config, 
                                  robot_cfg=robot_config, 
                                  max_speed=MAX_SPEED)
            # Map the worker function to all genomes
            genome_list = [g for _, g in genomes]
            fitnesses = pool.map(worker_func, genome_list)

            # Assign the calculated fitness back to each genome
            for i, (_, genome) in enumerate(genomes):
                genome.fitness = fitnesses[i]

    # =================================================
    # --- Visual Single-Core Mode ---
    # =================================================
    else:
        current_track = VirtualTrack(track_path)
        viewport.update_world_size(current_track.width, current_track.height)
        temp_surface = pygame.Surface((current_track.width, current_track.height))
        temp_surface.blit(current_track.surface, (0, 0))

        genes = []
        robots = []

        for _, genome in genomes:
            genome.fitness = 0
            line_follower = LineFollowerNEAT(
                genome=genome, neat_config=config, robot_config=robot_config,
                screen=temp_surface, sensor_type="semi_circle", draw_robot=True,
                img_path="agent/robot.png",
            )
            line_follower.max_speed = MAX_SPEED
            genes.append(genome)
            robots.append(line_follower)

        # --- Visual Simulation Loop ---
        running = True
        clock = pygame.time.Clock()
        ticks = 0
        accumulator = 0.0
        fixed_dt = 1/60.0
        last_time = pygame.time.get_ticks() / 1000.0
        sensor_read_interval = 1.0 / 30.0
        time_since_last_sensor_read = 0.0

        while running and robots and ticks < 60 * GEN_MAX_TIME:
            current_time = pygame.time.get_ticks() / 1000.0
            frame_time = min(current_time - last_time, 0.25)
            last_time = current_time
            accumulator += frame_time

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    pygame.quit()
                    exit()
                viewport.handle_events(event)

            temp_surface.blit(current_track.full_image, (0, 0))

            while accumulator >= fixed_dt:
                time_since_last_sensor_read += fixed_dt
                read_sensors_now = time_since_last_sensor_read >= sensor_read_interval
                if read_sensors_now:
                    time_since_last_sensor_read -= sensor_read_interval

                for robot in robots:
                    if read_sensors_now:
                        robot.get_line_sensor_readings()
                    robot.step(fixed_dt)
                
                calculate_fitness_visual(fixed_dt)
                accumulator -= fixed_dt
                ticks += 1

            viewport.handle_viewport_controls()
            draw_robots(robots)
            
            # --- Draw UI Text ---
            font = pygame.font.SysFont("Arial", 20)
            lines = [
                (f"Highest Fitness: {highest_fitness:.2f}", (0, 255, 0)),
                (f"Max Fitness: {max_fitness:.2f}", (0, 0, 0)),
                (f"Generation: {current_generation}", (0, 0, 255)),
                (f"Max Speed: {MAX_SPEED}", (255, 165, 0)),
                (f"Alive: {len(robots)}", (255, 0, 0)),
            ]
            for i, (line, color) in enumerate(lines):
                text_surface = font.render(line, True, color)
                temp_surface.blit(text_surface, (5, 5 + i * 22))
            
            fps_text = f"FPS: {int(clock.get_fps())}"
            fps_surface = font.render(fps_text, True, (255, 0, 0))
            temp_surface.blit(fps_surface, (5, 5 + len(lines) * 22))

            viewport.apply(temp_surface)
            pygame.display.flip()
            clock.tick(0)

    # --- Post-Generation Statistics (for both modes) ---
    all_fitnesses = [g.fitness for _, g in genomes if g.fitness is not None]
    if all_fitnesses:
        best_fitness = max(all_fitnesses)
        avg_fitness = statistics.mean(all_fitnesses)
        std_dev = statistics.stdev(all_fitnesses) if len(all_fitnesses) > 1 else 0.0
        generation_fitness_data.append({
            'generation': current_generation,
            'best_fitness': best_fitness,
            'average_fitness': avg_fitness,
            'plus_1_std': avg_fitness + std_dev,
            'minus_1_std': avg_fitness - std_dev
        })
        save_fitness_data_to_csv()

# -----------------------------------------------------------------------------
# FILE I/O AND MAIN FUNCTION
# -----------------------------------------------------------------------------
def save_model(winner):
    global MAX_SPEED
    os.makedirs(f"neat_results/{MAX_SPEED}/models", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = os.path.join(f"neat_results/{MAX_SPEED}/models", f"best_genome_{timestamp}.pkl")
    with open(filename, "wb") as f:
        pickle.dump(winner, f)
    print("Genome saved as", filename)

def save_fitness_data_to_csv():
    global MAX_SPEED, generation_fitness_data
    os.makedirs(f"neat_results/{MAX_SPEED}/statistics", exist_ok=True)
    filename = os.path.join(f"neat_results/{MAX_SPEED}/statistics", "generation_fitness_stats.csv")
    fieldnames = ['generation', 'best_fitness', 'average_fitness', 'plus_1_std', 'minus_1_std']
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(generation_fitness_data)
    print(f"Fitness statistics saved to {filename}")

def main(MAX_SPEED):
    global DISPLAY_TRAINING_WINDOW, robot_config, viewport, current_generation

    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Train a line-following robot using NEAT.")
    parser.add_argument('--headless', action='store_true', help="Run in headless mode without visualization for faster training.")
    args = parser.parse_args()

    # --- Global Configurations ---
    DISPLAY_TRAINING_WINDOW = args.headless  # just add 'not' before args.headless to invert the logic
    # MAX_SPEED = 100  # Set a fixed max speed for all generations (in pixels/sec)
    GENERATIONS = 300  # Number of generations to run NEAT

    print(f"Running in {'headless' if not DISPLAY_TRAINING_WINDOW else 'visual'} mode.")

    # --- Pygame Initialization ---
    if not DISPLAY_TRAINING_WINDOW:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    pygame.init()

    if DISPLAY_TRAINING_WINDOW:
        pygame.display.set_caption("Line Follower NEAT")
        viewport = Viewport(1200, 800)
    else:
        # FIX: Even in headless mode, a display must be set for image loading to work.
        pygame.display.set_mode((1, 1))
    
    # --- Load Configs ---
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')
    checkpoint_dir = os.path.join(local_dir, f"neat_results/{MAX_SPEED}/checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    with open("agent/config_line_follower.json", "r") as f:
        robot_config = json.load(f)

    # --- Population Setup ---
    continue_learning = True
    if continue_learning:
        print("Attempting to restore from checkpoint...")
        try:
            latest_checkpoint = max(glob.glob(os.path.join(checkpoint_dir, "neat-checkpoint-*")), key=os.path.getctime)
            population = neat.Checkpointer.restore_checkpoint(latest_checkpoint)
            current_generation = population.generation
            print(f"Restored from checkpoint: {latest_checkpoint} at generation {current_generation}")
        except (ValueError, FileNotFoundError):
            print("No checkpoint found. Starting a new population.")
            population = neat.Population(config)
    else:
        print("Starting a new population.")
        population = neat.Population(config)

    # --- NEAT Reporters ---
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    population.add_reporter(neat.Checkpointer(generation_interval=1, filename_prefix=os.path.join(checkpoint_dir, "neat-checkpoint-")))

    # --- Run NEAT ---
    winner = None
    try:
        winner = population.run(eval_genomes, GENERATIONS)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nAn error occurred during training: {e}")
        raise
    finally:
        if winner:
            print(f"\nBest genome found:\n{winner}")
            save_model(winner)
        else:
            print("\nNo winner genome produced.")
        
        save_fitness_data_to_csv()

        if stats.best_genome():
             visualize.plot_stats(stats, ylog=False, view=False, filename=f"neat_results/{MAX_SPEED}/statistics/stats.svg")
             visualize.plot_species(stats, view=False, filename=f"neat_results/{MAX_SPEED}/statistics/species.svg")
             visualize.draw_net(config, stats.best_genome(), view=False, filename=f"neat_results/{MAX_SPEED}/statistics/net.svg")
        
        pygame.quit()

if __name__ == "__main__":
    # This is required for multiprocessing to work correctly on some platforms
    multiprocessing.freeze_support() 
    for speed in [100, 200, 300, 400, 500]:
        MAX_SPEED = speed
        print(f"Starting training with MAX_SPEED = {MAX_SPEED}")
        # Reset for each speed
        locals()['generation_fitness_data'] = [] 
        globals()['current_generation'] = 0
        globals()['highest_fitness'] = -float('inf')
        globals()['highest_fitness_key'] = -1
        globals()['genes'] = []
        globals()['robots'] = []

        main(MAX_SPEED)
