# train_neat.py

"""
Line Follower AI Agent using NEAT and Pygame (Optimized)
-------------------------------------------------------

This script trains a line follower using the NEAT algorithm, simulating the agent with Pygame.
It supports saving checkpoints in a folder and saving the best genome with a timestamp.

Optimizations applied:
- Reduced redundant calls to find the highest fitness genome.
- Optimized list removal during fitness calculation.
- Ensured Pygame surfaces are converted for better performance.
"""

import os
import glob
import math
import pickle
import random
import datetime
import json
import time # For profiling if needed

import pygame
import neat
import visualize
from agent import LineFollowerNEAT # Assuming this imports correctly
from environment.track import VirtualTrack, Viewport # Assuming this imports correctly

# Initialize Pygame
# ------------------
DISPLAY_TRAINING_WINDOW = True # Set this to False to disable rendering and potentially speed up headless training

# Set environment variables FIRST if running headless
if not DISPLAY_TRAINING_WINDOW:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

# Initialize Pygame ONCE regardless of display mode
pygame.init()
pygame.font.init() # Explicitly initialize font module

# Screen setup
# -------------
if DISPLAY_TRAINING_WINDOW:
    pygame.display.set_caption("Line Follower NEAT")
    viewport = Viewport(1200, 800, DISPLAY_TRAINING_WINDOW)
else:
    # Create minimal viewport for headless mode
    viewport = Viewport(1, 1, DISPLAY_TRAINING_WINDOW)
current_track = None

# --- Global state for tracking fitness (Consider encapsulating later) ---
# It's generally better to manage state within classes or pass it explicitly,
# but we'll keep globals for now to stick closer to the original structure.
highest_ever_fitness = -float('inf') # Highest fitness achieved across all generations
best_genome_id_ever = -1

# Simulation time before moving to next generation (in seconds, assuming 60fps)
GEN_MAX_TIME_SECONDS = 60
GEN_MAX_TICKS = 60 * GEN_MAX_TIME_SECONDS # Pre-calculate max ticks

# --- Helper Functions ---

def find_current_best_robot_index(current_genes):
    """Find the index of the genome with the highest fitness IN THE CURRENT generation."""
    max_fitness = -float('inf')
    best_index = -1
    if not current_genes: # Handle empty list
        return -1, -float('inf')

    for i, genome in enumerate(current_genes):
        # Use getattr to safely access fitness, defaulting to -inf if not set
        current_fitness = getattr(genome, 'fitness', -float('inf'))
        if current_fitness > max_fitness:
            max_fitness = current_fitness
            best_index = i
    return best_index, max_fitness

def draw_robots_optimized(robots_to_draw, current_best_idx):
    """Draws robots, highlighting the current best."""
    # Ensure robots_to_draw is not empty and current_best_idx is valid
    if not robots_to_draw or current_best_idx < 0 or current_best_idx >= len(robots_to_draw):
        # Draw all with default opacity if no valid best index
        for robot in robots_to_draw:
             robot.draw(draw_robot=True, opacity=50) # Use a default opacity
        return

    # Draw all robots, highlighting the best one
    for i, robot in enumerate(robots_to_draw):
        opacity = 255 if i == current_best_idx else 50
        robot.draw(draw_robot=True, opacity=opacity)


def calculate_fitness_optimized(robots, genes, dt):
    """
    Calculates fitness for each robot and handles removals efficiently.

    Returns:
        tuple: (remaining_robots, remaining_genes) - New lists with finished/crashed robots removed.
    """
    global highest_ever_fitness, best_genome_id_ever # Access globals to update overall best

    indices_to_remove = []
    current_best_fitness_in_gen = -float('inf')
    current_best_genome_id_in_gen = -1

    for i in range(len(robots)):
        robot = robots[i]
        genome = genes[i]

        try:
            # --- Robot Simulation Step ---
            robot.step(dt) # Update robot physics/state

            # --- Get State Variables ---
            linear_velocity, angular_velocity_rad = robot.get_velocity()
            angular_velocity_deg = math.degrees(angular_velocity_rad) # Use math.degrees
            left_wheel_velocity = robot.left_wheel_velocity
            right_wheel_velocity = robot.right_wheel_velocity

            # --- Process Sensor Readings ---
            # Ensure sensor readings are up-to-date (assuming get_line_sensor_readings was called)
            sensor_readings = robot.sensor_readings
            if not sensor_readings: # Handle case where readings might be empty/None
                 # Apply a penalty or default behavior if sensors haven't read yet
                 genome.fitness -= 10 # Example penalty
                 continue # Skip fitness calculation for this step

            num_sensors = len(sensor_readings)
            center_idx = num_sensors // 2
            # Define center indices more robustly
            center_start = max(0, center_idx - 1)
            center_end = min(num_sensors, center_idx + 2)
            center_indices = range(center_start, center_end)

            center_strength = sum(sensor_readings[j] for j in center_indices)
            middle_sensor = sensor_readings[center_idx] if num_sensors > 0 else 0
            line_presence = any(sensor_readings) # Check if any sensor is active

            # --- Fitness Calculation Components ---
            abs_speed = abs(linear_velocity)

            # Modulated Speed Reward
            if center_strength >= 2: # Well-centered
                speed_reward = abs_speed * 10.0
            else:
                speed_reward = abs_speed * 0.5
            # Reduce reward if turning sharply
            if abs(angular_velocity_deg) > 20:
                speed_reward *= 0.5

            # Middle Sensor Bonus
            middle_bonus = 50.0 if middle_sensor == 1.0 else 0.0 # Use float for consistency

            # Center Alignment Bonus (only if moving forward reasonably)
            center_bonus = (center_strength ** 1.5) * 2.0 if linear_velocity > 3.0 else 0.0

            # Smooth Operation Bonus (similar wheel speeds)
            smooth_bonus = max(0.0, 2.0 - abs(left_wheel_velocity - right_wheel_velocity)) * 0.5

            # Off-Track Penalty (Progressive)
            if not line_presence:
                robot.off_track_time += 1 # Increment counter on the robot object
                off_track_penalty = 50.0 + (robot.off_track_time * 50.0)
            else:
                robot.off_track_time = 0 # Reset counter
                off_track_penalty = 0.0

            # Speed Penalty/Reward (Refined Logic)
            speed_penalty = 0.0
            if linear_velocity <= 0: # Moving backwards or stopped
                 speed_penalty = 100.0 * (abs_speed + 5.0) # Penalize backward movement heavily
            elif 0 < linear_velocity <= 3.0: # Too slow forward
                 speed_penalty = 100.0 * (3.0 - abs_speed)
            elif linear_velocity > 25.0: # Too fast forward
                 speed_penalty = 100.0 * (abs_speed - 25.0)
            # else: # Moderate speed is implicitly rewarded by speed_reward, no penalty needed

            # Steering Penalty (High angular velocity)
            steering_penalty = min(abs(angular_velocity_deg) * 0.8, 30.0) # Cap the penalty

            # --- Zone Detection & Early Termination ---
            middle_sensor_zone_color = robot.check_middle_sensor_color()
            zone_action = False
            if middle_sensor_zone_color == "green": # Finished lap / Reached goal
                genome.fitness += 5000.0 + (abs_speed * 100.0) # Significant reward
                indices_to_remove.append(i)
                zone_action = True
                print(f"Robot {genome.key} reached GREEN zone! Fitness: {genome.fitness:.2f}")
            elif middle_sensor_zone_color == "red": # Crashed / Out of bounds
                genome.fitness -= 500.0 # Penalty
                indices_to_remove.append(i)
                zone_action = True
                print(f"Robot {genome.key} reached RED zone! Fitness: {genome.fitness:.2f}")
            elif middle_sensor_zone_color == "yellow": # Checkpoint (optional)
                genome.fitness += 500.0 # Intermediate reward

            if zone_action:
                # Update overall best if this finished robot is better
                if genome.fitness > highest_ever_fitness:
                    highest_ever_fitness = genome.fitness
                    best_genome_id_ever = genome.key
                continue # Skip normal fitness update for this step

            # --- Total Fitness Update (for robots still running) ---
            fitness_delta = (
                speed_reward
                + middle_bonus
                + center_bonus
                + smooth_bonus
                - off_track_penalty
                - speed_penalty
                - steering_penalty
            )
            genome.fitness += fitness_delta

            # Track the best fitness within this generation
            if genome.fitness > current_best_fitness_in_gen:
                 current_best_fitness_in_gen = genome.fitness
                 current_best_genome_id_in_gen = genome.key

            # Also update the overall best tracker
            if genome.fitness > highest_ever_fitness:
                highest_ever_fitness = genome.fitness
                best_genome_id_ever = genome.key

        except IndexError as e:
            # Handle potential errors during simulation/sensor access
            print(f"Error processing robot {i} (Genome {genome.key}): {e}. Penalizing and removing.")
            genome.fitness -= 200.0 # Penalize for error
            if i not in indices_to_remove: # Avoid adding duplicates
                 indices_to_remove.append(i)
            continue # Move to the next robot
        except Exception as e: # Catch other potential errors
            print(f"Unexpected error processing robot {i} (Genome {genome.key}): {e}. Penalizing and removing.")
            genome.fitness -= 500.0 # Higher penalty for unexpected errors
            if i not in indices_to_remove:
                 indices_to_remove.append(i)
            continue

    # --- Efficiently Remove Robots/Genes ---
    if not indices_to_remove:
        return robots, genes # No removals needed

    # Sort indices in descending order to avoid index shifting issues during potential in-place removal
    # Although creating new lists is generally safer and often fast enough.
    indices_to_remove.sort(reverse=True)

    # Create new lists excluding the removed indices (safer than pop)
    remaining_robots = [robot for i, robot in enumerate(robots) if i not in indices_to_remove]
    remaining_genes = [gene for i, gene in enumerate(genes) if i not in indices_to_remove]

    # --- Alternative: In-place removal (use if memory is a major concern) ---
    # for index in indices_to_remove:
    #     robots.pop(index)
    #     genes.pop(index)
    # return robots, genes # Return the modified original lists

    # print(f"Gen Best Fitness: {current_best_fitness_in_gen:.2f} (ID: {current_best_genome_id_in_gen}) | Overall Best: {highest_ever_fitness:.2f} (ID: {best_genome_id_ever})")

    return remaining_robots, remaining_genes


def eval_genomes(genomes, config):
    """
    Evaluates each genome by simulating the robot's movement.
    Adjusts fitness based on sensor activity and differential drive behavior.
    Uses optimized fitness calculation and drawing.
    """
    global current_track, viewport, robot_config, main_screen # Added main_screen
    global highest_ever_fitness, best_genome_id_ever # Track overall best

    # --- Setup Generation ---
    # Load new random track (or keep the same one - adjust as needed)
    # track_path = random.choice(glob.glob("environment/tracks/train/*.png")) # Example random track
    track_path = "environment/tracks/train/track 1.png" # Keep fixed for now
    print(f"Using track: {track_path}")
    current_track = VirtualTrack(track_path) # Assuming VirtualTrack loads and converts the surface
    if DISPLAY_TRAINING_WINDOW:
        viewport.update_world_size(current_track.width, current_track.height)

    # Create temporary surface for accurate sensor readings (ensure it's converted)
    # This surface is drawn onto by robots for sensor checks, needs track background
    temp_surface_for_sensors = pygame.Surface((current_track.width, current_track.height)).convert()

    # --- Initialize Robots for the Generation ---
    current_genes = []
    current_robots = []
    for genome_id, genome in genomes:
        genome.fitness = 0 # Initialize fitness for the new generation
        # Ensure robot image is loaded and converted within LineFollowerNEAT
        line_follower = LineFollowerNEAT(
            genome=genome,
            neat_config=config,
            robot_config=robot_config,
            screen=temp_surface_for_sensors, # Pass the sensor surface
            sensor_type="jsumo_xline_v2",
            draw_robot=True, # Control drawing via draw_robots_optimized
            img_path="agent/robot.png", # Ensure this path is correct
        )
        # Initialize robot's off-track time counter
        line_follower.off_track_time = 0
        current_genes.append(genome)
        current_robots.append(line_follower)

    # --- Simulation Loop ---
    running = True
    clock = pygame.time.Clock()
    ticks = 0
    accumulator = 0.0
    fixed_dt = 1/60.0 # Physics timestep (60Hz)
    last_time = time.perf_counter() # Use perf_counter for higher precision

    # Sensor reading timing
    target_sensor_frequency = 30.0 # Hz
    sensor_read_interval = 1.0 / target_sensor_frequency
    time_since_last_sensor_read = 0.0

    # --- Main Loop for the Generation ---
    while running and current_robots and ticks < GEN_MAX_TICKS:
        current_time = time.perf_counter()
        frame_time = current_time - last_time
        last_time = current_time

        # Prevent spiral of death on slow systems or pauses
        frame_time = min(frame_time, 0.25)

        accumulator += frame_time
        time_since_last_sensor_read += frame_time # Accumulate time for sensor reads

        # --- Event Handling (Once per frame) ---
        if DISPLAY_TRAINING_WINDOW or True: # Always pump events, even headless, for QUIT
             for event in pygame.event.get():
                 if event.type == pygame.QUIT:
                     print("QUIT event received. Stopping generation.")
                     running = False
                     # Optionally, force end the entire training here if needed
                     # raise KeyboardInterrupt("Pygame Quit Event")
                 elif DISPLAY_TRAINING_WINDOW and event: # Pass event only if displaying
                     viewport.handle_events(event)
        if not running: break # Exit loop if QUIT detected

        # --- Sensor Reading Logic ---
        read_sensors_this_step = False
        if time_since_last_sensor_read >= sensor_read_interval:
            read_sensors_this_step = True
            time_since_last_sensor_read -= sensor_read_interval # Reset timer partially

        # --- Fixed Timestep Physics Update Loop ---
        while accumulator >= fixed_dt:
            if not current_robots: break # Exit if all robots finished

            # Reset sensor surface *before* sensor reads and physics steps for this dt
            temp_surface_for_sensors.blit(current_track.full_image, (0, 0)) # Use the track's converted image

            # Read sensors for all robots if interval passed
            if read_sensors_this_step:
                for robot in current_robots:
                    # Draw robot onto the sensor surface *just* for reading
                    # Assuming robot.draw updates internal state needed for get_line_sensor_readings
                    # If not, ensure robot position/rotation is correct before reading
                    robot.draw(draw_robot=True, opacity=255) # Draw fully opaque for sensor reading
                    robot.get_line_sensor_readings() # Reads and saves readings

            # Calculate fitness and update robot physics for this fixed step
            # Pass the fixed timestep dt
            current_robots, current_genes = calculate_fitness_optimized(current_robots, current_genes, fixed_dt)

            accumulator -= fixed_dt
            ticks += 1
            read_sensors_this_step = False # Sensors read only once per physics interval group

            if ticks >= GEN_MAX_TICKS: # Check time limit
                 print("Generation time limit reached.")
                 running = False
                 break # Exit physics loop

        if not running or not current_robots: break # Exit main loop if stopped or no robots left

        # --- Rendering (if enabled) ---
        if DISPLAY_TRAINING_WINDOW and main_screen:
            # Smooth viewport controls (handle panning/zooming)
            viewport.handle_viewport_controls()

            # Prepare rendering surface (can be the main screen or a temporary surface)
            # Using main_screen directly might be slightly faster if no complex viewport scaling needed
            # If viewport does scaling/offset, rendering to a temp surface then applying is correct.
            render_surface = temp_surface_for_sensors # Reuse sensor surface or create a new one if needed
            render_surface.blit(current_track.full_image, (0, 0)) # Fresh background for drawing

            # Find best robot index for drawing *once* per frame
            current_best_idx, current_max_fitness = find_current_best_robot_index(current_genes)

            # Draw robots onto the render surface
            draw_robots_optimized(current_robots, current_best_idx)

            # --- Draw UI Text ---
            screen_height = main_screen.get_height()
            padding = 10 # Reduced padding
            try:
                font = pygame.font.SysFont("Arial", 18) # Slightly smaller font
                fps_font = pygame.font.SysFont("Arial", 18, bold=True)
            except Exception as e:
                print(f"Font loading error: {e}. Using default.")
                font = pygame.font.Font(None, 24) # Pygame default font
                fps_font = pygame.font.Font(None, 24)


            # Prepare text lines
            lines = []
            lines.append((f"Generation: {population.generation + 1}", (255, 255, 255))) # White
            lines.append((f"Alive: {len(current_robots)}", (255, 255, 255)))
            if current_best_idx != -1:
                 lines.append((f"Gen Best Fitness: {current_max_fitness:.2f}", (0, 255, 0))) # Green
                 lines.append((f"Gen Best ID: {current_genes[current_best_idx].key}", (0, 255, 0)))
            lines.append((f"Overall Best Fitness: {highest_ever_fitness:.2f}", (0, 255, 255))) # Cyan
            lines.append((f"Overall Best ID: {best_genome_id_ever}", (0, 255, 255)))

            # Render text lines from bottom up
            y_pos = screen_height - padding
            for text, color in reversed(lines):
                text_surface = font.render(text, True, color)
                text_rect = text_surface.get_rect(bottomleft=(padding, y_pos))
                render_surface.blit(text_surface, text_rect)
                y_pos -= text_rect.height + 2 # Move up for next line

            # Render FPS
            fps = int(clock.get_fps())
            fps_text = f"FPS: {fps}"
            fps_surface = fps_font.render(fps_text, True, (255, 0, 0)) # Red
            render_surface.blit(fps_surface, (padding, padding)) # Top-left

            # Apply the rendered surface to the screen via viewport
            viewport.apply(render_surface) # Viewport handles scaling/offset and blits to main_screen

            pygame.display.flip() # Update the actual display

            # Cap rendering framerate (doesn't affect physics speed)
            clock.tick(60) # Aim for 60 FPS rendering
        else:
            # No rendering, potentially run physics faster?
            # clock.tick() # Uncomment to uncap FPS in headless? Might not be needed.
            pass # Physics loop handles its own timing

    # --- End of Generation ---
    print(f"Generation finished. Ticks: {ticks}. Remaining robots: {len(current_robots)}")
    # Any remaining robots might need their final fitness assigned or penalized
    # for not finishing, depending on the desired NEAT behavior.
    # For now, their fitness is as calculated up to the timeout.


def save_model(winner, stats):
    """Saves the best genome and stats to files with timestamps."""
    results_dir = "neat_results"
    models_dir = os.path.join(results_dir, "models")
    stats_dir = os.path.join(results_dir, "stats")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(stats_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save Genome
    genome_filename = os.path.join(models_dir, f"best_genome_{timestamp}.pkl")
    with open(genome_filename, "wb") as f:
        pickle.dump(winner, f)
    print(f"Best genome saved as: {genome_filename}")

    # Save Stats (optional but good practice)
    stats_filename = os.path.join(stats_dir, f"stats_{timestamp}.pkl")
    with open(stats_filename, "wb") as f:
        pickle.dump(stats, f)
    print(f"Stats saved as: {stats_filename}")


def main():
    """Main function to handle NEAT training and checkpointing."""
    global robot_config, population # Make population global for eval_genomes access

    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')
    checkpoint_dir = os.path.join(local_dir, "neat_results/checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Load NEAT configuration.
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                 neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                 config_path)

    # Load robot config
    robot_config_path = os.path.join(local_dir, "agent/config_line_follower.json")
    try:
        with open(robot_config_path, "r") as f:
            robot_config = json.load(f)
        print("Robot config loaded.")
    except FileNotFoundError:
        print(f"Error: Robot config file not found at {robot_config_path}")
        return # Exit if config is missing
    except json.JSONDecodeError:
        print(f"Error: Robot config file {robot_config_path} is not valid JSON.")
        return

    continue_learning = True # Set to False to force start from scratch

    # Restore from the latest checkpoint if continuing training.
    if continue_learning:
        print("Attempting to load checkpoint...")
        try:
            # Find the latest checkpoint based on modification time or naming convention
            checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "neat-checkpoint-*"))
            if checkpoint_files:
                latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
                print(f"Restoring from checkpoint: {latest_checkpoint}")
                population = neat.Checkpointer.restore_checkpoint(latest_checkpoint)
                # Important: Update population config if it changed since checkpoint
                population.config = config
                print(f"Restored population from generation {population.generation}")
            else:
                print("No checkpoint found. Starting a new population.")
                population = neat.Population(config)
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Starting a new population.")
            population = neat.Population(config)
    else:
        print("Training a new model from scratch...")
        population = neat.Population(config)

    # Add NEAT reporters.
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    # Save a checkpoint every N generations.
    checkpoint_interval = 1 # Save every generation
    checkpoint_prefix = os.path.join(checkpoint_dir, "neat-checkpoint-")
    population.add_reporter(neat.Checkpointer(generation_interval=checkpoint_interval,
                                              filename_prefix=checkpoint_prefix))

    # Run the NEAT algorithm.
    MAX_GENERATIONS = 200 # Set maximum number of generations to run

    winner = None # Initialize winner

    try:
        # The run method calls eval_genomes for each generation
        winner = population.run(eval_genomes, MAX_GENERATIONS)

        # If training completes successfully (reaches MAX_GENERATIONS)
        print("\nTraining finished successfully!")
        print(f"\nBest genome found:\n{winner}")
        save_model(winner, stats) # Save the final winner and stats

        # Visualize results
        visualize.plot_stats(stats, ylog=False, view=True, filename="final_stats.svg")
        visualize.plot_species(stats, view=True, filename="final_species.svg")
        if winner:
             node_names = {-1:'S1', -2: 'S2', -3:'S3', -4: 'S4', -5:'S5', -6:'S6', -7:'S7', -8:'S8', -9:'S9', -10:'S10', -11:'S11', -12:'S12', 0:'L_Motor', 1:'R_Motor'} # Example node names
             visualize.draw_net(config, winner, True, filename="final_net.gv", node_names=node_names)
             visualize.draw_net(config, winner, True, filename="final_net_details.gv", show_disabled=True, prune_unused=False, node_names=node_names)


    except KeyboardInterrupt:
        print("\nTraining interrupted by user (Ctrl+C).")
        # Optionally save the best genome found so far
        best_genome_so_far = stats.best_genome()
        if best_genome_so_far:
             print("Saving the best genome found before interruption...")
             save_model(best_genome_so_far, stats)
             # Visualize
             visualize.plot_stats(stats, ylog=False, view=True, filename="interrupted_stats.svg")
             visualize.plot_species(stats, view=True, filename="interrupted_species.svg")
             node_names = {-1:'S1', -2: 'S2', -3:'S3', -4: 'S4', -5:'S5', -6:'S6', -7:'S7', -8:'S8', -9:'S9', -10:'S10', -11:'S11', -12:'S12', 0:'L_Motor', 1:'R_Motor'}
             visualize.draw_net(config, best_genome_so_far, True, filename="interrupted_net.gv", node_names=node_names)
        else:
             print("No best genome found before interruption.")

    except Exception as e:
        print(f"\nAn unexpected error occurred during training: {e}")
        import traceback
        traceback.print_exc() # Print detailed traceback
        # Attempt to save state even on error
        best_genome_so_far = stats.best_genome()
        if best_genome_so_far:
             print("Attempting to save the best genome found before error...")
             save_model(best_genome_so_far, stats)
        else:
             print("No best genome found before error.")


    finally:
        # Clean up Pygame
        pygame.quit()
        print("Pygame quit.")
        # Note: Visualizations might still be open if view=True


if __name__ == "__main__":
    main()
