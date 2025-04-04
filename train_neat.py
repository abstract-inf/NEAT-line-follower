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

import neat.checkpoint
import pygame
import neat
import visualize
from agent import LineFollowerNEAT
from environment.track import VirtualTrack

# Initialize Pygame
pygame.init()

DISPLAY_TRAINING_WINDOW = True  # Set this to False to disable rendering

class Viewport:
    def __init__(self, screen_width, screen_height):
        if DISPLAY_TRAINING_WINDOW:
            self.screen = pygame.display.set_mode((screen_width, screen_height))
        else:
            # Create minimal hidden window
            self.screen = pygame.display.set_mode((1, 1), pygame.NOFRAME)
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        self.zoom = 1.0
        self.offset = pygame.Vector2(0, 0)
        self.world_size = pygame.Vector2(0, 0)
        self.screen_size = pygame.Vector2(screen_width, screen_height)
        self.dragging = False
        self.last_mouse = pygame.Vector2(0, 0)
        self.min_zoom = 1.0

    def update_world_size(self, width, height):
        self.world_size = pygame.Vector2(width, height)
        
        # Calculate minimum zoom to either fit image or show at 1:1
        width_ratio = self.screen_size.x / width
        height_ratio = self.screen_size.y / height
        self.min_zoom = min(width_ratio, height_ratio) if (width > self.screen_size.x or height > self.screen_size.y) else 1.0
        self.zoom = self.min_zoom
        
        # Center the image initially
        self.center_image()

    def center_image(self):
        """Center the image in the viewport"""
        visible_width = self.screen_size.x / self.zoom
        visible_height = self.screen_size.y / self.zoom
        
        # Calculate maximum possible offset
        max_offset_x = max(0, self.world_size.x - visible_width)
        max_offset_y = max(0, self.world_size.y - visible_height)
        
        # Set offset to center
        self.offset.x = max_offset_x / 2
        self.offset.y = max_offset_y / 2

    def handle_events(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left click
                self.dragging = True
                self.last_mouse = pygame.Vector2(event.pos)
                
            elif event.button in (4, 5):  # Mouse wheel
                # Get mouse position before zoom
                mouse_pos = pygame.Vector2(event.pos)
                old_world_pos = (mouse_pos / self.zoom) + self.offset
                
                # Apply zoom
                if event.button == 4:  # Zoom in
                    new_zoom = min(4.0, self.zoom * 1.1)
                else:  # Zoom out
                    new_zoom = max(self.min_zoom, self.zoom / 1.1)
                
                # Calculate new offset to maintain mouse position
                new_world_pos = (mouse_pos / new_zoom) + self.offset
                delta = old_world_pos - new_world_pos
                self.offset += delta
                self.zoom = new_zoom
                
                # Clamp offset after zoom
                self.offset.x = max(0, min(self.offset.x, self.world_size.x - (self.screen_size.x/self.zoom)))
                self.offset.y = max(0, min(self.offset.y, self.world_size.y - (self.screen_size.y/self.zoom)))

        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                self.dragging = False

        elif event.type == pygame.MOUSEMOTION and self.dragging:
            mouse_pos = pygame.Vector2(event.pos)
            delta = (mouse_pos - self.last_mouse) / self.zoom
            self.offset -= delta
            self.last_mouse = mouse_pos
            
            # Clamp to valid boundaries
            self.offset.x = max(0, min(self.offset.x, self.world_size.x - (self.screen_size.x / self.zoom)))
            self.offset.y = max(0, min(self.offset.y, self.world_size.y - (self.screen_size.y / self.zoom)))

    def apply(self, surface):
        """Render the track with proper scaling and positioning"""
        # Create white background
        final_surface = pygame.Surface(self.screen_size)
        final_surface.fill((255, 255, 255))
        
        try:
            # Calculate visible area
            visible_width = self.screen_size.x / self.zoom
            visible_height = self.screen_size.y / self.zoom
            src_rect = pygame.Rect(
                self.offset.x,
                self.offset.y,
                visible_width,
                visible_height
            )
            
            # Clamp to image boundaries
            src_rect.width = min(src_rect.width, self.world_size.x - src_rect.left)
            src_rect.height = min(src_rect.height, self.world_size.y - src_rect.top)
            
            if src_rect.width > 0 and src_rect.height > 0:
                # Get and scale subsurface
                subsurf = surface.subsurface(src_rect)
                scaled_size = (int(src_rect.width * self.zoom), int(src_rect.height * self.zoom))
                scaled_surf = pygame.transform.smoothscale(subsurf, scaled_size)
                
                # Center the image if smaller than screen
                pos_x = (self.screen_size.x - scaled_size[0]) // 2
                pos_y = (self.screen_size.y - scaled_size[1]) // 2
                final_surface.blit(scaled_surf, (pos_x, pos_y))
                
        except (ValueError, pygame.error):
            pass
        
        # Draw to screen
        self.screen.blit(final_surface, (0, 0))


# Screen setup
# Replace screen setup with:
viewport = Viewport(1200, 800)
current_track = None

# Simulation time before moving to next generation (in seconds, assuming 60fps)
GEN_MAX_TIME = 30

def handle_viewport_controls(): 
    keys = pygame.key.get_pressed() 
    pan_speed = 30 / viewport.zoom 
    
    if keys[pygame.K_LEFT]: 
        viewport.offset.x = max(0, viewport.offset.x - pan_speed) 
    if keys[pygame.K_RIGHT]: 
        max_x = max(0, viewport.world_size.x - (viewport.screen.get_width()/viewport.zoom)) 
        viewport.offset.x = min(max_x, viewport.offset.x + pan_speed) 
    if keys[pygame.K_UP]: 
        viewport.offset.y = max(0, viewport.offset.y - pan_speed) 
    if keys[pygame.K_DOWN]: 
        max_y = max(0, viewport.world_size.y - (viewport.screen.get_height()/viewport.zoom)) 
        viewport.offset.y = min(max_y, viewport.offset.y + pan_speed) 


def draw_robots(robots:LineFollowerNEAT):
    for robot in robots:
        robot.draw("agent/robot.png", draw_robot=True, opacity=200)
    
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
        line_follower = LineFollowerNEAT(genome=genome,
                                    neat_config=config,
                                    robot_config=robot_config,
                                    screen=temp_surface,
                                    sensor_count=15)
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
            handle_viewport_controls()
            
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
    GENERATIONS = 100

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
