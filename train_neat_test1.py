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
import colorsys

import pygame
import neat
import visualize

# Initialize Pygame
pygame.init()

# Screen setup
WIDTH, HEIGHT = 1000, 1000
screen = pygame.display.set_mode([WIDTH, HEIGHT])
pygame.display.set_caption("NEAT Line Follower")

# Load and scale the line path image
line_path = pygame.image.load("line_paths/line_path4.png")
line_path = pygame.transform.scale(line_path, (WIDTH, HEIGHT))

# Simulation time before moving to next generation (in relative to running the simulation at 60fps)
GEN_MAX_TIME = 20

class LineFollower:
    """Represents a robot agent with sensors, controlled by a NEAT-evolved neural network."""
    # Body dimensions
    rect_width, rect_height = 50, 20

    # Sensor properties
    num_sensors = 15
    front_sensor_distance = 50  # Distance for sensor reading
    spread_angle = 135          # Total spread angle for sensors
    spacing = spread_angle / (num_sensors - 1)  # Even spacing between sensors

    def __init__(self, genome, config, color=(255, 115, 115)):
        self.genome = genome
        self.config = config
        self.color = color  # Assign species color

        self.net = neat.nn.RecurrentNetwork.create(genome, config)

        # Create a rectangle surface with the species color
        self.rectangle = pygame.Surface((self.rect_width, self.rect_height), pygame.SRCALPHA)
        self.rectangle.fill(self.color)

        # Starting position (randomly chosen from two options)
        self.rect_x, self.rect_y = random.choice([(775, 840), (200, 840)])
        self.rect_angle = 90  # Facing upwards initially
        self.SPEED = 4        # Movement speed
        self.ROTATION_SPEED = 7  # Rotation speed

        self.sensors_data = [0 for _ in range(self.num_sensors)]

    def move(self):
        """Activates the network to move the robot and returns True if it turns."""
        self.sensors_data = self.get_sensors_data()
        output = self.net.activate(self.sensors_data)

        # Update position based on the current angle
        angle_rad = math.radians(self.rect_angle)
        self.rect_x += self.SPEED * math.cos(angle_rad)
        self.rect_y -= self.SPEED * math.sin(angle_rad)

        # Adjust rotation based on network output
        if output[0] > 0.5:  # Turn left
            self.rect_angle = (self.rect_angle + self.ROTATION_SPEED) % 360
            return True
        elif output[1] > 0.5:  # Turn right
            self.rect_angle = (self.rect_angle - self.ROTATION_SPEED) % 360
            return True
        else:
            return False

    def get_sensors_data(self, draw=True):
        """Computes sensor values based on the line path colors and optionally draws sensor circles."""
        for i in range(self.num_sensors):
            # Calculate sensor angle offset from the center
            offset = (i - (self.num_sensors - 1) / 2) * self.spacing
            angle_rad = math.radians(self.rect_angle)
            sensor_angle = angle_rad + math.radians(offset)
            sensor_x = int(self.rect_x + self.front_sensor_distance * math.cos(sensor_angle))
            sensor_y = int(self.rect_y - self.front_sensor_distance * math.sin(sensor_angle))

            # Check the color at the sensor position
            pixel_color = screen.get_at((sensor_x, sensor_y))[:3]
            if pixel_color == (255, 255, 255):  # White background
                self.sensors_data[i] = 0
                if draw:
                    pygame.draw.circle(screen, (125, 125, 125), (sensor_x, sensor_y), 5)
            elif pixel_color == (0, 0, 0):  # Black line
                self.sensors_data[i] = 1
                if draw:
                    pygame.draw.circle(screen, (255, 0, 0), (sensor_x, sensor_y), 5)

        return self.sensors_data

    def draw(self):
        """Draws the robot's rotated rectangle on the screen."""
        rotated_surface = pygame.transform.rotate(self.rectangle, self.rect_angle)
        rotated_rect = rotated_surface.get_rect(center=(self.rect_x, self.rect_y))
        screen.blit(rotated_surface, rotated_rect.topleft)

def get_species_colors(species_ids):
    """Assigns a unique color to each species ID."""
    unique_species = list(set(species_ids))
    num_species = len(unique_species)
    
    species_colors = {}
    for i, species in enumerate(unique_species):
        hue = i / max(1, num_species)  # Evenly distribute colors in the hue spectrum
        rgb = colorsys.hsv_to_rgb(hue, 1, 1)  # Convert HSV to RGB
        species_colors[species] = tuple(int(c * 255) for c in rgb)  # Convert to 0-255 range
    
    return species_colors


def draw_stop_button(screen):
    """Draws a stop button in pygame."""
    font = pygame.font.Font(None, 30)
    button_rect = pygame.Rect(10, 10, 120, 40)  # (x, y, width, height)
    
    pygame.draw.rect(screen, (200, 0, 0), button_rect)  # Red button
    text_surface = font.render("Stop", True, (255, 255, 255))
    screen.blit(text_surface, (button_rect.x + 30, button_rect.y + 10))
    
    return button_rect  # Return button rect for click detection

def calculate_fitness(robots, genes):
    # Process each robot
    for i, robot in enumerate(robots):
        try:
            if robot.move():
                genes[i].fitness -= 2  # Penalize turning
            robot.draw()

            # Fitness function: reward if any sensor is active
            if any(sensor == 1 for sensor in robot.sensors_data):
                if robot.sensors_data[robot.num_sensors // 2 + 1] == 1:
                    genes[i].fitness += 5
                else:
                    genes[i].fitness += 0.5
            else:
                genes[i].fitness -= 5  # Penalize if no sensor is active

        except IndexError:
            # In case of sensor index issues, heavily penalize and remove the robot
            robot.genome.fitness -= 500
            robots.pop(i)
            genes.pop(i)
            break
        
def eval_genomes(genomes, config):
    """
    Evaluates each genome by simulating the robot's movement.
    Adjusts fitness based on sensor activity and penalizes turning.
    """
    genes = [] 
    robots = []
    
    for genome_id, genome in genomes:
        genome.fitness = 0  # Initialize fitness

        line_follower = LineFollower(genome, config)
        genes.append(genome)
        robots.append(line_follower)


    running = True
    clock = pygame.time.Clock()
    ticks = 0

    # Run simulation loop for 10 seconds (assuming 60 ticks per second)
    while running and robots and ticks < 60 * GEN_MAX_TIME:
        ticks += 1
        clock.tick(0)

        # Draw background
        screen.fill((255, 255, 255))
        screen.blit(line_path, (0, 0))
    
        # Draw stop button
        stop_button = draw_stop_button(screen)
       
        # Check for events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False  # Quit program
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if stop_button.collidepoint(event.pos):  # Check if button clicked
                    # running = False  # Stop training
                    return


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

    # Load NEAT configuration
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_path)

    continue_learning = True

    # Restore from the latest checkpoint if continuing training
    if continue_learning:
        print("Loading checkpoint from last model...")
        checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "neat-checkpoint-*"))
        if checkpoint_files:
            latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
            population = neat.Checkpointer.restore_checkpoint(latest_checkpoint)
            print("Restored from checkpoint:", latest_checkpoint)
        else:
            print("No checkpoint found. Starting a new population.")
            population = neat.Population(config)
    else:
        print("Training a new model...")
        population = neat.Population(config)

    # Add NEAT reporters
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    # Save a checkpoint every 5 generations into the checkpoint folder
    checkpoint_prefix = os.path.join(checkpoint_dir, "neat-checkpoint-")
    population.add_reporter(neat.Checkpointer(generation_interval=10, filename_prefix=checkpoint_prefix))

    # Run the NEAT algorithm
    generations = 20 if continue_learning else 30
    winner = population.run(eval_genomes, generations)

    print("\nBest genome:\n{!s}".format(winner))
    save_model(winner)

    # Visualize training statistics and the best network
    visualize.plot_stats(stats, ylog=False, view=True, filename="")
    visualize.plot_species(stats, view=True, filename="")
    visualize.draw_net(config, winner, view=True, filename="Net")


if __name__ == "__main__":
    main()
