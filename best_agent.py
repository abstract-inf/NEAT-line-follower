# this code is a test on how to build a line follower AI agent using NEAT algorithm in a simulation using pygame

import pygame
import math
import neat
import os
import pickle
import visualize

pygame.init()

# Set up the drawing window
WIDTH, HEIGHT = 1000, 1000
screen = pygame.display.set_mode([WIDTH, HEIGHT])
pygame.display.set_caption("NEAT Line Follower")

# load line path image
line_path = pygame.image.load("line_paths/line_path4.png")
line_path = pygame.transform.scale(line_path, (WIDTH, HEIGHT))

class LineFollower:
    rect_width, rect_height = 50, 20  # body dimensions

    # Sensor properties
    num_sensors = 15
    front_sensor_distance = 50  # Length of the arrow
    spread_angle = 135  # Spread angle of the circles 
    spacing = spread_angle / (num_sensors - 1)  # Calculate the spacing so the circles are evenly distributed over a 90° arc
    
    def __init__(self, genome, config):
        self.genome = genome
        self.config = config

        self.net = neat.nn.RecurrentNetwork.create(genome, config)

        # Load/Create a rectangle surface
        self.rectangle = pygame.Surface((self.rect_width, self.rect_height), pygame.SRCALPHA)
        self.rectangle.fill((255, 115, 115))

        # Rectangle properties
        self.rect_x, self.rect_y = 200, 840  # starting position
        self.rect_angle = 90  # Facing upwards initially
        self.SPEED = 4  # Movement speed
        self.ROTATION_SPEED = 7  # Rotation speed

        self.sensors_data = [0 for _ in range(self.num_sensors)]  # sensor data

    def move(self):
        self.sensors_data = self.get_sensors_data()
        output = self.net.activate(self.sensors_data)
        # print(output)

        # Rotation (left/right)
        if output[0] > 0.5:  # Turn left
            self.rect_angle = (self.rect_angle + self.ROTATION_SPEED) % 360  
        if output[1] > 0.5:  # Turn right
            self.rect_angle = (self.rect_angle - self.ROTATION_SPEED) % 360  

        # Movement (forward/backward in direction of rotation)
        angle_rad = math.radians(self.rect_angle)
        # if output[2] > 0.5:  # Move forward
        # always move forward
        self.rect_x += self.SPEED * math.cos(angle_rad)
        self.rect_y -= self.SPEED * math.sin(angle_rad)
        # if output[3] > 0.5:  # Move backward
        #     self.rect_x -= self.SPEED * math.cos(angle_rad)
        #     self.rect_y += self.SPEED * math.sin(angle_rad)

    def get_sensors_data(self, draw=True):
        for i in range(self.num_sensors):
            # Calculate offset from the center (forward direction)
            offset = (i - (self.num_sensors - 1) / 2) * self.spacing
            angle_rad = math.radians(self.rect_angle)
            sensor_angle = angle_rad + math.radians(offset)
            sensor_x = int(self.rect_x + self.front_sensor_distance * math.cos(sensor_angle))
            sensor_y = int(self.rect_y - self.front_sensor_distance * math.sin(sensor_angle))

            # print(f"Circle {i} color: {screen.get_at((front_circle_x, front_circle_y))[:3]}", end=", ") # print the color under the circle

            if screen.get_at((sensor_x, sensor_y))[:3] == (255, 255, 255):  # if the color is white
                self.sensors_data[i] = 0
                if draw:
                    pygame.draw.circle(screen, (125,125,125), (sensor_x, sensor_y), 5)

            elif screen.get_at((sensor_x, sensor_y))[:3] == (0, 0, 0): # the color is black
                self.sensors_data[i] = 1
                if draw:
                    pygame.draw.circle(screen, (255, 0, 0), (sensor_x, sensor_y), 5)

        return self.sensors_data

    def draw(self):
        # Rotate the rectangle and update its position
        rotated_surface = pygame.transform.rotate(self.rectangle, self.rect_angle)
        rotated_rect = rotated_surface.get_rect(center=(self.rect_x, self.rect_y))

        # Draw the rotated rectangle
        screen.blit(rotated_surface, rotated_rect.topleft)

def run_example(genome, config):
    line_follower = LineFollower(genome, config)

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