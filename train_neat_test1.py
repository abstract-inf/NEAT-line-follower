# this code is a test on how to build a line follower AI agent using NEAT algorithm in a simulation using pygame

import pygame
import math
import neat
import os
import pickle
import visualize
import random

pygame.init()

# Set up the drawing window
WIDTH, HEIGHT = 1000, 1000
screen = pygame.display.set_mode([WIDTH, HEIGHT])
pygame.display.set_caption("NEAT Line Follower")

# load line path image
line_path = pygame.image.load("line_path2.png")
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
        self.rect_x, self.rect_y = random.choice([(840,610), (100,380)]) # starting position
        self.rect_angle = 90  # Facing upwards initially
        self.SPEED = 4  # Movement speed
        self.ROTATION_SPEED = 7  # Rotation speed

        self.sensors_data = [0 for _ in range(self.num_sensors)]  # sensor data

    def move(self):
        self.sensors_data = self.get_sensors_data()
        output = self.net.activate(self.sensors_data)

        # Movement (forward/backward in direction of rotation)
        angle_rad = math.radians(self.rect_angle)
        self.rect_x += self.SPEED * math.cos(angle_rad)
        self.rect_y -= self.SPEED * math.sin(angle_rad)

        # Rotation (left/right)
        if output[0] > 0.5:  # Turn left
            self.rect_angle = (self.rect_angle + self.ROTATION_SPEED) % 360
            return True
             
        elif output[1] > 0.5:  # Turn right
            self.rect_angle = (self.rect_angle - self.ROTATION_SPEED) % 360
            return True
        else:
            return False



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

def eval_genomes(genomes, config):
    genes = [] 
    robots = []
    for genome_id, genome in genomes:
        genome.fitness = 0  # start with fitness level of 0
        line_follower = LineFollower(genome, config)
        genes.append(genome)
        robots.append(line_follower)

    running = True
    clock = pygame.time.Clock()
    ticks = 0
    while running and len(robots) > 0 and ticks < 60*10: # ticks is less than 10 seconds
        ticks += 1
        clock.tick(0)

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


        for i, robot in enumerate(robots):
            try:
                if robot.move(): # if the robot turns
                    genes[i].fitness -= 1  # punish overturning

                robot.draw()
            
                # ---- Fitness function ----
                # Check if any sensor is active
                if any(sensor == 1 for sensor in robot.sensors_data):
                    # At least one sensor is active, so reward accordingly
                    # for j in range(robot.num_sensors // 2 + 1):
                        if robot.sensors_data[robot.num_sensors // 2] == 1:
                            genes[i].fitness += 5
                        else:
                            genes[i].fitness += 0.1
                        # elif robot.sensors_data[j] == 1 or robot.sensors_data[robot.num_sensors - j - 1] == 1:
                        #     genes[i].fitness -= 5 / (j+1)
                        # Note: No penalty here since some sensor is active
                else:
                    # All sensors are inactive: apply the penalty once
                    genes[i].fitness -= 5


            except IndexError:
                # print("IndexError, i:", i)
                # print("len(robots):", len(robots))
                # print("len(genes):", len(genes))
                robots[i].genome.fitness -= 500
                robots.pop(i)
                genes.pop(i)
                break

        # Update the display
        pygame.display.update()


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config_feedforward.txt')
    
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    #p.add_reporter(neat.Checkpointer(5))

    # Run for up to 50 generations.
    winner = p.run(eval_genomes, 100)
    pickle.dump(winner ,open("best_genome.pickle", "wb"))

    # show final stats
    print('\nBest genome:\n{!s}'.format(winner))

    # Visualize the training stats and species growth
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)

    # Draw the best network found
    visualize.draw_net(config, winner, view=True)