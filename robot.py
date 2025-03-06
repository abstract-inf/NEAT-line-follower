import neat 
import pygame
import random
import math

class LineFollower:
    """Represents a robot agent with sensors, controlled by a NEAT-evolved neural network."""
    # Body dimensions
    rect_width, rect_height = 50, 20

    # Sensor properties
    num_sensors = 15
    front_sensor_distance = 50  # Distance for sensor reading
    spread_angle = 135          # Total spread angle for sensors
    spacing = spread_angle / (num_sensors - 1)  # Even spacing between sensors

    def __init__(self, genome, config, screen, color=(255, 115, 115)):
        self.genome = genome
        self.config = config
        self.color = color  # Assign species color
        self.screen = screen

        self.net = neat.nn.RecurrentNetwork.create(genome, config)

        # Create a rectangle surface with the species color
        self.rectangle = pygame.Surface((self.rect_width, self.rect_height), pygame.SRCALPHA)
        self.rectangle.fill(self.color)

        # Starting position (randomly chosen from two options)
        self.rect_x, self.rect_y, self.rect_angle = random.choice([(1350, 600, 180), 
                                                                   (1350, 600, 180)])
        self.sensors_data = [0 for _ in range(self.num_sensors)]
        
        # motors control parameters
        self.MAX_SPEED = 8
        self.right_motor_speed = 0
        self.left_motor_speed = 0

        self.off_track_time = 0  # Track consecutive off-track time

    def move(self):
        """
        Activates the network to move the robot using differential drive logic.
        The network outputs represent the left and right motor speed factors.
        Returns the absolute angular change (in degrees) for potential fitness penalty.
        """
        self.sensors_data = self.get_sensors_data(draw=True)
        output = self.net.activate([self.right_motor_speed, self.left_motor_speed, *self.sensors_data])

        # network outputs two values in the range [-1, 1]
        self.left_motor = output[0]  * self.MAX_SPEED if output[0] > 0 else 0 # left motor speed factor
        self.right_motor = output[1] * self.MAX_SPEED if output[1] > 0 else 0 # right motor speed factor

        # Differential drive physics:
        # Forward velocity = average of left and right motor speeds scaled by self.SPEED.
        forward_velocity = ((self.left_motor + self.right_motor) / 2)

        # Angular velocity: difference between right and left speeds.
        # wheel_distance is the distance between the two motors (pixels).
        wheel_distance = 150.0
        angular_velocity_rad = (self.right_motor - self.left_motor) / wheel_distance  # radians per tick
        angular_velocity_deg = math.degrees(angular_velocity_rad)

        # Update position: move forward in the current direction.
        angle_rad = math.radians(self.rect_angle)
        self.rect_x += forward_velocity * math.cos(angle_rad)
        self.rect_y -= forward_velocity * math.sin(angle_rad)

        # Update orientation: add angular velocity (in degrees).
        self.rect_angle = (self.rect_angle + angular_velocity_deg) % 360

        # Return the magnitude of the angular change for fitness adjustments.
        return abs(angular_velocity_deg), self.left_motor, self.right_motor

    def get_sensors_data(self, draw=True):
        """Computes sensor values based on the line path colors and optionally draws sensor circles."""
        for i in range(self.num_sensors):
            # Calculate sensor angle offset from the center.
            offset = (i - (self.num_sensors - 1) / 2) * self.spacing
            angle_rad = math.radians(self.rect_angle)
            sensor_angle = angle_rad + math.radians(offset)
            sensor_x = int(self.rect_x + self.front_sensor_distance * math.cos(sensor_angle))
            sensor_y = int(self.rect_y - self.front_sensor_distance * math.sin(sensor_angle))

            # Check the color at the sensor position.
            pixel_color = self.screen.get_at((sensor_x, sensor_y))[:3]
            if pixel_color == (255, 255, 255):  # White background
                self.sensors_data[i] = 0
                if draw:
                    pygame.draw.circle(self.screen, (125, 125, 125), (sensor_x, sensor_y), 5)
            elif pixel_color == (0, 0, 0) or pixel_color == (255, 255, 0):  # Black or Yellow line 
                self.sensors_data[i] = 1
                if draw:
                    pygame.draw.circle(self.screen, (0, 0, 255), (sensor_x, sensor_y), 5)

        return self.sensors_data

    def get_color(self):
        """a method for checking the color of any of the sensors"""
        angle_rad = math.radians(self.rect_angle)
        sensor_x = int(self.rect_x + self.front_sensor_distance * math.cos(angle_rad))
        sensor_y = int(self.rect_y - self.front_sensor_distance * math.sin(angle_rad))
        # Check the color at the sensor position.
        pixel_color = self.screen.get_at((sensor_x, sensor_y))
        # pygame.draw.circle(self.screen, (100, 100, 255), (sensor_x, sensor_y), 5)
        if pixel_color == (0, 255, 0):  # Green background
            return "green"
        elif pixel_color == (255, 0, 0):  # Red background
            return "red"
        elif pixel_color == (255, 255, 0): # Yellow background
            return "yellow"
            
        return pixel_color


    def draw(self):
        """Draws the robot's rotated rectangle on the screen."""
        rotated_surface = pygame.transform.rotate(self.rectangle, self.rect_angle)
        rotated_rect = rotated_surface.get_rect(center=(self.rect_x, self.rect_y))
        self.screen.blit(rotated_surface, rotated_rect.topleft)

