import neat.config
import neat.genome
import numpy as np
import pygame

from .dc_motor import DCMotor


class LineFollower:
    """
    Class simulating a line following bot with differential steering in a pygame environment.
    This implementation uses a semi circle sensor array for line detection.
    """

    def __init__(self,
                 robot_config   : dict,
                 screen         :pygame.display,          
                 sensor_count   : int = 15):
        """
        Initialize the bot.
        :param genome: NEAT genome
        :param neat_config: NEAT configuration
        :param robot_config: Configuration dictionary with keys:
                       - 'start_xy': (x, y) coordinates for the robot to start at
                       - 'start_yaw': Starting yaw in RAD
                       - 'sensor_range': Distance from bot center (motors) to sensor tip.
                       - 'sensor_angle_range': Total angular range (in radians) for the sensor array.
                       - 'wheels_spacing': Distance between the two wheels in m.
                       - 'meter_to_pixels': Represent each meter in this quantity of pixels in the simulation
                       - 'max_speed': Maximum speed of the bot in m/s.
                       - 'motor_nominal_voltage': [V], 'motor_no_load_rpm': [RPM], 'motor_stall_torque': [Nm], 'volts': [V]
                         for motor simulation, note that volts is the applied voltage to the motor in the simulation.
        :param screen: PyGame Screen (e.g. screen = pygame.display.set_mode([WIDTH, HEIGHT]))
        :param sensor_count: Number of sensors in the semi circle array.
        """
        # convert normal units to pixels
        # Create a copy of the robot_config dictionary
        # this is done to not reference the same dictionary between different objects
        self.robot_config = robot_config['robot_config'].copy()
        self.robot_config["max_speed"] *= self.robot_config.get("meter_to_pixels")
        self.robot_config["wheels_spacing"] *= self.robot_config.get("meter_to_pixels")

        # set bot configs
        self.position = np.array(self.robot_config.get('start_xy', (0,0)), dtype=float)
        self.yaw = self.robot_config.get('start_yaw')

        # Differential drive state: wheel velocities (left and right)
        self.left_wheel_velocity = 0.0
        self.right_wheel_velocity = 0.0

        # Motor initialization (if using motor dynamics)
        self.volts = self.robot_config.get("volts", 12.0)
        nom_volt = self.robot_config.get("motor_nominal_voltage", 12.0)
        no_load_speed = self.robot_config.get("motor_no_load_rpm", 100.0) * 2*np.pi/60  # convert RPM to RAD/s
        stall_torque = self.robot_config.get("motor_stall_torque", 1.0)  # torque in [Nm]
        self.left_motor = DCMotor(nom_volt, no_load_speed, stall_torque)
        self.right_motor = DCMotor(nom_volt, no_load_speed, stall_torque)

        # Sensor configuration
        self.sensor_count = sensor_count
        self.sensor_range = self.robot_config.get("sensor_range", 50.0)  # in pixels or simulation units
        self.sensor_angle_range = self.robot_config.get("sensor_angle_range", np.pi)  # radians; default is a semi circle

        # pygame screen setup
        self.screen = screen

        # For storing sensor readings
        self.sensor_readings = np.zeros(self.sensor_count)

        # for fitness function purposes
        self.off_track_time = 0

        # step count is for the number of steps the bot lasted in the simulation int he smae attempt
        # this is used for logging the data of the bot in each attempt
        self.step_count = 0

    def reset(self, start_xy=None, start_yaw=None):
        """
        Reset the bot's position and orientation.
        :param start_xy: New starting (x, y) coordinates.
        :param start_yaw: New starting yaw in radians.
        """
        # position
        if start_xy is None:
            self.position = np.array(self.robot_config.get('start_xy', (0,0)), dtype=float)
        else:
            self.position = start_xy
        # yaw
        if start_yaw is None:
            self.yaw = start_yaw
        else:
            self.yaw = start_yaw
        # reset wheel speed
        self.left_wheel_velocity = 0.0
        self.right_wheel_velocity = 0.0

    def get_position(self):
        """
        Return the current position and yaw.
        :return: (position, yaw)
        """
        return self.position.copy(), self.yaw

    def get_velocity(self):
        """
        Compute bot's linear and angular velocities based on wheel speeds.
        :return: (linear_velocity, angular_velocity)
        """
        # Differential drive kinematics:
        v = (self.left_wheel_velocity + self.right_wheel_velocity) / 2.0
        wheels_spacing = self.robot_config.get("wheels_spacing", 30.0)  # default value if not provided
        omega = (self.right_wheel_velocity - self.left_wheel_velocity) / wheels_spacing
        return v, omega

    def _power_to_volts(self, l_pow, r_pow):
        """
        Convert normalized power values [-1, 1] to voltage.
        """
        l_pow = np.clip(l_pow, -1.0, 1.0)
        r_pow = np.clip(r_pow, -1.0, 1.0)
        return l_pow * self.volts, r_pow * self.volts

    def apply_action(self, action):
        """
        Apply an action to the bot.
        :param action: Tuple (left_power, right_power), each within [-1, 1].
        """
        l_volts, r_volts = self._power_to_volts(*action)
        # For simplicity, we assume instantaneous response.
        max_speed = self.robot_config.get("max_speed")  # in pixel/s
        self.left_wheel_velocity = (l_volts / self.volts) * max_speed
        self.right_wheel_velocity = (r_volts / self.volts) * max_speed

    def get_line_sensor_readings(self):
        """
        Get the line sensor readings by finding the pixel color under each sensor
        :return: numpy array of sensor readings.
        """
        readings = np.zeros(self.sensor_count)
        # Evenly distribute sensor rays over the sensor_angle_range.
        angles = np.linspace(-self.sensor_angle_range / 2, self.sensor_angle_range / 2, self.sensor_count)
        for i, sensor_angle in enumerate(angles):
            # Calculate the global angle for each sensor ray.
            ray_angle = self.yaw + sensor_angle
            
            # Compute sensor tip position.
            sensor_x = int(self.position[0] + self.sensor_range * np.cos(ray_angle))
            sensor_y = int(self.position[1] + self.sensor_range * np.sin(ray_angle))
            
            # Check if the sensor position is within the screen bounds.
            # Note: The screen is assumed to be the same size as the simulation area.
            if 0 <= sensor_x < self.screen.get_width() and 0 <= sensor_y < self.screen.get_height():
                # Check the color at the sensor position.
                pixel_color = self.screen.get_at((sensor_x, sensor_y))[:3]
                if pixel_color == (255, 255, 255):  # White background
                    readings[i] = 0
                elif pixel_color == (0, 0, 0) or pixel_color == (255, 255, 0):  # Black or Yellow line, yellow is checked for with black because detecting yellow is a reward 
                    readings[i] = 1
            
        self.sensor_readings = readings
        return readings
    
    def get_color(self):
        """a method for checking the color of any of the sensors"""
        yaw_rad = np.radians(self.yaw)
        sensor_x = int(self.position[0] + self.sensor_range * np.cos(yaw_rad))
        sensor_y = int(self.position[1] - self.sensor_range * np.sin(yaw_rad))
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

    def step(self, dt):
        """
        Update the bot's state over a time step.
        :param dt: Time step in seconds.
        :param track: Track object used for obtaining sensor readings.
        """
        # Compute differential drive kinematics.
        v = (self.left_wheel_velocity + self.right_wheel_velocity) / 2.0
        wheels_spacing = self.robot_config.get("wheels_spacing", 30.0)
        omega = (self.right_wheel_velocity - self.left_wheel_velocity) / wheels_spacing

        # Update position and orientation.
        self.position[0] += v * np.cos(self.yaw) * dt
        self.position[1] += v * np.sin(self.yaw) * dt
        self.yaw += omega * dt


    def draw(self,
             img_path:str=None,
             draw_robot:bool=True,
             opacity:int=255,
             image_size:tuple=(137, 195),
             surface=None):
        """
        Draw the bot and its sensor rays on the given pygame surface.
        :parap img_path: Path to the robot image to draw on the surface.
        :param draw_robot: Boolean flag to draw the bot.
        :param opacity: Opacity of the bot and sensor rays (0 to 255).
        :param image_size: Size of the robot image in pixels (width, height).
        :param surface: Pygame Surface object. This can be the main display surface (often called screen),
                        or any other surface for off-screen rendering.
        """
        # Draw the bot as a circle.
        bot_color = (0, 255, 0)
        bot_radius = 10  # in pixels

        # if no screen was provided use the default screen originaly defined
        if surface is None:
            surface = self.screen

        if draw_robot:
            robot_image = pygame.image.load(img_path).convert_alpha()
            robot_image.set_alpha(int(opacity))
            # Control the image size in pixels by setting desired dimensions.
            desired_size = image_size  # width and height in pixels
            robot_image = pygame.transform.scale(robot_image, desired_size)
            # Rotate the image to match the orientation of the bot.
            rotated_image = pygame.transform.rotate(robot_image, -np.degrees(self.yaw) - 90)  # -90 aligns image with bot's orientation
            # Center the image at the bot's current position.
            rect = rotated_image.get_rect(center=(int(self.position[0]), int(self.position[1])))
            surface.blit(rotated_image, rect)

        # Draw sensor rays.
        sensor_color = (125, 125, 125)
        sensor_active_color = (0, 0, 255)
        angles = np.linspace(-self.sensor_angle_range / 2, self.sensor_angle_range / 2, self.sensor_count)
        for i, sensor_angle in enumerate(angles):
            ray_angle = self.yaw + sensor_angle
            sensor_end_x = self.position[0] + self.sensor_range * np.cos(ray_angle)
            sensor_end_y = self.position[1] + self.sensor_range * np.sin(ray_angle)
            
            if self.sensor_readings[i] == 1:
                pygame.draw.circle(surface, sensor_active_color, (int(sensor_end_x), int(sensor_end_y)), 2)
            else:
                pygame.draw.circle(surface, sensor_color, (int(sensor_end_x), int(sensor_end_y)), 2)
                

class LineFollowerNEAT(LineFollower):
    """
    Class simulating a line following bot with differential steering in a pygame environment.
    This implementation uses a semi circle sensor array for line detection.
    """

    def __init__(self,
                 genome         : neat.genome,
                 neat_config    : neat.config.Config,
                 robot_config   : dict,
                 screen         :pygame.display,          
                 sensor_count   : int = 15):
        """
        Initialize the bot.
        :param genome: NEAT genome
        :param neat_config: NEAT configuration
        :param robot_config: Configuration dictionary with keys:
                       - 'start_xy': (x, y) coordinates for the robot to start at
                       - 'start_yaw': Starting yaw in RAD
                       - 'sensor_range': Distance from bot center (motors) to sensor tip.
                       - 'sensor_angle_range': Total angular range (in radians) for the sensor array.
                       - 'wheels_spacing': Distance between the two wheels in m.
                       - 'meter_to_pixels': Represent each meter in this quantity of pixels in the simulation
                       - 'max_speed': Maximum speed of the bot in m/s.
                       - 'motor_nominal_voltage': [V], 'motor_no_load_rpm': [RPM], 'motor_stall_torque': [Nm], 'volts': [V]
                         for motor simulation, note that volts is the applied voltage to the motor in the simulation.
        :param screen: PyGame Screen (e.g. screen = pygame.display.set_mode([WIDTH, HEIGHT]))
        :param sensor_count: Number of sensors in the semi circle array.
        """
        # convert normal units to pixels
        # Create a copy of the robot_config dictionary
        # this is done to not reference the same dictionary between different objects
        self.robot_config = robot_config['robot_config'].copy()
        self.robot_config["max_speed"] *= self.robot_config.get("meter_to_pixels")
        self.robot_config["wheels_spacing"] *= self.robot_config.get("meter_to_pixels")

        # set the neural netwrok of the bot
        self.net = neat.nn.RecurrentNetwork.create(genome, neat_config)

        # set bot configs
        self.position = np.array(self.robot_config.get('start_xy', (0,0)), dtype=float)
        self.yaw = self.robot_config.get('start_yaw')

        # Differential drive state: wheel velocities (left and right)
        self.left_wheel_velocity = 0.0
        self.right_wheel_velocity = 0.0

        # Motor initialization (if using motor dynamics)
        self.volts = self.robot_config.get("volts", 12.0)
        nom_volt = self.robot_config.get("motor_nominal_voltage", 12.0)
        no_load_speed = self.robot_config.get("motor_no_load_rpm", 100.0) * 2*np.pi/60  # convert RPM to RAD/s
        stall_torque = self.robot_config.get("motor_stall_torque", 1.0)  # torque in [Nm]
        self.left_motor = DCMotor(nom_volt, no_load_speed, stall_torque)
        self.right_motor = DCMotor(nom_volt, no_load_speed, stall_torque)

        # Sensor configuration
        self.sensor_count = sensor_count
        self.sensor_range = self.robot_config.get("sensor_range", 50.0)  # in pixels or simulation units
        self.sensor_angle_range = self.robot_config.get("sensor_angle_range", np.pi)  # radians; default is a semi circle

        # pygame screen setup
        self.screen = screen

        # For storing sensor readings
        self.sensor_readings = np.zeros(self.sensor_count)

        # for fitness function purposes
        self.off_track_time = 0

    def step(self, dt):
        """
        Update the bot's state over a time step.
        :param dt: Time step in seconds.
        :param track: Track object used for obtaining sensor readings.
        """
        # Update sensor readings.
        self.get_line_sensor_readings()

        # activate the network
        output = self.net.activate([self.left_wheel_velocity, self.right_wheel_velocity, *self.sensor_readings])

        # modify the motor speed (wheel velocity) based on the network's output
        self.apply_action(output)

        # Compute differential drive kinematics.
        v = (self.left_wheel_velocity + self.right_wheel_velocity) / 2.0
        wheels_spacing = self.robot_config.get("wheels_spacing", 30.0)
        omega = (self.right_wheel_velocity - self.left_wheel_velocity) / wheels_spacing

        # Update position and orientation.
        self.position[0] += v * np.cos(self.yaw) * dt
        self.position[1] += v * np.sin(self.yaw) * dt
        self.yaw += omega * dt


class LineFollowerPID(LineFollower):
    def __init__(self, config: dict, screen: pygame.display, sensor_count: int = 15):
        # Extract robot config for parent class
        super().__init__(config, screen, sensor_count)
        
        # PID parameters from config
        self.Kp = config['PID']['Kp']
        self.Ki = config['PID']['Ki']
        self.Kd = config['PID']['Kd']
        
        # PID state
        self.integral = 0.0
        self.prev_error = 0.0
        self.error_history = []

    def compute_pid_action(self, sensor_readings):
        # Calculate weighted error position
        sensor_count = len(sensor_readings)
        center_index = (sensor_count - 1) / 2
        weighted_sum = 0.0
        total_weight = 0.0
        
        for i, reading in enumerate(sensor_readings):
            distance_from_center = i - center_index
            weighted_sum += reading * distance_from_center
            total_weight += reading

        # Handle line detection failure
        if total_weight == 0:
            # Use decaying error from history
            error = self.prev_error * 0.8 if self.error_history else 0
            self.off_track_time += 1/60
        else:
            error = weighted_sum / total_weight
            self.off_track_time = max(0, self.off_track_time - 1/60)
            self.error_history.append(error)

        # PID calculations
        self.integral += error * self.Ki
        derivative = (error - self.prev_error) * self.Kd
        
        # Anti-windup clamping
        self.integral = np.clip(self.integral, -1.0, 1.0)
        
        # Combine terms
        control = (error * self.Kp) + self.integral + derivative
        
        # Update state
        self.prev_error = error
        
        # Calculate motor outputs
        base_speed = 0.7  # From config if needed
        left = np.clip(base_speed + control, 0.0, 1.0)
        right = np.clip(base_speed - control, 0.0, 1.0)

        return [left, right]
