# line_follower.py
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
                 sensor_type    : str = "jsumo_xline_v2",
                 sensor_count   : int = 15,
                 draw_robot:bool=True,
                 img_path:str=None,
                 opacity:int=255,
                 image_size:tuple=(190, 167)
                 ):
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
                       - 'motor_nominal_voltage': [V], 
                       - 'motor_no_load_rpm': [RPM], 
                       - 'motor_stall_torque': [Nm], 
                       - 'volts': [V]
                         for motor simulation, note that volts is the applied voltage to the motor in the simulation.
        :param screen: PyGame Screen (e.g. screen = pygame.display.set_mode([WIDTH, HEIGHT]))
        :param sensor_count: Number of sensors in the semi circle array.
        :parap img_path: Path to the robot image to draw on the surface.
        :param draw_robot: Boolean flag to draw the bot.
        :param opacity: Opacity of the bot and sensor rays (0 to 255).
        :param image_size: Size of the robot image in pixels (width, height).
        :param surface: Pygame Surface object. This can be the main display surface (often called screen),
                        or any other surface for off-screen rendering.
        """

        self.wheel_radius = 0.0165  # meters (adjust based on real robot)
        self.robot_mass = 0.55      # kg (adjust based on real robot)
        # self.wheel_inertia = 0.5 * self.robot_mass * (self.wheel_radius ** 2) * 100  # Simplified inertia
        self.wheel_inertia = 0.005  # 0.005 kg*m^2 (adjust based on real robot) 
        self.friction_coeff = 0.75  # Adjust based on real-world testing


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
        self.sensor_coordinates = []
        self.sensor_type = sensor_type

        # for fitness function purposes
        self.off_track_time = 0

        # step count is for the number of steps the bot lasted in the simulation int he smae attempt
        # this is used for logging the data of the bot in each attempt
        self.step_count = 0

        # drawing stuff
        self.img_path = img_path
        self.draw_robot = draw_robot
        self.opacity = opacity
        self.image_size = image_size
        self.sensor_color = (125, 125, 125)
        self.sensor_active_color = (0, 0, 255)

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
        # Convert rad/s to m/s using wheel radius
        left_lin = self.left_wheel_velocity * self.wheel_radius
        right_lin = self.right_wheel_velocity * self.wheel_radius
        
        v = (left_lin + right_lin) / 2.0
        omega = (right_lin - left_lin) / self.robot_config["wheels_spacing"]
        return v, omega

    def _power_to_volts(self, l_pow, r_pow):
        """
        Convert normalized power values [-1, 1] to voltage.
        """
        l_pow = np.clip(l_pow, -1.0, 1.0)
        r_pow = np.clip(r_pow, -1.0, 1.0)
        return l_pow * self.volts, r_pow * self.volts

    def apply_action(self, action, dt):
        """
        Smoothly apply action to the bot.
        Proportional ramping toward target speeds.
        Wheels aiming for higher speeds accelerate faster.
        :param action: Tuple (left_power, right_power), values in [-1, 1]
        :param dt: Time step in seconds
        """
        # action = [0.96,1]
        l_target_volts, r_target_volts = self._power_to_volts(*action)
        max_speed = self.robot_config.get("max_speed")  # in pixel/s

        # Convert to target speeds
        target_left_speed = (l_target_volts / self.volts) * max_speed
        target_right_speed = (r_target_volts / self.volts) * max_speed

        # Base acceleration rate
        base_accel = 2500  # px/s²

        # Proportional to command
        left_accel = base_accel * abs(action[0])
        right_accel = base_accel * abs(action[1])

        max_delta_left = left_accel * dt
        max_delta_right = right_accel * dt

        def approach(current, target, max_delta):
            if abs(target - current) <= max_delta:
                return target
            return current + np.sign(target - current) * max_delta

        self.left_wheel_velocity = approach(self.left_wheel_velocity, target_left_speed, max_delta_left)
        self.right_wheel_velocity = approach(self.right_wheel_velocity, target_right_speed, max_delta_right)



    # this function is no longer used
    def read_semi_circle_sensor(self):
        """
        Get the line sensor readings by finding the pixel color under each sensor
        :return: numpy array of sensor readings.
        """
        readings = np.zeros(self.sensor_count)

        # array of sensors coordinates
        sensors_coordinates = []

        # Evenly distribute sensor rays over the sensor_angle_range.
        angles = np.linspace(-self.sensor_angle_range / 2, self.sensor_angle_range / 2, self.sensor_count)
        for i, sensor_angle in enumerate(angles):
            # Calculate the global angle for each sensor ray.
            ray_angle = self.yaw + sensor_angle
            
            # Compute sensor tip position.
            sensor_x = int(self.position[0] + self.sensor_range * np.cos(ray_angle))
            sensor_y = int(self.position[1] + self.sensor_range * np.sin(ray_angle))
            
            sensors_coordinates.append((sensor_x, sensor_y))

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
        return sensors_coordinates, readings
    
    def jsumo_xline(self):
        """
        https://www.jsumo.com/xline-16-sensor-array-board-digital
        """
        # Not implemented yet
        pass

    # this is the default used line sensor function
    def jsumo_xline_v2(self):
        """
        https://www.jsumo.com/xline-16-line-sensor-board-digital-v2
        :return: numpy array of sensor readings.
        """
        # Constants
        HORIZONTAL_SENSOR_DISTANCE = 9  # mm/pixels
        VERTICAL_SENSOR_DISTANCE = 5    # mm/pixels
        SENSOR_COUNT = 15

        # Build the hardcoded XLINE-style layout
        sensor_points_local = []
        for i in range(SENSOR_COUNT):
            if i < 4:  # Left arm
                x = i* HORIZONTAL_SENSOR_DISTANCE
                y = i * VERTICAL_SENSOR_DISTANCE
            elif i < 10:  # Center
                x = i * HORIZONTAL_SENSOR_DISTANCE
                y = 4*VERTICAL_SENSOR_DISTANCE
            else:  # Right arm
                x = i* HORIZONTAL_SENSOR_DISTANCE
                y = (14-i) * VERTICAL_SENSOR_DISTANCE
            sensor_points_local.append((x-(128//2), y+self.sensor_range)) # 128 is to center the sensor (idk why it is 128 it is supposed to be 137 which is the width of the image but if it works it works)

        # Rotation matrix for robot's yaw
        cos_yaw = np.cos(self.yaw)
        sin_yaw = np.sin(self.yaw)

        # Prepare the reading array
        readings = np.zeros(SENSOR_COUNT)

        sensors_coordinates = []

        for i, (x_local, y_local) in enumerate(sensor_points_local):
            # Rotate 90° anti-clockwise in local frame
            x_90 = y_local
            y_90 = -x_local

            # Then rotate to match robot yaw
            x_rotated = x_90 * cos_yaw - y_90 * sin_yaw
            y_rotated = x_90 * sin_yaw + y_90 * cos_yaw

            # Translate to world coordinates
            sensor_x = int(self.position[0] + x_rotated)
            sensor_y = int(self.position[1] + y_rotated)

            sensors_coordinates.append((sensor_x, sensor_y))

            # Check boundaries
            if 0 <= sensor_x < self.screen.get_width() and 0 <= sensor_y < self.screen.get_height():
                pixel_color = self.screen.get_at((sensor_x, sensor_y))[:3]
                if pixel_color == (255, 255, 255):  # White
                    readings[i] = 0
                    # pygame.draw.circle(self.screen, self.sensor_color, (int(x), int(y)), 2)
                elif pixel_color == (0, 0, 0) or pixel_color == (255, 255, 0):  # Black or Yellow
                    readings[i] = 1
                    # pygame.draw.circle(self.screen, self.sensor_active_color, (int(x), int(y)), 2)

        # Store the result
        self.sensor_readings = readings
        self.sensor_coordinates = sensors_coordinates
        return readings
    
    def get_line_sensor_readings(self):
        """
        Get the line sensor readings based on the specified sensor type.
        :param sensor_type: Type of sensor array ('semi_circle' or 'full_circle').
        :return: numpy array of sensor readings.
        """
        if self.sensor_type.lower() == "semi_circle":
            return self.read_semi_circle_sensor()
        elif self.sensor_type.lower() == "jsumo_xline":
            return self.jsumo_xline()
        elif self.sensor_type.lower() == "jsumo_xline_v2":
            return self.jsumo_xline_v2()
        else:
            raise ValueError(f"Unknown sensor type: {self.sensor_type}")
    
    def check_middle_sensor_color(self):
        """
        Checks the color under the front middle sensor (index 7 for 15 sensors).
        Returns "red", "green", or "yellow" if the middle sensor detects that color.
        Returns None otherwise.
        Assumes self.sensor_coordinates is up-to-date.
        """
        # For a 15-sensor array (indices 0-14), the middle is index 7
        MIDDLE_SENSOR_INDEX = self.sensor_count // 2

        # Check if coordinates are available and list is long enough
        if not hasattr(self, 'sensor_coordinates') or not self.sensor_coordinates or len(self.sensor_coordinates) <= MIDDLE_SENSOR_INDEX:
            # print(f"Warn: Sensor coords not ready/long enough for middle sensor check (Idx: {MIDDLE_SENSOR_INDEX})")
            return None

        try:
            # Get the world coordinates of the middle sensor
            sx, sy = self.sensor_coordinates[MIDDLE_SENSOR_INDEX]
            sx_int, sy_int = int(sx), int(sy)

            # Check bounds before accessing screen pixel
            if 0 <= sx_int < self.screen.get_width() and 0 <= sy_int < self.screen.get_height():
                pixel_color = self.screen.get_at((sx_int, sy_int)) # Get RGBA

                # Compare against RGBA (assuming opaque colors)
                if pixel_color == (255, 0, 0, 255):       # Opaque Red
                    return "red"
                elif pixel_color == (0, 255, 0, 255):     # Opaque Green
                    return "green"
                elif pixel_color == (255, 255, 0, 255):   # Opaque Yellow
                    return "yellow"
                else:
                    return None # Middle sensor on background or normal line
            else:
                return None # Middle sensor is off-screen

        except IndexError:
            print(f"Error: IndexError accessing middle sensor coord (Idx: {MIDDLE_SENSOR_INDEX}).")
            return None
        except Exception as e:
            print(f"Error checking middle sensor color: {e}")
            return None


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
        wheels_spacing = self.robot_config.get("wheels_spacing")
        omega = (self.right_wheel_velocity - self.left_wheel_velocity) / wheels_spacing

        # Update position and orientation.
        self.position[0] += v * np.cos(self.yaw) * dt
        self.position[1] += v * np.sin(self.yaw) * dt
        self.yaw += omega * dt


    def draw(self,
             img_path:str=None,
             draw_robot:bool=True,
             opacity:int=255,
             image_size:tuple=(190, 167),
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
        # if no screen was provided use the default screen originaly defined
        if surface is None:
            surface = self.screen
        if img_path is None:
            img_path = self.img_path

        # sensor circle radius
        SENSOR_DRAW_RADIOUS = 4

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
        # Use sensor layout based on the specified sensor type.
        if self.sensor_type.lower() == "semi_circle":
            # Evenly distribute sensor rays over the sensor_angle_range.
            angles = np.linspace(-self.sensor_angle_range / 2, self.sensor_angle_range / 2, self.sensor_count)
            for i, sensor_angle in enumerate(angles):
                # Calculate the global angle for each sensor ray.
                ray_angle = self.yaw + sensor_angle
                
                # Compute sensor tip position.
                sensor_end_x = self.position[0] + self.sensor_range * np.cos(ray_angle)
                sensor_end_y = self.position[1] + self.sensor_range * np.sin(ray_angle)
                
                if self.sensor_readings[i] == 1:
                    pygame.draw.circle(surface, sensor_active_color, (int(sensor_end_x), int(sensor_end_y)), SENSOR_DRAW_RADIOUS)
                else:
                    pygame.draw.circle(surface, sensor_color, (int(sensor_end_x), int(sensor_end_y)), SENSOR_DRAW_RADIOUS)
        
        elif self.sensor_type.lower() == "jsumo_xline_v2":
            # print(f"self.sensor_coordinates: {self.sensor_coordinates}")
            # print(f"self.sensor_readings: {self.sensor_readings}")
            for i, (x, y) in enumerate(self.sensor_coordinates):
                if self.sensor_readings[i] == 1:
                    pygame.draw.circle(surface, sensor_active_color, (int(x), int(y)), SENSOR_DRAW_RADIOUS)
                else:
                    pygame.draw.circle(surface, sensor_color, (int(x), int(y)), SENSOR_DRAW_RADIOUS)

        else:
            # Fallback: default to semi_circle sensor drawing
            angles = np.linspace(-self.sensor_angle_range / 2, self.sensor_angle_range / 2, self.sensor_count)
            for i, sensor_angle in enumerate(angles):
                # Calculate the global angle for each sensor ray.
                ray_angle = self.yaw + sensor_angle
                
                # Compute sensor tip position.
                sensor_end_x = self.position[0] + self.sensor_range * np.cos(ray_angle)
                sensor_end_y = self.position[1] + self.sensor_range * np.sin(ray_angle)
                
                if self.sensor_readings[i] == 1:
                    pygame.draw.circle(surface, sensor_active_color, (int(sensor_end_x), int(sensor_end_y)), SENSOR_DRAW_RADIOUS)
                else:
                    pygame.draw.circle(surface, sensor_color, (int(sensor_end_x), int(sensor_end_y)), SENSOR_DRAW_RADIOUS)

                

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
                 sensor_type    : str = "jsumo_xline_v2",
                 sensor_count   : int = 15,
                 draw_robot:bool=True,
                 img_path:str=None,
                 opacity:int=255,
                 image_size:tuple=(190, 167)
                 ):
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
        super().__init__(robot_config=robot_config,
                         screen=screen,
                         sensor_type=sensor_type,
                         sensor_count=sensor_count,
                         draw_robot=draw_robot,
                         img_path=img_path,
                         opacity=opacity,
                         image_size=image_size)
        
        # set the neural netwrok of the bot
        self.net = neat.nn.RecurrentNetwork.create(genome, neat_config)

        # this is because the network is recurrent and needs to have a previous output to be able to work
        # the previous output is a list of the outputs of the network in the last step which is the speed of the motors in range between [-1,1]
        self.previous_output = [0, 0] 

    def step(self, dt):
        """
        Update the bot's state over a time step.
        :param dt: Time step in seconds.
        :param track: Track object used for obtaining sensor readings.
        """
        # Update sensor readings.
        self.get_line_sensor_readings()

        # activate the network
        output = self.net.activate([*self.previous_output, *self.sensor_readings])
        self.previous_output = output.copy()
        print(f"previous_output: {self.previous_output}, \nsensor_readings: {self.sensor_readings}\nOutput: {output}")
        # print(f"Output: {output}")


        # modify the motor speed (wheel velocity) based on the network's output
        self.apply_action(output, dt)

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
        super().__init__(config, screen, sensor_type="jsumo_xline_v2")
        
        # PID parameters from config
        self.Kp = config['PID']['Kp']
        self.Ki = config['PID']['Ki']
        self.Kd = config['PID']['Kd']
        
        # PID state
        self.integral = 0.0
        self.prev_error = 0.0
        self.error_history = []

    def compute_pid_action(self, sensor_readings, dt):
        # 1. Calculate digital-style weighted error
        n = len(sensor_readings)
        weights = np.linspace(-(n//2), n//2, n)  # e.g. [-7,...,7] if n=15 or 16
        error_sum = 0.0
        active = 0

        for i, val in enumerate(sensor_readings):
            # print(f"sensor_readings[{i}]: {val}, weights[{i}]: {weights[i]}")
            if not val:         # treat “below threshold” as line detected
                error_sum += weights[i]
                active += 1

        if active == 0:
            # no line → mimic last known error bias ±5
            error = 5.0 if self.prev_error > 0 else -5.0
        else:
            error = error_sum / active      # normalized

        # 2. PID terms
        P = self.Kp * error
        self.integral += self.Ki * error * dt
        D = self.Kd * (error - self.prev_error) / dt
        self.prev_error = error

        # 3 prevent jitter around center
        if abs(error) <= 1.0:
            error = 0.0
            self.integral = 0.0

        # 4. total output
        control = P + self.integral + D

        # 5. differential drive speeds
        base = 0.8
        left  = np.clip(base - control, -1.0, 1.0)
        right = np.clip(base + control, -1.0, 1.0)

        return [left, right]
