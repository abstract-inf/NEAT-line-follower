# this code is a test on how to build a line follower simulation using pygame

import pygame
import math

pygame.init()

# Set up the drawing window
WIDTH, HEIGHT = 1000, 900
screen = pygame.display.set_mode([WIDTH, HEIGHT])
pygame.display.set_caption("Steering Fix - Moving in Rotation Direction")

# load line path image
line_path = pygame.image.load("line_path.png")
line_path = pygame.transform.scale(line_path, (WIDTH, HEIGHT))

# Load/Create a rectangle surface
rect_width, rect_height = 50, 20
rectangle = pygame.Surface((rect_width, rect_height), pygame.SRCALPHA)
rectangle.fill((255, 115, 115))

# Rectangle properties
rect_x, rect_y = 882,682
rect_angle = 90  # Facing upwards initially
SPEED = 2  # Movement speed
ROTATION_SPEED = 2  # Rotation speed

# Run until the user asks to quit
running = True
clock = pygame.time.Clock()

while running:
    # Frame rate 60fps
    clock.tick(60)

    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Quit game when pressing ESC
    keys = pygame.key.get_pressed()
    if keys[pygame.K_ESCAPE]:
        running = False

    # Rotation (left/right)
    if keys[pygame.K_q] or keys[pygame.K_LEFT]:  # Turn left
        rect_angle = (rect_angle + ROTATION_SPEED) % 360  
    if keys[pygame.K_e] or keys[pygame.K_RIGHT]:  # Turn right
        rect_angle = (rect_angle - ROTATION_SPEED) % 360  

    # Movement (forward/backward in direction of rotation)
    angle_rad = math.radians(rect_angle)
    if keys[pygame.K_w] or keys[pygame.K_UP]:  # Move forward
        rect_x += SPEED * math.cos(angle_rad)
        rect_y -= SPEED * math.sin(angle_rad)
    if keys[pygame.K_s] or keys[pygame.K_DOWN]:  # Move backward
        rect_x -= SPEED * math.cos(angle_rad)
        rect_y += SPEED * math.sin(angle_rad)

    # Fill the background with white
    screen.fill((255, 255, 255))

    # Rotate the rectangle and update its position
    rotated_surface = pygame.transform.rotate(rectangle, rect_angle)
    rotated_rect = rotated_surface.get_rect(center=(rect_x, rect_y))

    # Draw the line path
    screen.blit(line_path, (0, 0))

    # Draw the rotated rectangle
    screen.blit(rotated_surface, rotated_rect.topleft)

    # --- Draw the Front Circles ---
    num_circles = 15
    front_circle_length = 50  # Length of the arrow
    spread_angle = 135  # Spread angle of the circles
    # Calculate the spacing so the circles are evenly distributed over a 90° arc
    spacing = spread_angle / (num_circles - 1)
    for i in range(num_circles):
        # Calculate offset from the center (forward direction)
        offset = (i - (num_circles - 1) / 2) * spacing
        circle_angle = angle_rad + math.radians(offset)
        front_circle_x = int(rect_x + front_circle_length * math.cos(circle_angle))
        front_circle_y = int(rect_y - front_circle_length * math.sin(circle_angle))

        print(f"Sensor {i}: {0 if screen.get_at((front_circle_x, front_circle_y))[:3] == (255, 255, 255) else 1}", end=", ") # print the color under the circle

        if screen.get_at((front_circle_x, front_circle_y))[:3] == (255, 255, 255):  # if the color is white
            pygame.draw.circle(screen, (125,125,125), (front_circle_x, front_circle_y), 5)

        elif screen.get_at((front_circle_x, front_circle_y))[:3] == (0, 0, 0): # the color is black
            pygame.draw.circle(screen, (255, 0, 0), (front_circle_x, front_circle_y), 5)

    print()
    print("-"*50)

    # Update the display
    pygame.display.flip()

pygame.quit()
