import pygame
import os
import json

class track:
    """
    Class to represent the track for the robot.
    It handles the loading of the track image and the drawing of the track on the screen.
    """
    def __init__(self, screen, img_path):
        self.screen = screen
        self.img_path = img_path
        self.track_img = pygame.image.load(img_path).convert_alpha()
        self.track_rect = self.track_img.get_rect()
        self.track_rect.topleft = (0, 0)  # Set position to top-left corner

    def draw(self):
        """
        Draws the track on the screen.
        """
        self.screen.blit(self.track_img, self.track_rect)


class VirtualTrack:
    def __init__(self, image_path):
        self.full_image = pygame.image.load(image_path).convert_alpha()
        self.width = self.full_image.get_width()
        self.height = self.full_image.get_height()
        
        # Create working surface
        self.surface = pygame.Surface((self.width, self.height)).convert_alpha()
        self.surface.blit(self.full_image, (0, 0))
        
        # Store pixel data correctly
        with pygame.PixelArray(self.full_image.copy()) as px_array:
            self.pixel_data = px_array.make_surface()  # Convert to surface

    def get_pixel(self, x, y):
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.pixel_data.get_at((x, y))[:3]
        return (0, 0, 0)