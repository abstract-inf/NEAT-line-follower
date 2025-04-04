import pygame
import os
import json


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
    
class Viewport:
    def __init__(self, screen_width, screen_height, DISPLAY_TRAINING_WINDOW=True):
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
        
    def handle_viewport_controls(self): 
        keys = pygame.key.get_pressed() 
        pan_speed = 30 / self.zoom 
        
        if keys[pygame.K_LEFT]: 
            self.offset.x = max(0, self.offset.x - pan_speed) 
        if keys[pygame.K_RIGHT]: 
            max_x = max(0, self.world_size.x - (self.screen.get_width()/self.zoom)) 
            self.offset.x = min(max_x, self.offset.x + pan_speed) 
        if keys[pygame.K_UP]: 
            self.offset.y = max(0, self.offset.y - pan_speed) 
        if keys[pygame.K_DOWN]: 
            max_y = max(0, self.world_size.y - (self.screen.get_height()/self.zoom)) 
            self.offset.y = min(max_y, self.offset.y + pan_speed) 