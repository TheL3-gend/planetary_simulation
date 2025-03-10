#!/usr/bin/env python3
# check_renderer.py - Isolated test to check if renderer is working

import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import traceback
import sys
import time
import numpy as np
import logging
import os

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
logger = logging.getLogger("RendererTest")

# Constants (replace with imports if needed)
WINDOW_WIDTH = 1024
WINDOW_HEIGHT = 768
MAX_FPS = 60
TEXTURE_DIR = "textures"
SCALE_FACTOR = 1e9

# Create directory for textures if needed
os.makedirs(TEXTURE_DIR, exist_ok=True)

class Body:
    """Minimal celestial body implementation"""
    def __init__(self, name, radius, position, color):
        self.name = name
        self.radius = radius  # in meters
        self.position = np.array(position, dtype=np.float64)
        self.color = color
        self.visual_radius = max(0.1, radius / 1.5e8)  # Scaled for visualization
        self.texture_name = f"{name.lower()}.jpg"
        self.texture_id = None
        self.has_rings = name in ["Saturn", "Jupiter", "Uranus", "Neptune"]

class TextureManager:
    """Minimal texture manager for testing"""
    def __init__(self):
        self.textures = {}
        self.default_texture = self.create_checkerboard()
    
    def create_checkerboard(self):
        """Create a simple checkerboard texture for testing"""
        try:
            texture_id = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, texture_id)
            
            # Create a checkerboard pattern
            size = 64
            data = []
            for i in range(size):
                for j in range(size):
                    if (i // 8 + j // 8) % 2:
                        data.extend([255, 255, 255, 255])  # White
                    else:
                        data.extend([128, 128, 128, 255])  # Gray
            
            data = np.array(data, dtype=np.uint8)
            
            # Set texture parameters
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
            
            # Upload texture data
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, size, size, 0, GL_RGBA, GL_UNSIGNED_BYTE, data)
            
            return texture_id
        except Exception as e:
            logger.error(f"Error creating checkerboard texture: {e}")
            return 0
    
    def load_texture(self, filename):
        """Load texture or return default if not found"""
        if filename in self.textures:
            return self.textures[filename]
        
        # Check if file exists
        filepath = os.path.join(TEXTURE_DIR, filename)
        if os.path.exists(filepath):
            try:
                # Load image using pygame
                image = pygame.image.load(filepath)
                
                # Convert to RGBA
                if image.get_bytesize() == 3:  # RGB format
                    image = image.convert_alpha()
                
                # Get image data
                data = pygame.image.tostring(image, "RGBA", 1)
                width, height = image.get_size()
                
                # Create texture
                texture_id = glGenTextures(1)
                glBindTexture(GL_TEXTURE_2D, texture_id)
                
                # Set parameters
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
                
                # Upload data
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data)
                
                # Store and return
                self.textures[filename] = texture_id
                logger.info(f"Loaded texture: {filename}")
                return texture_id
            except Exception as e:
                logger.error(f"Error loading texture {filename}: {e}")
        
        # Use default
        logger.warning(f"Texture {filename} not found, using default")
        self.textures[filename] = self.default_texture
        return self.default_texture
    
    def cleanup(self):
        """Clean up textures"""
        for texture_id in self.textures.values():
            if texture_id:
                glDeleteTextures(1, [texture_id])
        
        self.textures.clear()

class Camera:
    """Simple camera implementation for testing"""
    def __init__(self):
        self.distance = 20.0
        self.x_angle = 0.0
        self.y_angle = 0.3
        self.target = np.array([0.0, 0.0, 0.0])
    
    def setup_view(self):
        """Set up camera view"""
        glLoadIdentity()
        
        # Calculate camera position
        eye_x = self.target[0] + self.distance * np.sin(self.x_angle) * np.cos(self.y_angle)
        eye_y = self.target[1] + self.distance * np.sin(self.y_angle)
        eye_z = self.target[2] + self.distance * np.cos(self.x_angle) * np.cos(self.y_angle)
        
        # Use gluLookAt for camera positioning
        gluLookAt(
            eye_x, eye_y, eye_z,
            self.target[0], self.target[1], self.target[2],
            0, 1, 0
        )
    
    def handle_mouse_motion(self, rel, buttons):
        """Handle mouse motion for camera control"""
        if buttons[0]:  # Left button
            dx, dy = rel
            self.x_angle -= dx * 0.01
            self.y_angle -= dy * 0.01
            self.y_angle = np.clip(self.y_angle, -np.pi/2 + 0.1, np.pi/2 - 0.1)
    
    def handle_mouse_wheel(self, y):
        """Handle mouse wheel for zooming"""
        self.distance *= 0.9 if y > 0 else 1.1
        self.distance = max(5.0, min(100.0, self.distance))

class SimpleRenderer:
    """Simple renderer for testing"""
    def __init__(self):
        self.texture_manager = TextureManager()
        self.sphere_quality = 32  # Detail level for spheres
        logger.info("Renderer initialized")
    
    def setup_opengl(self):
        """Set up OpenGL state"""
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)
        glClearColor(0.0, 0.0, 0.05, 1.0)
        
        # Enable lighting (optional, for testing)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        
        # Set light position (at origin, like the sun)
        glLightfv(GL_LIGHT0, GL_POSITION, [0.0, 0.0, 0.0, 1.0])
        
        logger.info("OpenGL state set up")
    
    def draw_body(self, body):
        """Draw a celestial body"""
        # Skip if position is invalid
        if not np.all(np.isfinite(body.position)):
            logger.warning(f"Invalid position for {body.name}")
            return
        
        glPushMatrix()
        
        # Apply translation
        scaled_pos = body.position / SCALE_FACTOR
        glTranslatef(scaled_pos[0], scaled_pos[1], scaled_pos[2])
        
        # Set color
        glColor3f(body.color[0], body.color[1], body.color[2])
        
        # Create and draw sphere
        quadric = gluNewQuadric()
        
        # Enable texture if available
        if body.texture_id:
            gluQuadricTexture(quadric, GL_TRUE)
            glBindTexture(GL_TEXTURE_2D, body.texture_id)
            glEnable(GL_TEXTURE_2D)
        
        # Draw sphere
        gluSphere(quadric, body.visual_radius, self.sphere_quality, self.sphere_quality//2)
        gluDeleteQuadric(quadric)
        
        # Disable texture
        if body.texture_id:
            glDisable(GL_TEXTURE_2D)
        
        # Draw rings if needed
        if body.has_rings:
            self.draw_rings(body)
        
        glPopMatrix()
    
    def draw_rings(self, body):
        """Draw rings for a planet"""
        # Ring dimensions
        inner_radius = body.visual_radius * 1.2
        outer_radius = body.visual_radius * 2.0
        
        glPushMatrix()
        
        # Rings lie flat on the xz plane
        glRotatef(90, 1, 0, 0)  # Rotate to horizontal
        
        # Draw rings using a simple quadric disk
        quadric = gluNewQuadric()
        glColor4f(body.color[0], body.color[1], body.color[2], 0.7)  # Semi-transparent
        gluDisk(quadric, inner_radius, outer_radius, self.sphere_quality, 4)
        gluDeleteQuadric(quadric)
        
        glPopMatrix()
    
    def draw_coordinate_axes(self):
        """Draw coordinate axes for reference"""
        glDisable(GL_LIGHTING)
        glDisable(GL_TEXTURE_2D)
        
        glBegin(GL_LINES)
        # X axis (red)
        glColor3f(1, 0, 0)
        glVertex3f(0, 0, 0)
        glVertex3f(10, 0, 0)
        # Y axis (green)
        glColor3f(0, 1, 0)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 10, 0)
        # Z axis (blue)
        glColor3f(0, 0, 1)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 0, 10)
        glEnd()
        
        glEnable(GL_LIGHTING)
    
    def render(self, bodies, camera):
        """Render all bodies"""
        # Set up camera view
        camera.setup_view()
        
        # Draw coordinate axes
        self.draw_coordinate_axes()
        
        # Check if bodies list is valid
        if not bodies:
            logger.warning("No bodies to render")
            return
        
        # Draw each body
        for body in bodies:
            self.draw_body(body)
    
    def cleanup(self):
        """Clean up resources"""
        self.texture_manager.cleanup()
        logger.info("Renderer cleaned up")

def main():
    """Main function"""
    # Initialize pygame
    pygame.init()
    
    # Set up display
    pygame.display.set_caption("Renderer Test")
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), DOUBLEBUF | OPENGL)
    
    # Print OpenGL information
    print(f"OpenGL Version: {glGetString(GL_VERSION).decode()}")
    print(f"OpenGL Renderer: {glGetString(GL_RENDERER).decode()}")
    print(f"OpenGL Vendor: {glGetString(GL_VENDOR).decode()}")
    
    # Set up projection
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, WINDOW_WIDTH / WINDOW_HEIGHT, 0.1, 1000.0)
    glMatrixMode(GL_MODELVIEW)
    
    # Create camera
    camera = Camera()
    
    # Create renderer
    renderer = SimpleRenderer()
    renderer.setup_opengl()
    
    # Create some bodies for testing
    bodies = [
        Body("Sun", 695700000, [0, 0, 0], (1.0, 0.7, 0.2)),
        Body("Earth", 6371000, [149.6e9, 0, 0], (0.2, 0.4, 0.8)),
        Body("Mars", 3389500, [227.9e9, 0, 0], (0.8, 0.3, 0.2)),
        Body("Saturn", 58232000, [1433.5e9, 0, 0], (0.9, 0.8, 0.5))
    ]
    
    # Load textures
    for body in bodies:
        body.texture_id = renderer.texture_manager.load_texture(body.texture_name)
    
    # Set initial camera target to Earth
    camera.target = bodies[1].position / SCALE_FACTOR
    
    # Set up clock
    clock = pygame.time.Clock()
    
    # Main loop
    running = True
    frame_count = 0
    start_time = time.time()
    current_body_index = 0
    
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == K_ESCAPE:
                    running = False
                elif event.key == K_TAB:
                    # Cycle through bodies
                    current_body_index = (current_body_index + 1) % len(bodies)
                    camera.target = bodies[current_body_index].position / SCALE_FACTOR
                    print(f"Focused on {bodies[current_body_index].name}")
                elif event.key == K_r:
                    # Reset camera
                    camera = Camera()
                    camera.target = bodies[current_body_index].position / SCALE_FACTOR
            elif event.type == pygame.MOUSEMOTION:
                camera.handle_mouse_motion(event.rel, pygame.mouse.get_pressed())
            elif event.type == pygame.MOUSEWHEEL:
                camera.handle_mouse_wheel(event.y)
        
        # Clear screen
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Render scene
        renderer.render(bodies, camera)
        
        # Check for errors
        error = glGetError()
        if error != GL_NO_ERROR:
            print(f"OpenGL error: {error}")
        
        # Update display
        pygame.display.flip()
        
        # Calculate FPS
        frame_count += 1
        current_time = time.time()
        if current_time - start_time >= 1.0:
            fps = frame_count / (current_time - start_time)
            pygame.display.set_caption(f"Renderer Test - FPS: {fps:.1f}")
            frame_count = 0
            start_time = current_time
        
        # Cap FPS
        clock.tick(MAX_FPS)
    
    # Clean up
    renderer.cleanup()
    pygame.quit()
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        pygame.quit()
        sys.exit(1)