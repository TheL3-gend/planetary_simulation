#!/usr/bin/env python3
# diagnostic_main.py - Diagnostic version of main.py to troubleshoot black screen

import sys
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import traceback
import logging
import time
import os

# Configure logging with more detail
logging.basicConfig(
    level=logging.DEBUG,  # More detailed logging
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("DiagnosticSim")

# Import constants but prepare fallbacks
try:
    from constants import *
    # Validate some critical constants
    if not isinstance(WINDOW_WIDTH, int) or WINDOW_WIDTH <= 0:
        logger.warning("Invalid WINDOW_WIDTH, using default")
        WINDOW_WIDTH = 1024
    if not isinstance(WINDOW_HEIGHT, int) or WINDOW_HEIGHT <= 0:
        logger.warning("Invalid WINDOW_HEIGHT, using default")
        WINDOW_HEIGHT = 768
    if not hasattr(globals(), 'SCALE_FACTOR') or not isinstance(SCALE_FACTOR, (int, float)):
        logger.warning("Invalid SCALE_FACTOR, using default")
        SCALE_FACTOR = 1e9
except ImportError:
    logger.error("Failed to import constants, using defaults")
    # Default constants
    WINDOW_WIDTH = 1024
    WINDOW_HEIGHT = 768
    SCALE_FACTOR = 1e9
    ANTI_ALIASING = False
    VSYNC = True
    FULLSCREEN = False
    MAX_FPS = 60
    SHOW_AXES = True

# Try to import components, falling back to minimal versions if needed
try:
    from camera import Camera
    logger.info("Camera module imported")
except ImportError:
    logger.error("Failed to import camera, using minimal version")
    # Minimal camera implementation
    class Camera:
        def __init__(self):
            self.distance = 20.0
            self.x_angle = 0.0
            self.y_angle = 0.3
            self.target = [0.0, 0.0, 0.0]
        
        def setup_view(self):
            glLoadIdentity()
            glTranslatef(0, 0, -self.distance)
            glRotatef(self.y_angle * 180 / 3.14159, 1, 0, 0)
            glRotatef(self.x_angle * 180 / 3.14159, 0, 1, 0)
        
        def handle_mouse_motion(self, pos, rel, buttons):
            if buttons[0]:  # Left button
                self.x_angle += rel[0] * 0.01
                self.y_angle += rel[1] * 0.01
        
        def handle_mouse_wheel(self, y):
            self.distance *= 0.9 if y > 0 else 1.1
            self.distance = max(5.0, min(100.0, self.distance))
        
        def handle_key(self, key):
            pass
        
        def handle_key_up(self, key):
            pass
        
        def update(self, dt, selected_body=None):
            pass

try:
    from simulation import Simulation
    logger.info("Simulation module imported")
except ImportError:
    logger.error("Failed to import simulation, using minimal version")
    # Minimal simulation with a sun and earth
    class Body:
        def __init__(self, name, radius, position, color):
            self.name = name
            self.radius = radius
            self.position = position
            self.color = color
            self.visual_radius = radius / SCALE_FACTOR
    
    class Simulation:
        def __init__(self):
            # Create minimal solar system
            self.bodies = [
                Body("Sun", 695700000, [0, 0, 0], (1.0, 0.7, 0.0)),
                Body("Earth", 6371000, [149.6e9, 0, 0], (0.0, 0.5, 1.0))
            ]
            self.selected_body = self.bodies[0]
            self.show_trails = True
            self.show_labels = True
        
        def update(self):
            pass
        
        def handle_key(self, key):
            if key == K_TAB:
                # Cycle selected body
                current = self.bodies.index(self.selected_body)
                next_index = (current + 1) % len(self.bodies)
                self.selected_body = self.bodies[next_index]

def draw_debug_objects():
    """Draw some simple objects to test if rendering is working"""
    # Draw a colored triangle
    glLoadIdentity()
    glTranslatef(0, 0, -5)  # Move back 5 units
    
    glDisable(GL_LIGHTING)
    glDisable(GL_TEXTURE_2D)
    
    glBegin(GL_TRIANGLES)
    glColor3f(1.0, 0.0, 0.0)  # Red
    glVertex3f(0, 1, 0)
    glColor3f(0.0, 1.0, 0.0)  # Green
    glVertex3f(-1, -1, 0)
    glColor3f(0.0, 0.0, 1.0)  # Blue
    glVertex3f(1, -1, 0)
    glEnd()
    
    # Draw coordinate axes
    glBegin(GL_LINES)
    # X axis (red)
    glColor3f(1, 0, 0)
    glVertex3f(0, 0, 0)
    glVertex3f(2, 0, 0)
    # Y axis (green)
    glColor3f(0, 1, 0)
    glVertex3f(0, 0, 0)
    glVertex3f(0, 2, 0)
    # Z axis (blue)
    glColor3f(0, 0, 1)
    glVertex3f(0, 0, 0)
    glVertex3f(0, 0, 2)
    glEnd()
    
    # Draw a sphere using GLU quadric
    glTranslatef(3, 0, 0)  # Move to the right
    sphere = gluNewQuadric()
    glColor3f(1.0, 1.0, 0.0)  # Yellow
    gluSphere(sphere, 1.0, 32, 16)
    gluDeleteQuadric(sphere)

def main():
    """Main function with additional diagnostics"""
    logger.info("Starting diagnostic simulation")
    
    # Initialize pygame
    if not pygame.get_init():
        pygame.init()
    if not pygame.display.get_init():
        pygame.display.init()
    
    logger.info(f"Pygame initialized: {pygame.get_init()}")
    
    # Set up display
    pygame.display.set_caption("Diagnostic Simulation")
    
    # Try to create window with OpenGL
    try:
        screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), DOUBLEBUF | OPENGL)
        logger.info(f"Display created: {WINDOW_WIDTH}x{WINDOW_HEIGHT}")
    except pygame.error as e:
        logger.error(f"Failed to create display: {e}")
        try:
            # Try with minimal settings
            screen = pygame.display.set_mode((800, 600), DOUBLEBUF | OPENGL)
            logger.info("Created display with minimal settings")
        except:
            logger.critical("Failed to create any display")
            return 1
    
    # Print OpenGL information
    try:
        version = glGetString(GL_VERSION).decode('utf-8')
        vendor = glGetString(GL_VENDOR).decode('utf-8')
        renderer = glGetString(GL_RENDERER).decode('utf-8')
        
        logger.info(f"OpenGL Version: {version}")
        logger.info(f"OpenGL Vendor: {vendor}")
        logger.info(f"OpenGL Renderer: {renderer}")
    except:
        logger.error("Failed to get OpenGL info")
    
    # Set up basic OpenGL state
    glClearColor(0.0, 0.0, 0.1, 1.0)  # Dark blue background
    glEnable(GL_DEPTH_TEST)
    
    # Set up projection
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, WINDOW_WIDTH / WINDOW_HEIGHT, 0.1, 1000.0)
    logger.info("Projection matrix set")
    
    # Switch to modelview matrix
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    logger.info("Modelview matrix set")
    
    # Create camera and simulation
    camera = Camera()
    simulation = Simulation()
    logger.info(f"Created camera and simulation with {len(simulation.bodies)} bodies")
    
    # Set up clock
    clock = pygame.time.Clock()
    
    # Render mode - try different approaches to debug
    # 0: Regular planet rendering
    # 1: Debug objects only
    # 2: Both
    render_mode = 0
    
    # Main loop
    running = True
    frame_count = 0
    last_time = time.time()
    
    logger.info("Entering main loop")
    while running:
        # Track frame rate
        current_time = time.time()
        frame_count += 1
        if current_time - last_time >= 1.0:
            fps = frame_count / (current_time - last_time)
            pygame.display.set_caption(f"Diagnostic Simulation - FPS: {fps:.1f}")
            frame_count = 0
            last_time = current_time
            logger.info(f"FPS: {fps:.1f}")
        
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == K_ESCAPE:
                    running = False
                elif event.key == K_1:
                    # Toggle render mode
                    render_mode = (render_mode + 1) % 3
                    logger.info(f"Render mode: {render_mode}")
                elif event.key == K_r:
                    # Reset camera
                    camera = Camera()
                    logger.info("Camera reset")
                else:
                    simulation.handle_key(event.key)
                    camera.handle_key(event.key)
            elif event.type == pygame.KEYUP:
                camera.handle_key_up(event.key)
            elif event.type == pygame.MOUSEMOTION:
                camera.handle_mouse_motion(event.pos, event.rel, pygame.mouse.get_pressed())
            elif event.type == pygame.MOUSEWHEEL:
                camera.handle_mouse_wheel(event.y)
        
        # Update simulation
        simulation.update()
        
        # Update camera
        camera.update(1.0/MAX_FPS, simulation.selected_body)
        
        # Clear the screen
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Set up camera view
        glLoadIdentity()
        camera.setup_view()
        
        # Draw based on render mode
        if render_mode in (0, 2):
            # Draw planets
            for body in simulation.bodies:
                glPushMatrix()
                
                # Position
                glTranslatef(
                    body.position[0] / SCALE_FACTOR,
                    body.position[1] / SCALE_FACTOR,
                    body.position[2] / SCALE_FACTOR
                )
                
                # Draw body
                glColor3f(body.color[0], body.color[1], body.color[2])
                sphere = gluNewQuadric()
                gluSphere(sphere, body.visual_radius, 32, 16)
                gluDeleteQuadric(sphere)
                
                glPopMatrix()
                
            logger.debug(f"Drew {len(simulation.bodies)} bodies")
        
        if render_mode in (1, 2):
            # Draw debug objects
            draw_debug_objects()
            logger.debug("Drew debug objects")
        
        # Draw coordinate system at origin
        if SHOW_AXES:
            glLoadIdentity()
            glTranslatef(0, 0, -20)
            
            glBegin(GL_LINES)
            # X axis (red)
            glColor3f(1, 0, 0)
            glVertex3f(0, 0, 0)
            glVertex3f(5, 0, 0)
            # Y axis (green)
            glColor3f(0, 1, 0)
            glVertex3f(0, 0, 0)
            glVertex3f(0, 5, 0)
            # Z axis (blue)
            glColor3f(0, 0, 1)
            glVertex3f(0, 0, 0)
            glVertex3f(0, 0, 5)
            glEnd()
        
        # Check for OpenGL errors
        error = glGetError()
        if error != GL_NO_ERROR:
            logger.error(f"OpenGL error: {error}")
        
        # Swap buffers
        pygame.display.flip()
        
        # Cap frame rate
        clock.tick(MAX_FPS)
    
    # Clean up and exit
    pygame.quit()
    logger.info("Simulation ended")
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        traceback.print_exc()
        try:
            pygame.quit()
        except:
            pass
        sys.exit(1)