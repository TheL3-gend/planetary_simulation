#!/usr/bin/env python3
# simplified_main.py - Simplified main program that uses direct rendering

import sys
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import traceback
import logging
import time
import os
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("SimplifiedMain")

# Import necessary modules
try:
    from constants import *
    from camera import Camera
    from simulation import Simulation
    from ui import UI
    
    # Verify some critical constants
    if not isinstance(WINDOW_WIDTH, int) or WINDOW_WIDTH <= 0:
        logger.warning("Invalid WINDOW_WIDTH, using default")
        WINDOW_WIDTH = 1024
    if not isinstance(WINDOW_HEIGHT, int) or WINDOW_HEIGHT <= 0:
        logger.warning("Invalid WINDOW_HEIGHT, using default")
        WINDOW_HEIGHT = 768
    
    logger.info("Successfully imported all modules")
except ImportError as e:
    logger.error(f"Failed to import module: {e}")
    logger.error("Make sure all required modules are available")
    sys.exit(1)

class SimpleRenderer:
    """A simplified renderer that uses direct OpenGL calls without complex shaders"""
    
    def __init__(self, simulation, camera):
        """Initialize the renderer"""
        self.simulation = simulation
        self.camera = camera
        
        # Create texture directory if needed
        if not os.path.exists(TEXTURE_DIR):
            os.makedirs(TEXTURE_DIR)
            
        # Dictionary for textures
        self.textures = {}
        
        # Create default texture
        self.default_texture = self.create_default_texture()
        
        # Load textures for all bodies
        self.load_textures()
        
        # Set sphere detail level
        self.sphere_detail = 32
        
        logger.info("SimpleRenderer initialized")
    
    def create_default_texture(self):
        """Create a default checkerboard texture"""
        try:
            texture_id = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, texture_id)
            
            # Create checkerboard pattern
            size = 64
            data = []
            for i in range(size):
                for j in range(size):
                    if (i // 8 + j // 8) % 2:
                        data.extend([255, 255, 255, 255])  # White
                    else:
                        data.extend([128, 128, 128, 255])  # Gray
            
            # Convert to uint8 array
            data = np.array(data, dtype=np.uint8)
            
            # Set texture parameters
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
            
            # Upload texture data
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, size, size, 0, GL_RGBA, GL_UNSIGNED_BYTE, data)
            
            logger.info("Created default texture")
            return texture_id
        except Exception as e:
            logger.error(f"Error creating default texture: {e}")
            return 0
    
    def load_texture(self, filename):
        """Load a texture from file"""
        # Return cached texture if available
        if filename in self.textures:
            return self.textures[filename]
        
        try:
            # Build full path
            filepath = os.path.join(TEXTURE_DIR, filename)
            
            # Check if file exists
            if not os.path.exists(filepath):
                logger.warning(f"Texture not found: {filepath}")
                self.textures[filename] = self.default_texture
                return self.default_texture
            
            # Load with pygame
            image = pygame.image.load(filepath)
            
            # Convert to RGBA if needed
            if image.get_bytesize() == 3:  # RGB
                image = image.convert_alpha()
            
            # Get image data and dimensions
            image_data = pygame.image.tostring(image, "RGBA", 1)
            width, height = image.get_size()
            
            # Create texture
            texture_id = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, texture_id)
            
            # Set parameters
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
            
            # Upload texture data
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, image_data)
            
            # Store and return
            self.textures[filename] = texture_id
            logger.info(f"Loaded texture: {filename} ({width}x{height})")
            return texture_id
        except Exception as e:
            logger.error(f"Error loading texture {filename}: {e}")
            self.textures[filename] = self.default_texture
            return self.default_texture
    
    def load_textures(self):
        """Load textures for all bodies"""
        for body in self.simulation.bodies:
            if hasattr(body, 'texture_name') and body.texture_name:
                texture_id = self.load_texture(body.texture_name)
                # Store directly on the body
                body.texture_id = texture_id
    
    def setup_opengl(self):
        """Set up OpenGL state"""
        # Enable depth testing
        glEnable(GL_DEPTH_TEST)
        
        # Enable blending for transparency (rings, etc.)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # Enable backface culling
        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)
        
        # Set clear color (dark space)
        glClearColor(0.0, 0.0, 0.05, 1.0)
        
        # Set up lighting
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        
        # Set light at origin (sun position)
        glLightfv(GL_LIGHT0, GL_POSITION, [0.0, 0.0, 0.0, 1.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [1.0, 1.0, 1.0, 1.0])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.2, 0.2, 0.2, 1.0])
        
        logger.info("OpenGL state set up")
    
    def draw_body(self, body):
        """Draw a single celestial body"""
        # Skip if position is invalid
        if not np.all(np.isfinite(body.position)):
            return
            
        glPushMatrix()
        
        # Position body
        scaled_pos = body.position / SCALE_FACTOR
        glTranslatef(float(scaled_pos[0]), float(scaled_pos[1]), float(scaled_pos[2]))
        
        # Set color
        glColor3f(body.color[0], body.color[1], body.color[2])
        
        # Use texture if available
        texture_id = getattr(body, 'texture_id', None)
        if texture_id:
            glEnable(GL_TEXTURE_2D)
            glBindTexture(GL_TEXTURE_2D, texture_id)
        
        # Create and draw sphere
        quadric = gluNewQuadric()
        gluQuadricTexture(quadric, GL_TRUE)
        gluQuadricNormals(quadric, GLU_SMOOTH)
        
        # Scale radius (use visual_radius if available, otherwise compute)
        if hasattr(body, 'visual_radius'):
            radius = body.visual_radius
        else:
            # Fallback: compute from physical radius with logarithmic scaling
            radius = max(0.1, np.log10(body.radius / 1000) / 2)
        
        # Draw sphere
        gluSphere(quadric, radius, self.sphere_detail, self.sphere_detail // 2)
        gluDeleteQuadric(quadric)
        
        # Disable texture
        if texture_id:
            glDisable(GL_TEXTURE_2D)
        
        # Draw rings if body has them
        if hasattr(body, 'has_rings') and body.has_rings:
            self.draw_rings(body)
        
        glPopMatrix()
    
    def draw_rings(self, body):
        """Draw rings around a planet"""
        # Set ring dimensions
        inner_radius = body.visual_radius * 1.2
        outer_radius = body.visual_radius * 2.0
        
        # Use correct ring size if available
        if hasattr(body, 'ring_inner_radius') and hasattr(body, 'ring_outer_radius'):
            inner_radius = body.ring_inner_radius * body.visual_radius
            outer_radius = body.ring_outer_radius * body.visual_radius
        
        glPushMatrix()
        
        # Rings lie flat on the xz plane
        glRotatef(90, 1, 0, 0)
        
        # Semi-transparent ring color
        r, g, b = body.color
        glColor4f(r, g, b, 0.7)
        
        # Draw ring using quadric disk
        quadric = gluNewQuadric()
        gluQuadricTexture(quadric, GL_TRUE)
        gluDisk(quadric, inner_radius, outer_radius, self.sphere_detail, 4)
        gluDeleteQuadric(quadric)
        
        glPopMatrix()
    
    def draw_orbit_trails(self):
        """Draw orbital trails"""
        if not hasattr(self.simulation, 'show_trails') or not self.simulation.show_trails:
            return
            
        glDisable(GL_LIGHTING)
        glDisable(GL_TEXTURE_2D)
        
        for body in self.simulation.bodies:
            if not hasattr(body, 'trail') or not body.trail:
                continue
                
            # Use body color but slightly fainter
            r, g, b = body.color
            glColor4f(r, g, b, 0.5)
            
            # Draw trail as line strip
            glBegin(GL_LINE_STRIP)
            for point in body.trail:
                scaled_point = point / SCALE_FACTOR
                glVertex3f(
                    float(scaled_point[0]),
                    float(scaled_point[1]),
                    float(scaled_point[2])
                )
            glEnd()
        
        glEnable(GL_LIGHTING)
    
    def draw_axes(self):
        """Draw coordinate axes for reference"""
        if not SHOW_AXES:
            return
            
        glDisable(GL_LIGHTING)
        glDisable(GL_TEXTURE_2D)
        
        # Draw coordinate axes at origin
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
        
        glEnable(GL_LIGHTING)
    
    def render(self):
        """Render the scene"""
        # Set up camera view
        self.camera.setup_view()
        
        # Draw coordinate axes
        self.draw_axes()
        
        # Draw orbital trails
        self.draw_orbit_trails()
        
        # Draw all bodies
        for body in self.simulation.bodies:
            self.draw_body(body)
    
    def cleanup(self):
        """Clean up resources"""
        # Delete all textures
        for texture_id in self.textures.values():
            if texture_id:
                glDeleteTextures(1, [texture_id])
        
        # Delete default texture
        if self.default_texture:
            glDeleteTextures(1, [self.default_texture])
        
        self.textures.clear()
        logger.info("Renderer cleaned up")

def main():
    """Main function"""
    components = {}  # Store components for cleanup
    
    try:
        logger.info("Starting simplified main")
        
        # Initialize pygame
        if not pygame.get_init():
            pygame.init()
        
        if not pygame.display.get_init():
            pygame.display.init()
        
        # Set up display
        pygame.display.set_caption("Simplified Planetary Simulation")
        
        # Use a compatibility profile to ensure it works on more systems
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_COMPATIBILITY)
        
        # Create the window
        screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), DOUBLEBUF | OPENGL)
        
        # Print OpenGL info
        logger.info(f"OpenGL Version: {glGetString(GL_VERSION).decode()}")
        logger.info(f"OpenGL Renderer: {glGetString(GL_RENDERER).decode()}")
        
        # Set up projection
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, WINDOW_WIDTH / WINDOW_HEIGHT, 0.1, 1000.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        # Initialize components
        logger.info("Creating camera")
        components["camera"] = Camera()
        
        logger.info("Creating simulation")
        components["simulation"] = Simulation()
        
        logger.info("Creating renderer")
        components["renderer"] = SimpleRenderer(components["simulation"], components["camera"])
        components["renderer"].setup_opengl()
        
        logger.info("Creating UI")
        components["ui"] = UI(components["simulation"])
        
        # Set up clock
        clock = pygame.time.Clock()
        
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
                pygame.display.set_caption(f"Simplified Planetary Simulation - FPS: {fps:.1f}")
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
                    else:
                        components["simulation"].handle_key(event.key)
                        components["camera"].handle_key(event.key)
                elif event.type == pygame.KEYUP:
                    components["camera"].handle_key_up(event.key)
                elif event.type == pygame.MOUSEMOTION:
                    components["camera"].handle_mouse_motion(event.pos, event.rel, pygame.mouse.get_pressed())
                elif event.type == pygame.MOUSEWHEEL:
                    components["camera"].handle_mouse_wheel(event.y)
            
            # Update simulation
            components["simulation"].update()
            
            # Update camera
            components["camera"].update(1.0/MAX_FPS, components["simulation"].selected_body)
            
            # Clear screen
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            
            # Render scene
            components["renderer"].render()
            
            # Render UI
            components["ui"].render(screen)
            
            # Check for errors
            error = glGetError()
            if error != GL_NO_ERROR:
                logger.error(f"OpenGL error: {error}")
            
            # Swap buffers
            pygame.display.flip()
            
            # Cap frame rate
            clock.tick(MAX_FPS)
        
        # Clean up
        logger.info("Cleaning up")
        if "renderer" in components:
            components["renderer"].cleanup()
        
        pygame.quit()
        logger.info("Application exited normally")
        return 0
    
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        traceback.print_exc()
        
        # Clean up
        try:
            if "renderer" in components:
                components["renderer"].cleanup()
            pygame.quit()
        except:
            pass
            
        return 1

if __name__ == "__main__":
    sys.exit(main())