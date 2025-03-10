#!/usr/bin/env python3
# main.py - Main entry point for gravity simulation (FIXED VERSION)

import sys
import pygame
from pygame.locals import *
from OpenGL.GL import *
import traceback
import logging
import time
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("GravitySim")


# Import our modules
from constants import *
from camera import Camera
from simulation import Simulation
from renderer import Renderer
from ui import UI

# Import debug utilities or create simplified versions if missing
try:
    import debug_utils
    has_debug_utils = True
except (ImportError, AttributeError):
    has_debug_utils = False
    # Create minimal debug functions to prevent errors
    class MinimalDebugUtils:
        @staticmethod
        def initialize_debug():
            print("Debug system initialized (minimal)")
            
        @staticmethod
        def debug_print(message, level="INFO"):
            print(f"[DEBUG] {message}")
            
        @staticmethod
        def mark_gl_initialized():
            pass
            
        @staticmethod
        def print_gl_info():
            try:
                version = glGetString(GL_VERSION).decode('utf-8')
                vendor = glGetString(GL_VENDOR).decode('utf-8')
                renderer = glGetString(GL_RENDERER).decode('utf-8')
                print(f"OpenGL Version: {version}")
                print(f"OpenGL Vendor: {vendor}")
                print(f"OpenGL Renderer: {renderer}")
            except:
                print("Could not get OpenGL info")
                
        @staticmethod
        def check_gl_errors(where=""):
            error = glGetError()
            if error != GL_NO_ERROR:
                print(f"OpenGL error at {where}: {error}")
                return True
            return False
    
    # Use minimal debug utils as a fallback
    debug_utils = MinimalDebugUtils()
    print("Using minimal debug utilities (debug_utils.py not properly loaded)")

def initialize_renderer_extensions(renderer):
    """Initialize renderer extensions if available"""
    try:
        # First check if the file exists
        if not os.path.exists("renderer_high_res.py"):
            logger.warning("renderer_high_res.py file not found, skipping high-res initialization")
            return False

        import renderer_high_res
        success = renderer_high_res.initialize_renderer(renderer)
        if success:
            logger.info("High-resolution renderer extensions initialized")
        else:
            logger.warning("High-resolution renderer extensions failed to initialize")
        return success
    except ImportError as e:
        logger.info(f"High-resolution renderer extensions not available: {e}")
        return False
    except Exception as e:
        logger.error(f"Error initializing renderer extensions: {e}")
        traceback.print_exc()
        return False

def setup_opengl_context():
    """Set up the OpenGL context with proper attributes"""
    # Set OpenGL attributes BEFORE creating the window
    try:
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
    except:
        logger.warning("Could not set OpenGL context version - using default")
    
    # Set up anti-aliasing if enabled
    if ANTI_ALIASING:
        try:
            pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 1)
            pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, MSAA_SAMPLES)
        except:
            logger.warning("Could not enable anti-aliasing")
    
    # Set up double buffering
    try:
        pygame.display.gl_set_attribute(pygame.GL_DOUBLEBUFFER, 1)
    except:
        logger.warning("Could not set double buffering")
    
    # Set up depth buffer
    try:
        pygame.display.gl_set_attribute(pygame.GL_DEPTH_SIZE, 24)
    except:
        logger.warning("Could not set depth buffer size")
    
    logger.info("OpenGL context attributes set")

def main():
    """Main entry point for the application"""
    components = {}  # Store components for proper cleanup
    
    try:
        # Start debug system (with fallback)
        if has_debug_utils:
            debug_utils.initialize_debug()
        print("Starting Planetary Simulation")
        
        # Initialize pygame
        if not pygame.get_init():
            pygame.init()
        
        if not pygame.display.get_init():
            pygame.display.init()
        
        # Set up OpenGL context attributes
        setup_opengl_context()
            
        # Set up display
        pygame.display.set_caption("Gravity Simulation")
        flags = DOUBLEBUF | OPENGL
        if VSYNC:
            flags |= pygame.HWSURFACE
        if FULLSCREEN:
            flags |= pygame.FULLSCREEN
            
        # Create the screen
        try:
            screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), flags)
        except pygame.error as e:
            logger.error(f"Failed to create display: {e}")
            # Try with reduced settings
            pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 0)
            pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, 0)
            screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), DOUBLEBUF | OPENGL)
        
        # Print OpenGL information
        if has_debug_utils:
            debug_utils.print_gl_info()
            debug_utils.mark_gl_initialized()
        
        # Initialize components
        logger.info("Initializing simulation components...")
        if has_debug_utils:
            debug_utils.debug_print("Creating camera")
        components["camera"] = Camera()
        
        if has_debug_utils:
            debug_utils.debug_print("Creating simulation")
        components["simulation"] = Simulation()
        
        if has_debug_utils:
            debug_utils.debug_print("Creating renderer")
        components["renderer"] = Renderer(components["simulation"], components["camera"])
        
        if has_debug_utils:
            debug_utils.debug_print("Creating UI")
        components["ui"] = UI(components["simulation"])
        
        # Initialize renderer extensions
        initialize_renderer_extensions(components["renderer"])
        
        # Setup clock for frame timing
        clock = pygame.time.Clock()
        
        # Debug flag to test fixed camera and objects
        debug_fixed_view = False
        if debug_fixed_view:
            print("Using fixed debug view")
            components["camera"].distance = 50.0
            components["camera"].x_angle = 0.0
            components["camera"].y_angle = 0.2
            
        # Main loop
        running = True
        frame_count = 0
        last_time = time.time()
        
        # Ensure the screen is cleared initially
        glClearColor(0.0, 0.0, 0.05, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        pygame.display.flip()
        
        print("Entering main loop")
        while running:
            # Track frame rate
            current_time = time.time()
            frame_count += 1
            if current_time - last_time >= 1.0:
                fps = frame_count / (current_time - last_time)
                pygame.display.set_caption(f"Gravity Simulation - FPS: {fps:.1f}")
                frame_count = 0
                last_time = current_time
            
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == K_ESCAPE:
                        running = False
                    elif event.key == K_d and pygame.key.get_mods() & KMOD_CTRL:
                        # Toggle debug mode with Ctrl+D
                        if has_debug_utils:
                            debug_utils._debug_mode = not debug_utils._debug_mode
                            print(f"Debug mode: {'ON' if debug_utils._debug_mode else 'OFF'}")
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
            
            # Clear the screen
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            
            # Render scene
            components["renderer"].render()
            if has_debug_utils:
                debug_utils.check_gl_errors("after render")
            
            # Render UI on top
            components["ui"].render(screen)
            
            # Swap buffers
            pygame.display.flip()
            
            # Cap frame rate
            clock.tick(MAX_FPS)
        
        # Clean up resources
        logger.info("Cleaning up resources...")
        cleanup(components)
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        traceback.print_exc()
        cleanup(components)
        return 1
        
    logger.info("Application exited normally")
    return 0

def cleanup(components):
    """Clean up resources before exit"""
    # Clean up renderer first (OpenGL resources)
    if "renderer" in components:
        try:
            components["renderer"].cleanup()
        except Exception as e:
            logger.error(f"Error cleaning up renderer: {e}")
    
    # Clean up pygame
    try:
        pygame.quit()
    except Exception as e:
        logger.error(f"Error cleaning up pygame: {e}")

if __name__ == "__main__":
    sys.exit(main())