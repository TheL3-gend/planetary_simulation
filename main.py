#!/usr/bin/env python3
# main.py - Main entry point for gravity simulation

import sys
import pygame
from pygame.locals import *
from OpenGL.GL import *
import traceback
import logging

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

def initialize_renderer_extensions(renderer):
    """Initialize renderer extensions if available"""
    try:
        import renderer_high_res
        success = renderer_high_res.initialize_renderer(renderer)
        if success:
            logger.info("High-resolution renderer extensions initialized")
        else:
            logger.warning("High-resolution renderer extensions failed to initialize")
    except ImportError:
        logger.info("High-resolution renderer extensions not available")
    except Exception as e:
        logger.error(f"Error initializing renderer extensions: {e}")

def main():
    """Main entry point for the application"""
    components = {}  # Store components for proper cleanup
    
    try:
        # Initialize pygame
        if not pygame.get_init():
            pygame.init()
        
        if not pygame.display.get_init():
            pygame.display.init()
            
        # Set up display
        pygame.display.set_caption("Gravity Simulation")
        flags = DOUBLEBUF | OPENGL | pygame.HWSURFACE
        if FULLSCREEN:
            flags |= pygame.FULLSCREEN
            
        screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), flags)
        
        # Initialize components
        logger.info("Initializing simulation components...")
        components["camera"] = Camera()
        components["simulation"] = Simulation()
        components["renderer"] = Renderer(components["simulation"], components["camera"])
        components["ui"] = UI(components["simulation"])
        
        # Initialize renderer extensions
        initialize_renderer_extensions(components["renderer"])
        
        # Setup clock for frame timing
        clock = pygame.time.Clock()
        
        # Main loop
        running = True
        while running:
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
            
            # Render scene
            components["renderer"].render()
            
            # Render UI on top
            components["ui"].render(screen)
            
            # Swap buffers
            pygame.display.flip()
            
            # Cap frame rate
            clock.tick(MAX_FPS)
            
            # Optional: print FPS
            # logger.debug(f"FPS: {clock.get_fps():.1f}")
        
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