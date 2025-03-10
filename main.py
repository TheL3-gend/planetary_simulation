#!/usr/bin/env python3
# main.py - Main entry point for gravity simulation

import sys
import pygame
from pygame.locals import *
from OpenGL.GL import *

# Import our modules
from constants import *
from camera import Camera
from simulation import Simulation
from renderer import Renderer
from ui import UI
# Near the top of your main.py, add:
import renderer_high_res

# After creating your renderer, add this line:
renderer_high_res.initialize_renderer(renderer)

def main():
    """Main entry point for the application"""
    try:
        # Initialize pygame
        pygame.init()
        
        # Set up display
        pygame.display.set_caption("Gravity Simulation")
        flags = DOUBLEBUF | OPENGL | pygame.HWSURFACE
        if FULLSCREEN:
            flags |= pygame.FULLSCREEN
            
        screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), flags)
        
        # Initialize components
        camera = Camera()
        simulation = Simulation()
        renderer = Renderer(simulation, camera)
        ui = UI(simulation)
        
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
                        simulation.handle_key(event.key)
                        camera.handle_key(event.key)
                elif event.type == pygame.MOUSEMOTION:
                    camera.handle_mouse_motion(event.pos, event.rel, pygame.mouse.get_pressed())
                elif event.type == pygame.MOUSEWHEEL:
                    camera.handle_mouse_wheel(event.y)
            
            # Update simulation
            simulation.update()
            
            # Render scene
            renderer.render()
            
            # Render UI on top
            ui.render(screen)
            
            # Swap buffers
            pygame.display.flip()
            
            # Cap frame rate
            clock.tick(MAX_FPS)
        
        # Clean up resources
        renderer.cleanup()
        pygame.quit()
        
    except Exception as e:
        print(f"Fatal error: {e}")
        pygame.quit()
        sys.exit(1)

if __name__ == "__main__":
    main()