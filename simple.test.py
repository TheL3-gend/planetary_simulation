#!/usr/bin/env python3
# simple_test.py - Test OpenGL functionality

import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import sys
import math
import time

def main():
    """Simple test of OpenGL functionality"""
    print("Starting OpenGL test...")
    
    # Initialize Pygame
    pygame.init()
    
    # Set up display
    width, height = 800, 600
    screen = pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL)
    pygame.display.set_caption("OpenGL Test")
    
    # Print OpenGL version
    print(f"OpenGL Version: {glGetString(GL_VERSION).decode()}")
    print(f"OpenGL Renderer: {glGetString(GL_RENDERER).decode()}")
    print(f"OpenGL Vendor: {glGetString(GL_VENDOR).decode()}")
    
    # Set up projection
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, width/height, 0.1, 50.0)
    
    # Set up modelview
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glTranslatef(0, 0, -5)
    
    # Enable depth testing
    glEnable(GL_DEPTH_TEST)
    
    # Set clear color
    glClearColor(0.1, 0.1, 0.2, 1.0)
    
    # Create a simple sphere using GLU quadric
    sphere = gluNewQuadric()
    gluQuadricDrawStyle(sphere, GLU_FILL)
    gluQuadricNormals(sphere, GLU_SMOOTH)
    
    # Create a clock for timing
    clock = pygame.time.Clock()
    
    # Set up initial rotation
    rotation_x = 0
    rotation_y = 0
    
    # Variables for mouse controls
    dragging = False
    last_mouse_pos = None
    
    # Main loop
    running = True
    start_time = time.time()
    frames = 0
    
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == K_ESCAPE:
                    running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left mouse button
                    dragging = True
                    last_mouse_pos = event.pos
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:  # Left mouse button
                    dragging = False
            elif event.type == pygame.MOUSEMOTION:
                if dragging and last_mouse_pos:
                    dx, dy = event.pos[0] - last_mouse_pos[0], event.pos[1] - last_mouse_pos[1]
                    rotation_y += dx * 0.5
                    rotation_x += dy * 0.5
                    last_mouse_pos = event.pos
        
        # Clear the screen
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Render
        glPushMatrix()
        
        # Apply rotations
        glRotatef(rotation_x, 1, 0, 0)
        glRotatef(rotation_y, 0, 1, 0)
        
        # Draw colored sphere
        glColor3f(0.0, 0.5, 1.0)  # Blue
        gluSphere(sphere, 1.0, 32, 16)
        
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
        
        glPopMatrix()
        
        # Swap buffers
        pygame.display.flip()
        
        # Cap frame rate
        clock.tick(60)
        
        # Calculate FPS
        frames += 1
        if frames % 60 == 0:
            current_time = time.time()
            fps = frames / (current_time - start_time)
            pygame.display.set_caption(f"OpenGL Test - FPS: {fps:.1f}")
    
    # Clean up
    gluDeleteQuadric(sphere)
    pygame.quit()
    print("Test completed.")
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        pygame.quit()
        sys.exit(1)