#!/usr/bin/env python3
# test_opengl.py - Minimal OpenGL test

import pygame
from OpenGL.GL import *
from OpenGL.GLU import *
import sys

def main():
    # Initialize pygame
    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, pygame.DOUBLEBUF | pygame.OPENGL)
    
    # Set up OpenGL
    glClearColor(0.0, 0.0, 0.2, 1.0)
    glEnable(GL_DEPTH_TEST)
    
    # Set up perspective
    gluPerspective(45, (display[0]/display[1]), 0.1, 50.0)
    glTranslatef(0.0, 0.0, -5)
    
    # Main loop
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Clear screen
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Draw a simple triangle
        glBegin(GL_TRIANGLES)
        glColor3f(1.0, 0.0, 0.0)
        glVertex3f(0.0, 1.0, 0.0)
        glColor3f(0.0, 1.0, 0.0)
        glVertex3f(-1.0, -1.0, 0.0)
        glColor3f(0.0, 0.0, 1.0)
        glVertex3f(1.0, -1.0, 0.0)
        glEnd()
        
        # Update screen
        pygame.display.flip()
        pygame.time.wait(10)
    
    pygame.quit()
    return 0

if __name__ == "__main__":
    sys.exit(main())