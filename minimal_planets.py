#!/usr/bin/env python3
# minimal_planets.py - Minimal planetary simulation with OpenGL and Pygame

import sys
import os
import math
import time
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np

# Create necessary directories
os.makedirs("textures", exist_ok=True)

# Basic constants
WINDOW_WIDTH = 1024
WINDOW_HEIGHT = 768
MAX_FPS = 60

# Basic solar system data (name, radius, distance, rotation speed, color)
PLANETS = [
    ("Sun", 1.0, 0.0, 0.0, (1.0, 0.7, 0.0)),
    ("Mercury", 0.2, 3.0, 0.8, (0.7, 0.7, 0.7)),
    ("Venus", 0.3, 5.0, 0.6, (0.9, 0.7, 0.0)),
    ("Earth", 0.35, 7.0, 0.5, (0.0, 0.5, 1.0)),
    ("Mars", 0.25, 9.0, 0.4, (1.0, 0.0, 0.0)),
    ("Jupiter", 0.8, 13.0, 0.2, (0.8, 0.6, 0.0)),
    ("Saturn", 0.7, 17.0, 0.15, (0.9, 0.7, 0.2)),
    ("Uranus", 0.5, 21.0, 0.1, (0.5, 0.7, 0.9)),
    ("Neptune", 0.5, 25.0, 0.05, (0.0, 0.0, 0.8)),
]

class Camera:
    def __init__(self):
        self.distance = 20.0
        self.x_angle = 0.0
        self.y_angle = 0.3
        self.target = np.array([0.0, 0.0, 0.0])
        self.panning = False
        self.pan_reference = None
    
    def setup_view(self):
        glLoadIdentity()
        # Calculate camera position in Cartesian coordinates
        eye_x = self.target[0] + self.distance * np.sin(self.x_angle) * np.cos(self.y_angle)
        eye_y = self.target[1] + self.distance * np.sin(self.y_angle)
        eye_z = self.target[2] + self.distance * np.cos(self.x_angle) * np.cos(self.y_angle)
        
        # Set up look-at matrix
        gluLookAt(
            eye_x, eye_y, eye_z,  # Eye position
            self.target[0], self.target[1], self.target[2],  # Target
            0, 1, 0  # Up vector
        )
    
    def handle_mouse_motion(self, pos, rel, buttons):
        if buttons[0]:  # Left mouse button - rotation
            dx, dy = rel
            self.x_angle -= dx * 0.01
            self.y_angle -= dy * 0.01
            self.y_angle = np.clip(self.y_angle, -np.pi/2 + 0.1, np.pi/2 - 0.1)
        elif buttons[1]:  # Middle mouse button - panning
            dx, dy = rel
            
            # Calculate right and up vectors in camera space
            right_vec = np.array([
                np.cos(self.x_angle),
                0,
                -np.sin(self.x_angle)
            ])
            
            up_vec = np.array([
                np.sin(self.x_angle) * np.sin(self.y_angle),
                np.cos(self.y_angle),
                np.cos(self.x_angle) * np.sin(self.y_angle)
            ])
            
            # Move target in these directions
            pan_amount = self.distance * 0.005
            self.target -= right_vec * dx * pan_amount
            self.target += up_vec * dy * pan_amount
    
    def handle_mouse_wheel(self, y):
        self.distance *= 0.9 if y > 0 else 1.1
        self.distance = max(5.0, min(50.0, self.distance))

def create_sphere(radius, slices, stacks):
    # Create a sphere mesh using vertices and triangles
    vertices = []
    
    for stack in range(stacks + 1):
        phi = stack * math.pi / stacks
        for slice in range(slices + 1):
            theta = slice * 2 * math.pi / slices
            
            # Calculate vertex position
            x = radius * math.sin(phi) * math.cos(theta)
            y = radius * math.cos(phi)
            z = radius * math.sin(phi) * math.sin(theta)
            
            # Add vertex with position, normal, and texture coordinates
            vertices.append((x, y, z))
    
    # Create triangle indices
    indices = []
    for stack in range(stacks):
        for slice in range(slices):
            # Calculate indices for two triangles per quad
            p1 = stack * (slices + 1) + slice
            p2 = p1 + 1
            p3 = p1 + (slices + 1)
            p4 = p3 + 1
            
            # First triangle
            indices.append((p1, p3, p2))
            # Second triangle
            indices.append((p2, p3, p4))
    
    return vertices, indices

def draw_sphere(radius, slices, stacks):
    # Draw a sphere using GLU quadrics - simpler than manual vertex setup
    quadric = gluNewQuadric()
    gluQuadricTexture(quadric, GL_TRUE)
    gluQuadricNormals(quadric, GLU_SMOOTH)
    gluSphere(quadric, radius, slices, stacks)
    gluDeleteQuadric(quadric)

def draw_orbit(radius, segments=100):
    # Draw circular orbit path
    glBegin(GL_LINE_LOOP)
    for i in range(segments):
        angle = i * 2 * math.pi / segments
        glVertex3f(radius * math.cos(angle), 0, radius * math.sin(angle))
    glEnd()

def main():
    # Initialize pygame
    pygame.init()
    
    # Set up display
    pygame.display.set_caption("Minimal Planetary Simulation")
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), DOUBLEBUF | OPENGL)
    
    # Initialize OpenGL
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
    
    # Set up light position
    glLightfv(GL_LIGHT0, GL_POSITION, (0, 0, 0, 1))
    glLightfv(GL_LIGHT0, GL_DIFFUSE, (1, 1, 1, 1))
    
    # Set up projection matrix
    glMatrixMode(GL_PROJECTION)
    gluPerspective(45, WINDOW_WIDTH / WINDOW_HEIGHT, 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)
    
    # Create camera
    camera = Camera()
    
    # Create clock for timing
    clock = pygame.time.Clock()
    
    # Simulation parameters
    time_factor = 0.2
    paused = False
    
    # Main loop
    running = True
    start_time = time.time()
    
    while running:
        # Process events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == K_ESCAPE:
                    running = False
                elif event.key == K_SPACE:
                    paused = not paused
                elif event.key == K_EQUALS or event.key == K_PLUS:
                    time_factor *= 1.5
                elif event.key == K_MINUS:
                    time_factor /= 1.5
            elif event.type == pygame.MOUSEMOTION:
                camera.handle_mouse_motion(event.pos, event.rel, pygame.mouse.get_pressed())
            elif event.type == pygame.MOUSEWHEEL:
                camera.handle_mouse_wheel(event.y)
        
        # Clear screen
        glClearColor(0.0, 0.0, 0.05, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Set up camera
        camera.setup_view()
        
        # Calculate simulation time
        if not paused:
            sim_time = (time.time() - start_time) * time_factor
        else:
            sim_time = (pygame.time.get_ticks() / 1000) * 0  # Time doesn't advance when paused
        
        # Draw orbits
        glDisable(GL_LIGHTING)
        glColor3f(0.3, 0.3, 0.3)
        for planet_name, planet_radius, planet_distance, planet_speed, planet_color in PLANETS:
            if planet_distance > 0:  # Skip sun
                draw_orbit(planet_distance)
        glEnable(GL_LIGHTING)
        
        # Draw planets
        for planet_name, planet_radius, planet_distance, planet_speed, planet_color in PLANETS:
            glPushMatrix()
            
            # Position planet in orbit
            if planet_distance > 0:  # Not the sun
                angle = sim_time * planet_speed
                x = planet_distance * math.cos(angle)
                z = planet_distance * math.sin(angle)
                glTranslatef(x, 0, z)
            
            # Set planet color
            glColor3f(*planet_color)
            
            # Draw planet
            draw_sphere(planet_radius, 32, 16)
            
            glPopMatrix()
        
        # Swap buffers
        pygame.display.flip()
        
        # Cap frame rate
        clock.tick(MAX_FPS)
    
    # Clean up
    pygame.quit()
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