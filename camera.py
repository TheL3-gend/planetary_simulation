#!/usr/bin/env python3
# camera.py - Camera controls for the simulation

import numpy as np
import math
from pygame.locals import *
import pygame
from constants import *

class Camera:
    """Handles camera positioning and movement in the 3D scene"""
    
    def __init__(self):
        """Initialize camera with default values"""
        self.distance = 100.0
        self.x_angle = 0.0
        self.y_angle = 0.2  # Slight angle to see the ecliptic plane
        self.target = np.array([0.0, 0.0, 0.0])
        self.follow_target = None
        self.smooth_factor = 0.1  # For smooth camera movement
        self.target_distance = self.distance  # For smooth zooming
        
        # For panning with middle mouse button
        self.panning = False
        self.pan_reference = None
        self.pan_sensitivity = 0.1
        
        # Camera mode
        self.free_mode = False  # If True, camera moves freely; if False, orbits target
        
        # Movement speed
        self.movement_speed = 5.0
        
        # Key states for continuous movement
        self.key_states = {
            K_w: False,
            K_a: False,
            K_s: False,
            K_d: False,
            K_q: False,
            K_e: False
        }
        
    def setup_view(self):
        """Set up the OpenGL view matrix based on camera properties"""
        # Calculate camera position in Cartesian coordinates
        eye_x = self.target[0] + self.distance * np.sin(self.x_angle) * np.cos(self.y_angle)
        eye_y = self.target[1] + self.distance * np.sin(self.y_angle)
        eye_z = self.target[2] + self.distance * np.cos(self.x_angle) * np.cos(self.y_angle)
        
        # Set up look-at matrix
        from OpenGL.GLU import gluLookAt
        gluLookAt(
            eye_x, eye_y, eye_z,  # Eye position
            self.target[0], self.target[1], self.target[2],  # Target
            0, 1, 0  # Up vector
        )
        
    def handle_mouse_motion(self, pos, rel, buttons):
        """Handle mouse motion for camera control"""
        if buttons[0]:  # Left mouse button - rotation
            dx, dy = rel
            
            # Fix inverted mouse by negating dy
            self.x_angle -= dx * 0.01  # Negative to fix inversion
            self.y_angle -= dy * 0.01  # Negative to fix inversion
            
            # Clamp y angle to avoid gimbal lock
            self.y_angle = np.clip(self.y_angle, -np.pi/2 + 0.1, np.pi/2 - 0.1)
            
        elif buttons[1]:  # Middle mouse button - panning
            if not self.panning:
                self.panning = True
                self.pan_reference = pos
            else:
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
                pan_amount = self.distance * self.pan_sensitivity
                self.target -= right_vec * dx * pan_amount
                self.target += up_vec * dy * pan_amount
        else:
            self.panning = False
            
    def handle_mouse_wheel(self, y):
        """Handle mouse wheel for zooming"""
        zoom_factor = 0.9 if y > 0 else 1.1
        self.target_distance *= zoom_factor
        self.target_distance = max(10.0, min(5000.0, self.target_distance))
        
    def handle_key(self, key):
        """Handle keyboard input for camera control"""
        if key == K_f:
            self.free_mode = not self.free_mode
        elif key == K_c:
            self.reset()
        elif key == K_LEFT:
            self.x_angle += 0.1
        elif key == K_RIGHT:
            self.x_angle -= 0.1
        elif key == K_UP:
            self.y_angle += 0.1
            self.y_angle = min(self.y_angle, np.pi/2 - 0.1)
        elif key == K_DOWN:
            self.y_angle -= 0.1
            self.y_angle = max(self.y_angle, -np.pi/2 + 0.1)
        elif key in self.key_states:
            self.key_states[key] = True
        
        # Key up events handled by pygame.KEYUP
        
    def handle_key_up(self, key):
        """Handle key release events"""
        if key in self.key_states:
            self.key_states[key] = False
            
    def update(self, dt, selected_body=None):
        """Update camera position and orientation"""
        # Smooth zoom
        self.distance += (self.target_distance - self.distance) * self.smooth_factor
        
        # If in free mode, handle movement based on key states
        if self.free_mode:
            self.update_free_movement(dt)
        elif selected_body:
            # Smoothly move target to selected body
            target_pos = selected_body.position / SCALE_FACTOR
            self.target += (target_pos - self.target) * self.smooth_factor
            
    def update_free_movement(self, dt):
        """Handle free camera movement based on key states"""
        # Calculate movement vectors in camera space
        forward_vec = np.array([
            np.sin(self.x_angle) * np.cos(self.y_angle),
            np.sin(self.y_angle),
            np.cos(self.x_angle) * np.cos(self.y_angle)
        ])
        
        right_vec = np.array([
            np.cos(self.x_angle),
            0,
            -np.sin(self.x_angle)
        ])
        
        up_vec = np.array([0, 1, 0])
        
        # Apply movement based on key states
        move_amount = self.movement_speed * dt
        
        if self.key_states[K_w]:  # Forward
            self.target += forward_vec * move_amount
        if self.key_states[K_s]:  # Backward
            self.target -= forward_vec * move_amount
        if self.key_states[K_a]:  # Left
            self.target -= right_vec * move_amount
        if self.key_states[K_d]:  # Right
            self.target += right_vec * move_amount
        if self.key_states[K_q]:  # Down
            self.target -= up_vec * move_amount
        if self.key_states[K_e]:  # Up
            self.target += up_vec * move_amount
            
    def reset(self):
        """Reset camera to default position"""
        self.distance = 100.0
        self.target_distance = 100.0
        self.x_angle = 0.0
        self.y_angle = 0.2
        self.target = np.array([0.0, 0.0, 0.0])
        self.free_mode = False