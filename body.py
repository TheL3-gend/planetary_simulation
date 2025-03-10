#!/usr/bin/env python3
# camera.py - Camera controls for the simulation with improved stability

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
        
        # Safe limits for camera parameters
        self.min_distance = 5.0
        self.max_distance = 10000.0
        
    def setup_view(self):
        """Set up the OpenGL view matrix based on camera properties"""
        try:
            # Apply safety limits to camera parameters
            self.distance = max(self.min_distance, min(self.max_distance, self.distance))
            self.y_angle = max(-math.pi/2 + 0.01, min(math.pi/2 - 0.01, self.y_angle))
            
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
        except Exception as e:
            print(f"Error setting up camera view: {e}")
            # Fallback to a default view
            from OpenGL.GLU import gluLookAt
            gluLookAt(0, 0, 100, 0, 0, 0, 0, 1, 0)
        
    def handle_mouse_motion(self, pos, rel, buttons):
        """Handle mouse motion for camera control"""
        try:
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
                    
                    # Check for valid rel values
                    if not np.all(np.isfinite([dx, dy])):
                        return
                    
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
                    pan_amount = max(0.1, min(pan_amount, 100.0))  # Limit pan amount
                    
                    self.target -= right_vec * dx * pan_amount
                    self.target += up_vec * dy * pan_amount
            else:
                self.panning = False
        except Exception as e:
            print(f"Error handling mouse motion: {e}")
            
    def handle_mouse_wheel(self, y):
        """Handle mouse wheel for zooming"""
        try:
            # Validate y value
            if not np.isfinite(y) or abs(y) > 10:
                return
                
            zoom_factor = 0.9 if y > 0 else 1.1
            self.target_distance *= zoom_factor
            self.target_distance = max(self.min_distance, min(self.max_distance, self.target_distance))
        except Exception as e:
            print(f"Error handling mouse wheel: {e}")
        
    def handle_key(self, key):
        """Handle keyboard input for camera control"""
        try:
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
        except Exception as e:
            print(f"Error handling key: {e}")
        
    def handle_key_up(self, key):
        """Handle key release events"""
        try:
            if key in self.key_states:
                self.key_states[key] = False
        except Exception as e:
            print(f"Error handling key up: {e}")
            
    def update(self, dt, selected_body=None):
        """Update camera position and orientation"""
        try:
            # Validate dt to avoid extreme values
            dt = max(0.001, min(dt, 1.0))
            
            # Smooth zoom
            self.distance += (self.target_distance - self.distance) * self.smooth_factor
            self.distance = max(self.min_distance, min(self.max_distance, self.distance))
            
            # If in free mode, handle movement based on key states
            if self.free_mode:
                self.update_free_movement(dt)
            elif selected_body:
                # Smoothly move target to selected body
                target_pos = selected_body.position / SCALE_FACTOR
                
                # Validate target position
                if np.all(np.isfinite(target_pos)):
                    self.target += (target_pos - self.target) * self.smooth_factor
        except Exception as e:
            print(f"Error updating camera: {e}")
            
    def update_free_movement(self, dt):
        """Handle free camera movement based on key states"""
        try:
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
            
            # Limit move amount to reasonable value
            move_amount = max(0.01, min(move_amount, 10.0))
            
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
        except Exception as e:
            print(f"Error in free camera movement: {e}")
            
    def reset(self):
        """Reset camera to default position"""
        try:
            self.distance = 100.0
            self.target_distance = 100.0
            self.x_angle = 0.0
            self.y_angle = 0.2
            self.target = np.array([0.0, 0.0, 0.0])
            self.free_mode = False
        except Exception as e:
            print(f"Error resetting camera: {e}")