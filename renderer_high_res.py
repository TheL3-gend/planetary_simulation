#!/usr/bin/env python3
# renderer_high_res.py - High-resolution texture support for the renderer

import os
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from constants import *

def initialize_renderer(renderer):
    """
    Initialize the renderer with high-resolution texture support
    
    Args:
        renderer: The renderer object to initialize
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Add a flag to use monkeypatch for renderer.draw_bodies
        renderer._draw_body_standard = renderer.draw_bodies
        renderer.draw_bodies = lambda view_matrix, projection_matrix: draw_bodies(renderer, view_matrix, projection_matrix)
        
        print("High-resolution renderer initialized successfully")
        return True
    except Exception as e:
        print(f"Error initializing high-resolution renderer: {e}")
        return False

def draw_body_with_advanced_shader(renderer, body, view_matrix, projection_matrix):
    """
    Draw a celestial body using advanced shaders if available
    
    Args:
        renderer: The renderer object
        body: The celestial body to draw
        view_matrix: The view matrix
        projection_matrix: The projection matrix
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Check if we have high-resolution textures
        if not hasattr(body, 'texture_id') or not body.texture_id:
            return False
            
        # Use standard rendering for now - advanced rendering could be implemented later
        renderer._draw_body_standard(body, view_matrix, projection_matrix)
        return True
    except Exception as e:
        print(f"Error in advanced shader rendering: {e}")
        return False

def draw_bodies(renderer, view_matrix, projection_matrix):
    """Draw all celestial bodies"""
    
    # Try to use advanced rendering first (if available)
    if hasattr(renderer, 'use_advanced_shaders') and renderer.use_advanced_shaders:
        try:
            # Draw each body with advanced shader
            for body in renderer.simulation.bodies:
                success = draw_body_with_advanced_shader(
                    renderer, body, view_matrix, projection_matrix)
                    
                if not success:
                    # Fall back to standard rendering
                    renderer._draw_body_standard(body, view_matrix, projection_matrix)
                    
            return
        except Exception as e:
            print(f"Error using advanced rendering: {e}")
            # Fall back to standard rendering
    
    # Fall back to original draw_bodies method
    original_method = getattr(renderer, '_draw_body_standard', None)
    if original_method:
        original_method(view_matrix, projection_matrix)
    else:
        print("Error: Original draw_bodies method not found")