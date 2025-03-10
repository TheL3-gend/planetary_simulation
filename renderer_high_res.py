#!/usr/bin/env python3
# renderer_high_res.py - High-resolution rendering extension for planetary simulation

import os
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
import logging
from constants import SHADER_DIR, TEXTURE_DIR

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("RendererHighRes")

def initialize_renderer(renderer):
    """
    Initialize high-resolution rendering features
    
    Args:
        renderer: The main renderer object
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.info("Initializing high-resolution renderer")
        
        # Store the original draw_bodies method
        if not hasattr(renderer, '_original_draw_bodies'):
            renderer._original_draw_bodies = renderer.draw_bodies
        
        # Replace draw_bodies with our version that will delegate back to the original
        def new_draw_bodies(view_matrix, projection_matrix):
            return draw_bodies(renderer, view_matrix, projection_matrix)
            
        renderer.draw_bodies = new_draw_bodies
        
        # Add high-res flag
        renderer.use_high_res = True
        
        # Load shader for advanced rendering
        load_high_res_shaders(renderer)
        
        # Mark initialization as successful
        logger.info("High-resolution renderer initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize high-resolution renderer: {e}")
        # Revert any changes made
        if hasattr(renderer, '_original_draw_bodies'):
            renderer.draw_bodies = renderer._original_draw_bodies
        return False

def load_high_res_shaders(renderer):
    """
    Load advanced shaders for high-resolution rendering
    
    Args:
        renderer: The renderer object
    """
    if not hasattr(renderer, 'shader_manager'):
        logger.error("Renderer has no shader manager")
        return False
    
    # Check for advanced shader files
    vertex_path = os.path.join(SHADER_DIR, "advanced_planet_vertex.glsl")
    fragment_path = os.path.join(SHADER_DIR, "advanced_planet_fragment.glsl")
    
    # Try alternative files if the primary ones don't exist
    alt_vertex = os.path.join(SHADER_DIR, "advanced-planet-shader.txt")
    alt_fragment = os.path.join(SHADER_DIR, "advanced-planet-fragment.txt")
    
    # Use the correct files if they exist
    if os.path.exists(vertex_path) and os.path.exists(fragment_path):
        try:
            # Load vertex shader
            with open(vertex_path, 'r') as f:
                vertex_src = f.read()
                
            # Load fragment shader
            with open(fragment_path, 'r') as f:
                fragment_src = f.read()
                
            # Compile shaders
            shader_program = renderer.shader_manager.compile_shader(vertex_src, fragment_src)
            if shader_program:
                renderer.shader_manager.shaders["advanced_planet"] = shader_program
                logger.info("Loaded advanced planet shader")
                return True
        except Exception as e:
            logger.error(f"Error loading advanced shaders: {e}")
    
    # Try alternative files
    elif os.path.exists(alt_vertex) and os.path.exists(alt_fragment):
        try:
            # Load vertex shader
            with open(alt_vertex, 'r') as f:
                vertex_src = f.read()
                
            # Load fragment shader
            with open(alt_fragment, 'r') as f:
                fragment_src = f.read()
                
            # Compile shaders
            shader_program = renderer.shader_manager.compile_shader(vertex_src, fragment_src)
            if shader_program:
                renderer.shader_manager.shaders["advanced_planet"] = shader_program
                logger.info("Loaded advanced planet shader from alternative files")
                return True
        except Exception as e:
            logger.error(f"Error loading advanced shaders from alternative files: {e}")
    
    logger.warning("Advanced shaders not found, using standard shaders")
    return False

def draw_bodies(renderer, view_matrix, projection_matrix):
    """
    Draw all celestial bodies with high-resolution features if available
    
    Args:
        renderer: The renderer object
        view_matrix: The view matrix
        projection_matrix: The projection matrix
    """
    # Just use the original method - high-res not implemented yet
    try:
        renderer._original_draw_bodies(view_matrix, projection_matrix)
    except Exception as e:
        logger.error(f"Error in high-resolution draw_bodies: {e}")
        # Clean up OpenGL state
        glBindVertexArray(0)
        glUseProgram(0)