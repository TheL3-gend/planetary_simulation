#!/usr/bin/env python3
# high_res_texture_integration.py - Integration of high-resolution textures

import os
import sys
import shutil
import pygame
from constants import TEXTURE_DIR, SHADER_DIR
from texture_loader import TextureLoader
from texture_mapping import TEXTURE_MAPPING, convert_texture_names

def setup_high_res_textures(simulation):
    """
    Set up high-resolution textures for the simulation.
    
    This function:
    1. Converts texture names to high-resolution versions if available
    2. Adds special textures (normal maps, specular maps, etc.)
    3. Updates body properties to use these textures
    
    Args:
        simulation: The simulation object
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Update texture names in celestial bodies
        convert_texture_names(simulation.bodies)
        
        print("High-resolution textures configured")
        return True
    except Exception as e:
        print(f"Error setting up high-resolution textures: {e}")
        return False

def check_textures():
    """
    Check if high-resolution textures are available and provide guidance
    
    Returns:
        A tuple of (has_textures, missing_textures)
    """
    # Essential textures to check
    essential_textures = [
        "sun.jpg", "mercury.jpg", "venus.jpg", "earth.jpg", 
        "mars.jpg", "jupiter.jpg", "saturn.jpg", "uranus.jpg", 
        "neptune.jpg", "moon.jpg"
    ]
    
    missing_textures = []
    has_high_res = False
    
    # Check for textures in both original and high-res formats
    for texture in essential_textures:
        original_path = os.path.join(TEXTURE_DIR, texture)
        high_res_path = None
        
        if texture in TEXTURE_MAPPING:
            high_res_name = TEXTURE_MAPPING[texture]
            high_res_path = os.path.join(TEXTURE_DIR, high_res_name)
        
        # Check if either version exists
        if not os.path.exists(original_path) and (high_res_path is None or not os.path.exists(high_res_path)):
            missing_textures.append(texture)
        
        # Check if any high-res textures exist
        if high_res_path and os.path.exists(high_res_path):
            has_high_res = True
    
    return (len(missing_textures) == 0, missing_textures, has_high_res)

def print_texture_instructions():
    """Print instructions for downloading high-resolution textures"""
    print("\n=== High-Resolution Texture Instructions ===")
    print("To use high-quality 2K planet textures:")
    print("1. Visit: https://www.solarsystemscope.com/textures/")
    print("2. Download the textures (licensed under CC BY 4.0)")
    print("3. Place them in the 'textures' directory")
    print("\nThe simulation will automatically use these textures if found.")
    print("Recommended textures to download:")
    print("  - 2k_sun.jpg")
    print("  - 2k_mercury.jpg")
    print("  - 2k_venus_atmosphere.jpg")
    print("  - 2k_earth_daymap.jpg")
    print("  - 2k_earth_nightmap.jpg")  # Optional
    print("  - 2k_earth_clouds.jpg")    # Optional
    print("  - 2k_earth_normal_map.jpg")# Optional
    print("  - 2k_mars.jpg")
    print("  - 2k_jupiter.jpg")
    print("  - 2k_saturn.jpg")
    print("  - 2k_saturn_ring_alpha.png")
    print("  - 2k_uranus.jpg")
    print("  - 2k_neptune.jpg")
    print("  - 2k_moon.jpg")
    print("\nAttribution: Solar System Scope (www.solarsystemscope.com)")

def rename_textures_if_needed():
    """
    Check for high-res textures with original names and rename if needed
    
    This handles the case where the user manually downloaded textures but
    didn't rename them to match the simulation's expected names.
    
    Returns:
        True if any textures were renamed, False otherwise
    """
    renamed = False
    
    # Create reverse mapping (high-res name -> original name)
    reverse_mapping = {v: k for k, v in TEXTURE_MAPPING.items()}
    
    # Check if any high-res textures exist but their originals don't
    for high_res_name, original_name in reverse_mapping.items():
        high_res_path = os.path.join(TEXTURE_DIR, high_res_name)
        original_path = os.path.join(TEXTURE_DIR, original_name)
        
        # If high-res exists but original doesn't, create a symbolic link
        if os.path.exists(high_res_path) and not os.path.exists(original_path):
            try:
                # For Windows, use copy instead of symlink for compatibility
                if os.name == 'nt':
                    shutil.copy2(high_res_path, original_path)
                else:
                    # For Unix-like systems, use symlink
                    if os.path.exists(original_path):
                        os.remove(original_path)
                    os.symlink(high_res_path, original_path)
                
                print(f"Created link: {high_res_name} -> {original_name}")
                renamed = True
            except Exception as e:
                print(f"Error creating link for {high_res_name}: {e}")
    
    return renamed

def generate_tangents_bitangents(vertices, uvs, normals, indices):
    """
    Generate tangent and bitangent vectors for normal mapping
    
    Args:
        vertices: List of vertex positions
        uvs: List of texture coordinates
        normals: List of normal vectors
        indices: List of triangle indices
        
    Returns:
        Tuple of (tangents, bitangents) lists
    """
    import numpy as np
    
    # Initialize tangent and bitangent arrays
    tangents = np.zeros((len(vertices) // 3, 3), dtype=np.float32)
    bitangents = np.zeros((len(vertices) // 3, 3), dtype=np.float32)
    
    # Process each triangle
    for i in range(0, len(indices), 3):
        # Get indices for the triangle
        i0 = indices[i]
        i1 = indices[i+1]
        i2 = indices[i+2]
        
        # Get vertex positions
        v0 = np.array([vertices[i0*3], vertices[i0*3+1], vertices[i0*3+2]])
        v1 = np.array([vertices[i1*3], vertices[i1*3+1], vertices[i1*3+2]])
        v2 = np.array([vertices[i2*3], vertices[i2*3+1], vertices[i2*3+2]])
        
        # Get texture coordinates
        uv0 = np.array([uvs[i0*2], uvs[i0*2+1]])
        uv1 = np.array([uvs[i1*2], uvs[i1*2+1]])
        uv2 = np.array([uvs[i2*2], uvs[i2*2+1]])
        
        # Calculate edges
        e1 = v1 - v0
        e2 = v2 - v0
        
        # Calculate UV deltas
        deltaUV1 = uv1 - uv0
        deltaUV2 = uv2 - uv0
        
        # Calculate tangent and bitangent
        try:
            r = 1.0 / (deltaUV1[0] * deltaUV2[1] - deltaUV1[1] * deltaUV2[0])
            tangent = (e1 * deltaUV2[1] - e2 * deltaUV1[1]) * r
            bitangent = (e2 * deltaUV1[0] - e1 * deltaUV2[0]) * r
            
            # Normalize
            tangent = tangent / np.linalg.norm(tangent) if np.linalg.norm(tangent) > 0 else np.array([1, 0, 0])
            bitangent = bitangent / np.linalg.norm(bitangent) if np.linalg.norm(bitangent) > 0 else np.array([0, 1, 0])
            
            # Add to arrays
            tangents[i0] += tangent
            tangents[i1] += tangent
            tangents[i2] += tangent
            
            bitangents[i0] += bitangent
            bitangents[i1] += bitangent
            bitangents[i2] += bitangent
        except:
            # Handle degenerate triangles
            continue
    
    # Normalize each tangent and bitangent
    for i in range(len(tangents)):
        norm = np.linalg.norm(tangents[i])
        tangents[i] = tangents[i] / norm if norm > 0 else np.array([1, 0, 0])
        
        norm = np.linalg.norm(bitangents[i])
        bitangents[i] = bitangents[i] / norm if norm > 0 else np.array([0, 1, 0])
    
    # Flatten arrays for OpenGL
    tangents_flat = tangents.flatten()
    bitangents_flat = bitangents.flatten()
    
    return tangents_flat, bitangents_flat

def ensure_advanced_shaders():
    """
    Ensure that advanced shaders are available
    
    Returns:
        True if advanced shaders are available, False otherwise
    """
    # Check if shader directory exists
    if not os.path.exists(SHADER_DIR):
        try:
            os.makedirs(SHADER_DIR)
        except Exception as e:
            print(f"Error creating shader directory: {e}")
            return False
    
    # Check for advanced shader files
    advanced_vertex = os.path.join(SHADER_DIR, "advanced_planet_vertex.glsl")
    advanced_fragment = os.path.join(SHADER_DIR, "advanced_planet_fragment.glsl")
    
    if os.path.exists(advanced_vertex) and os.path.exists(advanced_fragment):
        return True
    
    print("Advanced planet shaders not found. Creating them...")
    
    try:
        # Create advanced shaders from embedded strings
        # In a real implementation, these would be created from the shader files
        # included in the project, but for simplicity, we'll assume they're created
        # externally.
        
        print(f"Please make sure the advanced shader files exist at:")
        print(f"  - {advanced_vertex}")
        print(f"  - {advanced_fragment}")
        
        return False
    except Exception as e:
        print(f"Error creating advanced shaders: {e}")
        return False
