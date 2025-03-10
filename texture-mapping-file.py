#!/usr/bin/env python3
# texture_mapping.py - Mapping for Solar System Scope high-resolution textures

import os
from constants import TEXTURE_DIR

# Map between expected texture filenames and Solar System Scope filenames
TEXTURE_MAPPING = {
    "sun.jpg": "2k_sun.jpg",
    "mercury.jpg": "2k_mercury.jpg",
    "venus.jpg": "2k_venus_atmosphere.jpg",
    "earth.jpg": "2k_earth_daymap.jpg",
    "mars.jpg": "2k_mars.jpg",
    "jupiter.jpg": "2k_jupiter.jpg",
    "saturn.jpg": "2k_saturn.jpg",
    "uranus.jpg": "2k_uranus.jpg", 
    "neptune.jpg": "2k_neptune.jpg",
    "moon.jpg": "2k_moon.jpg",
    "phobos.jpg": "2k_phobos.jpg",
    "deimos.jpg": "2k_deimos.jpg",
    "io.jpg": "2k_io.jpg",
    "europa.jpg": "2k_europa.jpg",
    "ganymede.jpg": "2k_ganymede.jpg",
    "callisto.jpg": "2k_callisto.jpg",
    "titan.jpg": "2k_titan.jpg",
    "enceladus.jpg": "2k_enceladus.jpg",
    "saturn_rings.png": "2k_saturn_ring_alpha.png"
}

# Map between texture name and normal map (if available)
NORMAL_MAPPING = {
    "earth.jpg": "2k_earth_normal_map.jpg",
    "moon.jpg": "2k_moon_normal_map.jpg",
    "mars.jpg": "2k_mars_normal_map.jpg"
}

# Map between texture name and specular map (if available)
SPECULAR_MAPPING = {
    "earth.jpg": "2k_earth_specular_map.jpg",
}

# Additional special textures
SPECIAL_TEXTURES = {
    "earth_clouds": "2k_earth_clouds.jpg",
    "earth_night": "2k_earth_nightmap.jpg",
    "jupiter_clouds": "2k_jupiter_clouds.jpg",
    "venus_atmosphere": "2k_venus_atmosphere.jpg"
}

def get_texture_filename(texture_name):
    """
    Get the actual filename to use for a texture.
    Checks both the original name and the mapped high-res name.
    
    Args:
        texture_name: The texture name requested by the simulation
        
    Returns:
        The actual filename to use
    """
    # First check if the original texture exists
    original_path = os.path.join(TEXTURE_DIR, texture_name)
    if os.path.exists(original_path):
        return texture_name
    
    # Then check if the mapped high-res texture exists
    if texture_name in TEXTURE_MAPPING:
        high_res_name = TEXTURE_MAPPING[texture_name]
        high_res_path = os.path.join(TEXTURE_DIR, high_res_name)
        if os.path.exists(high_res_path):
            return high_res_name
    
    # Return the original name as fallback (the texture loader will handle missing files)
    return texture_name

def get_normal_map(texture_name):
    """
    Get the normal map filename for a texture, if available.
    
    Args:
        texture_name: The texture name
        
    Returns:
        The normal map filename or None if not available
    """
    if texture_name in NORMAL_MAPPING:
        normal_map = NORMAL_MAPPING[texture_name]
        normal_path = os.path.join(TEXTURE_DIR, normal_map)
        if os.path.exists(normal_path):
            return normal_map
    return None

def get_specular_map(texture_name):
    """
    Get the specular map filename for a texture, if available.
    
    Args:
        texture_name: The texture name
        
    Returns:
        The specular map filename or None if not available
    """
    if texture_name in SPECULAR_MAPPING:
        specular_map = SPECULAR_MAPPING[texture_name]
        specular_path = os.path.join(TEXTURE_DIR, specular_map)
        if os.path.exists(specular_path):
            return specular_map
    return None

def get_special_texture(texture_key):
    """
    Get a special texture filename if available.
    
    Args:
        texture_key: The special texture key
        
    Returns:
        The special texture filename or None if not available
    """
    if texture_key in SPECIAL_TEXTURES:
        special_texture = SPECIAL_TEXTURES[texture_key]
        special_path = os.path.join(TEXTURE_DIR, special_texture)
        if os.path.exists(special_path):
            return special_texture
    return None

def convert_texture_names(body_list):
    """
    Updates texture names for a list of celestial bodies.
    Replaces the texture_name attribute with the actual filename to use.
    
    Args:
        body_list: A list of Body objects
        
    Returns:
        None (modifies the objects in-place)
    """
    for body in body_list:
        body.texture_name = get_texture_filename(body.texture_name)
        
        # Add normal map if available
        body.normal_map_name = get_normal_map(body.texture_name)
        
        # Add specular map if available
        body.specular_map_name = get_specular_map(body.texture_name)
        
        # Add special textures for specific bodies
        if body.name == "Earth":
            body.clouds_texture_name = get_special_texture("earth_clouds")
            body.night_texture_name = get_special_texture("earth_night")
        elif body.name == "Jupiter":
            body.clouds_texture_name = get_special_texture("jupiter_clouds")
