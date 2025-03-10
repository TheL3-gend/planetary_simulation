#!/usr/bin/env python3
# texture_mapping.py - Fixed version of texture mapping utilities

import os
import logging
from constants import TEXTURE_DIR

# Configure logging
logger = logging.getLogger("GravitySim.TextureMapping")

# Map between expected texture filenames and high-resolution filenames
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

# Map special texture types to filenames
NORMAL_MAPPING = {
    "earth.jpg": "2k_earth_normal_map.jpg",
    "moon.jpg": "2k_moon_normal_map.jpg",
    "mars.jpg": "2k_mars_normal_map.jpg"
}

SPECULAR_MAPPING = {
    "earth.jpg": "2k_earth_specular_map.jpg",
}

SPECIAL_TEXTURES = {
    "earth_clouds": "2k_earth_clouds.jpg",
    "earth_night": "2k_earth_nightmap.jpg",
    "jupiter_clouds": "2k_jupiter_clouds.jpg",
    "venus_atmosphere": "2k_venus_atmosphere.jpg"
}

def get_texture_filename(texture_name):
    """
    Get the appropriate filename for a texture, using high-res if available.
    
    Args:
        texture_name (str): The base texture name
        
    Returns:
        str: The filename to use (original or high-res)
    """
    if not texture_name:
        logger.warning("Empty texture name passed to get_texture_filename")
        return "sun.jpg"  # Default fallback
        
    # First check if the original texture exists
    original_path = os.path.join(TEXTURE_DIR, texture_name)
    if os.path.exists(original_path):
        return texture_name
    
    # Then check for high-res version
    if texture_name in TEXTURE_MAPPING:
        high_res_name = TEXTURE_MAPPING[texture_name]
        high_res_path = os.path.join(TEXTURE_DIR, high_res_name)
        if os.path.exists(high_res_path):
            return high_res_name
    
    # Return original as fallback
    return texture_name

def get_normal_map(base_texture):
    """Get normal map filename for a texture if available"""
    return get_special_texture_filename("normal", base_texture)
    
def get_specular_map(base_texture):
    """Get specular map filename for a texture if available"""
    return get_special_texture_filename("specular", base_texture)

def get_special_texture_filename(texture_type, base_texture):
    """
    Get filename for a special texture type (normal map, specular, etc.)
    
    Args:
        texture_type (str): The type of special texture
        base_texture (str): The base texture name
        
    Returns:
        str or None: The special texture filename or None if not available
    """
    if not base_texture:
        return None
        
    mapping = None
    if texture_type == "normal":
        mapping = NORMAL_MAPPING
    elif texture_type == "specular":
        mapping = SPECULAR_MAPPING
    elif texture_type.startswith("special_"):
        special_key = texture_type[8:]  # Remove "special_" prefix
        if special_key in SPECIAL_TEXTURES:
            special_texture = SPECIAL_TEXTURES[special_key]
            special_path = os.path.join(TEXTURE_DIR, special_texture)
            if os.path.exists(special_path):
                return special_texture
        return None
        
    if mapping and base_texture in mapping:
        special_texture = mapping[base_texture]
        special_path = os.path.join(TEXTURE_DIR, special_texture)
        if os.path.exists(special_path):
            return special_texture
            
    return None

def convert_texture_names(body_list):
    """
    Apply texture mappings to a list of celestial bodies
    
    Args:
        body_list (list): List of Body objects
    """
    if not body_list:
        return
    
    try:
        for body in body_list:
            # Skip bodies without texture names
            if not hasattr(body, 'texture_name') or not body.texture_name:
                continue
                
            # Set main texture to high-res version if available
            high_res_name = get_texture_filename(body.texture_name)
            if high_res_name != body.texture_name:
                logger.info(f"Using high-res texture for {body.name}: {high_res_name}")
                body.texture_name = high_res_name
            
            # Set special textures
            # Normal map
            normal_map = get_normal_map(body.texture_name)
            if normal_map:
                body.normal_map_name = normal_map
                
            # Specular map
            specular_map = get_specular_map(body.texture_name)
            if specular_map:
                body.specular_map_name = specular_map
            
            # Body-specific special textures
            if body.name == "Earth":
                earth_clouds = get_special_texture_filename("special_earth_clouds", "")
                if earth_clouds:
                    body.clouds_texture_name = earth_clouds
                    
                earth_night = get_special_texture_filename("special_earth_night", "")
                if earth_night:
                    body.night_texture_name = earth_night
                    
            elif body.name == "Jupiter":
                jupiter_clouds = get_special_texture_filename("special_jupiter_clouds", "")
                if jupiter_clouds:
                    body.clouds_texture_name = jupiter_clouds
                    
    except Exception as e:
        logger.error(f"Error converting texture names: {e}")