#!/usr/bin/env python3
# texture_mapping.py - Mapping for texture resources with improved reliability

import os
from constants import TEXTURE_DIR

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
    assert isinstance(texture_name, str), "Texture name must be a string"
    
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

def get_special_texture_filename(texture_type, base_texture):
    """
    Get filename for a special texture type (normal map, specular, etc.)
    
    Args:
        texture_type (str): The type of special texture
        base_texture (str): The base texture name
        
    Returns:
        str or None: The special texture filename or None if not available
    """
    assert isinstance(texture_type, str), "Texture type must be a string"
    assert isinstance(base_texture, str), "Base texture must be a string"
    
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

def check_textures(required_textures=None):
    """
    Check if required textures are available
    
    Args:
        required_textures (list): List of required texture names
        
    Returns:
        tuple: (has_all_textures, missing_textures, has_high_res)
    """
    if required_textures is None:
        # Default essential textures
        required_textures = [
            "sun.jpg", "mercury.jpg", "venus.jpg", "earth.jpg", 
            "mars.jpg", "jupiter.jpg", "saturn.jpg", "uranus.jpg", 
            "neptune.jpg", "moon.jpg"
        ]
    
    assert isinstance(required_textures, list), "Required textures must be a list"
    
    missing_textures = []
    has_high_res = False
    
    # Check each required texture
    for texture in required_textures:
        original_path = os.path.join(TEXTURE_DIR, texture)
        high_res_path = None
        
        if texture in TEXTURE_MAPPING:
            high_res_name = TEXTURE_MAPPING[texture]
            high_res_path = os.path.join(TEXTURE_DIR, high_res_name)
            if os.path.exists(high_res_path):
                has_high_res = True
        
        # Check if either version exists
        if not os.path.exists(original_path) and (high_res_path is None or not os.path.exists(high_res_path)):
            missing_textures.append(texture)
    
    return (len(missing_textures) == 0, missing_textures, has_high_res)

def apply_texture_mappings(body_list):
    """
    Apply texture mappings to a list of celestial bodies
    
    Args:
        body_list (list): List of Body objects
        
    Returns:
        bool: True if successful, False otherwise
    """
    assert isinstance(body_list, list), "Body list must be a list"
    if not body_list:
        return True  # Empty list is valid
    
    try:
        for body in body_list:
            # Set main texture
            body.texture_name = get_texture_filename(body.texture_name)
            
            # Set special textures
            body.normal_map_name = get_special_texture_filename("normal", body.texture_name)
            body.specular_map_name = get_special_texture_filename("specular", body.texture_name)
            
            # Add body-specific special textures
            if body.name == "Earth":
                body.clouds_texture_name = get_special_texture_filename("special_earth_clouds", "")
                body.night_texture_name = get_special_texture_filename("special_earth_night", "")
            elif body.name == "Jupiter":
                body.clouds_texture_name = get_special_texture_filename("special_jupiter_clouds", "")
                
        return True
    except Exception as e:
        print(f"Error applying texture mappings: {e}")
        return False