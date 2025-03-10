#!/usr/bin/env python3
# texture_loader.py - Safe texture loading with proper resource management

import os
import numpy as np
import pygame
from OpenGL.GL import *
import logging

# Configure logging
logger = logging.getLogger("TextureLoader")

class TextureLoader:
    """Manages loading and optimization of textures with proper resource management"""
    
    def __init__(self, texture_dir):
        """
        Initialize the texture loader
        
        Args:
            texture_dir (str): Directory containing texture files
        """
        assert isinstance(texture_dir, str), "Texture directory must be a string"
        
        self.texture_dir = texture_dir
        self.textures = {}  # Maps filenames to OpenGL texture IDs
        
        # Ensure texture directory exists
        if not os.path.exists(texture_dir):
            try:
                os.makedirs(texture_dir)
                logger.info(f"Created texture directory: {texture_dir}")
            except OSError as e:
                logger.error(f"Failed to create texture directory: {e}")
                
        # Get maximum texture size supported by GPU
        try:
            self.max_texture_size = glGetIntegerv(GL_MAX_TEXTURE_SIZE)
            logger.info(f"Maximum texture size: {self.max_texture_size}x{self.max_texture_size}")
        except Exception as e:
            logger.warning(f"Could not determine maximum texture size: {e}")
            self.max_texture_size = 4096  # Safe default
    
    def create_default_texture(self):
        """
        Create a default checkerboard texture
        
        Returns:
            int: OpenGL texture ID
        """
        texture_id = 0
        
        try:
            # Generate texture ID
            texture_id = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, texture_id)
            
            # Create checkerboard pattern
            size = 64  # Fixed size for default texture
            checkerboard = []
            
            for i in range(size):
                for j in range(size):
                    is_white = (i // 8 + j // 8) % 2
                    if is_white:
                        checkerboard.extend([255, 255, 255, 255])  # White
                    else:
                        checkerboard.extend([128, 128, 128, 255])  # Gray
            
            # Convert to uint8 byte array
            checkerboard = np.array(checkerboard, dtype=np.uint8)
            
            # Set texture parameters
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
            
            # Upload texture data
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, size, size, 0, GL_RGBA, GL_UNSIGNED_BYTE, checkerboard)
            glGenerateMipmap(GL_TEXTURE_2D)
            
            # Verify texture was created successfully
            if glIsTexture(texture_id) == GL_FALSE:
                logger.error(f"Failed to create default texture (invalid texture ID)")
                return 0
                
            return texture_id
            
        except Exception as e:
            logger.error(f"Error creating default texture: {e}")
            
            # Clean up on failure
            if texture_id != 0 and glIsTexture(texture_id) == GL_TRUE:
                glDeleteTextures(1, [texture_id])
                
            # Create absolute minimum fallback texture
            try:
                fallback_id = glGenTextures(1)
                glBindTexture(GL_TEXTURE_2D, fallback_id)
                data = np.ones((4, 4, 4), dtype=np.uint8) * 255  # White
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 4, 4, 0, GL_RGBA, GL_UNSIGNED_BYTE, data)
                return fallback_id
            except:
                logger.critical("Failed to create fallback texture")
                return 0
    
    def load_texture(self, filename, mipmap=True):
        """
        Load a texture from file
        
        Args:
            filename (str): Name of the texture file
            mipmap (bool): Whether to generate mipmaps
            
        Returns:
            int: OpenGL texture ID
        """
        assert isinstance(filename, str), "Filename must be a string"
        assert isinstance(mipmap, bool), "Mipmap flag must be a boolean"
        
        # Return cached texture if already loaded
        if filename in self.textures:
            texture_id = self.textures[filename]
            if glIsTexture(texture_id) == GL_TRUE:
                return texture_id
            else:
                # Remove invalid texture from cache
                logger.warning(f"Cached texture {filename} is invalid, reloading")
                del self.textures[filename]
        
        # Create default texture as fallback
        default_texture = self.create_default_texture()
        
        try:
            # Full path to texture file
            filepath = os.path.join(self.texture_dir, filename)
            
            # Check if file exists
            if not os.path.exists(filepath):
                logger.warning(f"Texture file not found: {filepath}")
                self.textures[filename] = default_texture
                return default_texture
            
            # Load image using pygame
            try:
                image = pygame.image.load(filepath)
            except pygame.error as e:
                logger.error(f"Failed to load image {filepath}: {e}")
                self.textures[filename] = default_texture
                return default_texture
                
            # Convert image to RGBA format if needed
            if image.get_bytesize() == 3:  # RGB format
                image = image.convert_alpha()
            
            # Get image dimensions
            width, height = image.get_size()
            
            # Check if image needs resizing (too large for GPU)
            max_dim = max(width, height)
            if max_dim > self.max_texture_size:
                scale_factor = self.max_texture_size / max_dim
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                logger.info(f"Resizing texture {filename} from {width}x{height} to {new_width}x{new_height}")
                image = pygame.transform.smoothscale(image, (new_width, new_height))
                width, height = new_width, new_height
            
            # Check if dimensions are powers of two (optimal for mipmapping)
            width_pot = self._next_power_of_two(width)
            height_pot = self._next_power_of_two(height)
            
            # Resize to power of two if needed and not too much larger
            if (width != width_pot or height != height_pot) and \
               (width_pot <= 1.2 * width and height_pot <= 1.2 * height):
                logger.info(f"Optimizing texture {filename} to {width_pot}x{height_pot}")
                image = pygame.transform.smoothscale(image, (width_pot, height_pot))
                width, height = width_pot, height_pot
            
            # Extract image data
            try:
                image_data = pygame.image.tostring(image, "RGBA", 1)
            except Exception as e:
                logger.error(f"Failed to convert image to string: {e}")
                self.textures[filename] = default_texture
                return default_texture
            
            # Generate OpenGL texture
            texture_id = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, texture_id)
            
            # Set texture parameters
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            
            if mipmap:
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
            else:
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
                
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
            
            # Enable anisotropic filtering if supported
            if self._check_anisotropic_support():
                try:
                    max_aniso = glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT)
                    max_aniso = min(max_aniso, 16.0)  # Limit to reasonable value
                    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, max_aniso)
                except Exception as e:
                    logger.debug(f"Failed to set anisotropic filtering: {e}")
            
            # Upload texture data
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, image_data)
            
            # Generate mipmaps if requested
            if mipmap:
                glGenerateMipmap(GL_TEXTURE_2D)
            
            # Verify texture was created successfully
            if glIsTexture(texture_id) == GL_FALSE:
                logger.error(f"Failed to create texture (invalid texture ID)")
                self.textures[filename] = default_texture
                return default_texture
            
            # Store and return the texture ID
            self.textures[filename] = texture_id
            return texture_id
            
        except Exception as e:
            logger.error(f"Error loading texture {filename}: {e}")
            
            # Use default texture as fallback
            self.textures[filename] = default_texture
            return default_texture
    
    def _check_anisotropic_support(self):
        """
        Check if anisotropic filtering is supported
        
        Returns:
            bool: True if supported, False otherwise
        """
        try:
            extensions = glGetString(GL_EXTENSIONS)
            if extensions is None:
                return False
                
            extensions = extensions.decode('utf-8').split()
            return 'GL_EXT_texture_filter_anisotropic' in extensions
        except Exception:
            return False
    
    def _next_power_of_two(self, x):
        """
        Get the next power of two greater than or equal to x
        
        Args:
            x (int): Input value
            
        Returns:
            int: Next power of two
        """
        assert isinstance(x, int) and x > 0, "Input must be a positive integer"
        return 1 if x == 0 else 2 ** (x - 1).bit_length()
    
    def cleanup(self):
        """Delete all textures and free resources"""
        # Copy keys to avoid modifying dictionary during iteration
        texture_ids = list(self.textures.values())
        
        for texture_id in texture_ids:
            if texture_id and glIsTexture(texture_id) == GL_TRUE:
                try:
                    glDeleteTextures(1, [texture_id])
                except Exception as e:
                    logger.warning(f"Error deleting texture {texture_id}: {e}")
        
        self.textures.clear()
        logger.info("All textures cleaned up")