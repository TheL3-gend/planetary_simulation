#!/usr/bin/env python3
# texture_loader.py - Robust texture loading with proper error handling

import os
import numpy as np
import pygame
from OpenGL.GL import *
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("TextureLoader")

class TextureLoader:
    """Manages loading and optimization of textures with proper resource management"""
    
    def __init__(self, texture_dir):
        """
        Initialize the texture loader
        
        Args:
            texture_dir (str): Directory containing texture files
        """
        self.texture_dir = texture_dir
        self.textures = {}  # Maps filenames to OpenGL texture IDs
        self.default_texture_id = None  # Will be created on first use
        
        # Ensure texture directory exists
        if not os.path.exists(texture_dir):
            try:
                os.makedirs(texture_dir)
                logger.info(f"Created texture directory: {texture_dir}")
            except OSError as e:
                logger.error(f"Failed to create texture directory: {e}")
                
        # Initialize pygame if not already done
        if not pygame.get_init():
            try:
                pygame.init()
            except Exception as e:
                logger.warning(f"Failed to initialize pygame: {e}")
        
        # Get max texture size
        try:
            self.max_texture_size = glGetIntegerv(GL_MAX_TEXTURE_SIZE)
            logger.info(f"Max texture size: {self.max_texture_size}")
        except Exception:
            # Default to a safe value
            self.max_texture_size = 2048
            logger.warning(f"Could not determine max texture size, using {self.max_texture_size}")
    
    def get_default_texture(self):
        """
        Create or return the default checkerboard texture
        
        Returns:
            int: OpenGL texture ID
        """
        # Return cached default texture if it exists
        if self.default_texture_id is not None and glIsTexture(self.default_texture_id):
            return self.default_texture_id
            
        try:
            # Generate new texture
            texture_id = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, texture_id)
            
            # Create a simple checkerboard pattern
            size = 64
            checkerboard = []
            for i in range(size):
                for j in range(size):
                    if (i // 8 + j // 8) % 2:
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
            
            # Store the ID for future use
            self.default_texture_id = texture_id
            return texture_id
        except Exception as e:
            logger.error(f"Error creating default texture: {e}")
            
            # Create minimal fallback texture
            try:
                fallback_id = glGenTextures(1)
                glBindTexture(GL_TEXTURE_2D, fallback_id)
                data = np.ones((4, 4, 4), dtype=np.uint8) * 255  # White
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 4, 4, 0, GL_RGBA, GL_UNSIGNED_BYTE, data)
                self.default_texture_id = fallback_id
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
        # Validate input
        if not isinstance(filename, str) or not filename:
            logger.error("Invalid filename")
            return self.get_default_texture()
            
        # Return cached texture if already loaded
        if filename in self.textures:
            texture_id = self.textures[filename]
            if glIsTexture(texture_id):
                return texture_id
            else:
                # Remove invalid texture from cache
                logger.warning(f"Cached texture {filename} is invalid, reloading")
                del self.textures[filename]
        
        try:
            # Full path to texture file
            filepath = os.path.join(self.texture_dir, filename)
            
            # Check if file exists
            if not os.path.exists(filepath):
                logger.warning(f"Texture file not found: {filepath}")
                return self.get_default_texture()
            
            # Load image using pygame
            try:
                image = pygame.image.load(filepath)
            except pygame.error as e:
                logger.error(f"Failed to load image {filepath}: {e}")
                return self.get_default_texture()
                
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
                logger.info(f"Resizing texture {filename} to {new_width}x{new_height}")
                image = pygame.transform.smoothscale(image, (new_width, new_height))
                width, height = new_width, new_height
            
            # Extract image data
            try:
                image_data = pygame.image.tostring(image, "RGBA", 1)
            except Exception as e:
                logger.error(f"Failed to convert image to string: {e}")
                return self.get_default_texture()
            
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
            
            # Try to enable anisotropic filtering if supported
            try:
                if GL_EXT_texture_filter_anisotropic:
                    max_aniso = glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT)
                    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, max_aniso)
            except:
                pass  # Not supported, ignore
            
            # Upload texture data
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, image_data)
            
            # Generate mipmaps if requested
            if mipmap:
                glGenerateMipmap(GL_TEXTURE_2D)
            
            # Store and return the texture ID
            self.textures[filename] = texture_id
            logger.info(f"Loaded texture: {filename} ({width}x{height})")
            return texture_id
            
        except Exception as e:
            logger.error(f"Error loading texture {filename}: {e}")
            return self.get_default_texture()
    
    def cleanup(self):
        """Delete all textures and free resources"""
        # Get list of texture IDs first to avoid modifying dict during iteration
        texture_ids = list(self.textures.values())
        
        # Add default texture if it exists
        if self.default_texture_id is not None:
            texture_ids.append(self.default_texture_id)
        
        # Delete all textures
        for texture_id in texture_ids:
            if texture_id and glIsTexture(texture_id):
                try:
                    glDeleteTextures(1, [texture_id])
                except Exception as e:
                    logger.warning(f"Error deleting texture {texture_id}: {e}")
        
        # Clear maps
        self.textures.clear()
        self.default_texture_id = None
        logger.info("All textures cleaned up")