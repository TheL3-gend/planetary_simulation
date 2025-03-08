#!/usr/bin/env python3
# texture_loader.py - Efficient loading of high-resolution planet textures

import os
import sys
import numpy as np
import pygame
from OpenGL.GL import *
import traceback

class TextureLoader:
    """Manages loading and optimization of high-resolution planet textures"""
    
    def __init__(self, texture_dir):
        """Initialize the texture loader"""
        self.texture_dir = texture_dir
        self.textures = {}
        self.max_texture_size = self._get_max_texture_size()
        
        # Create texture directory if it doesn't exist
        if not os.path.exists(texture_dir):
            try:
                os.makedirs(texture_dir)
                print(f"Created texture directory: {texture_dir}")
            except OSError as e:
                print(f"Error creating texture directory: {e}")
                
        # Initialize pygame if not already done
        if not pygame.get_init():
            try:
                pygame.init()
            except Exception as e:
                print(f"Warning: Error initializing pygame: {e}")
                
    def _get_max_texture_size(self):
        """Get the maximum texture size supported by the GPU"""
        try:
            max_size = glGetIntegerv(GL_MAX_TEXTURE_SIZE)
            print(f"Maximum texture size supported: {max_size}x{max_size}")
            return max_size
        except Exception as e:
            print(f"Could not determine maximum texture size: {e}")
            # Default to a safe value
            return 4096
    
    def create_default_texture(self):
        """Create a default checkerboard texture"""
        try:
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
            
            return texture_id
        except Exception as e:
            print(f"Error creating default texture: {e}")
            traceback.print_exc()
            
            # Create an absolute minimum fallback texture
            try:
                texture_id = glGenTextures(1)
                glBindTexture(GL_TEXTURE_2D, texture_id)
                data = np.ones((4, 4, 4), dtype=np.uint8) * 255  # White texture
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 4, 4, 0, GL_RGBA, GL_UNSIGNED_BYTE, data)
                return texture_id
            except:
                print("Critical error creating fallback texture")
                return 0
    
    def load_texture(self, filename, mipmap=True, anisotropic=True):
        """Load a texture from file with optimizations for high-resolution textures"""
        # Return cached texture if already loaded
        if filename in self.textures:
            return self.textures[filename]
        
        # Create a default texture as fallback
        default_texture = self.create_default_texture()
        
        try:
            # Full path to texture file
            filepath = os.path.join(self.texture_dir, filename)
            
            # Check if file exists
            if not os.path.exists(filepath):
                print(f"Texture file not found: {filepath}")
                self.textures[filename] = default_texture
                return default_texture
            
            # Load image using pygame
            try:
                image = pygame.image.load(filepath)
            except pygame.error as e:
                print(f"Failed to load image {filepath}: {e}")
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
                print(f"Resizing texture {filename} from {width}x{height} to {new_width}x{new_height}")
                image = pygame.transform.smoothscale(image, (new_width, new_height))
                width, height = new_width, new_height
            
            # Check if dimensions are powers of two (optimal for mipmapping)
            width_pot = self._next_power_of_two(width)
            height_pot = self._next_power_of_two(height)
            
            # Resize to power of two if needed and not too much larger
            if (width != width_pot or height != height_pot) and \
               (width_pot <= 1.2 * width and height_pot <= 1.2 * height):
                print(f"Optimizing texture {filename} to power-of-two dimensions: {width_pot}x{height_pot}")
                image = pygame.transform.smoothscale(image, (width_pot, height_pot))
                width, height = width_pot, height_pot
            
            # Extract image data
            try:
                image_data = pygame.image.tostring(image, "RGBA", 1)
            except:
                print(f"Failed to convert image to string: {filepath}")
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
            if anisotropic and self._check_anisotropic_support():
                max_aniso = glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT)
                glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, max_aniso)
            
            # Upload texture data
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, image_data)
            
            # Generate mipmaps if requested
            if mipmap:
                glGenerateMipmap(GL_TEXTURE_2D)
            
            # Store and return the texture ID
            self.textures[filename] = texture_id
            return texture_id
            
        except Exception as e:
            print(f"Error loading texture {filename}: {e}")
            traceback.print_exc()
            
            # Use default texture as fallback
            self.textures[filename] = default_texture
            return default_texture
    
    def _check_anisotropic_support(self):
        """Check if anisotropic filtering is supported"""
        try:
            return GL_EXT_texture_filter_anisotropic in glGetString(GL_EXTENSIONS)
        except:
            return False
    
    def _next_power_of_two(self, x):
        """Get the next power of two greater than or equal to x"""
        return 1 if x == 0 else 2 ** (x - 1).bit_length()
    
    def cleanup(self):
        """Delete all textures"""
        try:
            # Convert to list to avoid modifying dictionary during iteration
            texture_ids = list(self.textures.values())
            
            for texture_id in texture_ids:
                if texture_id:
                    try:
                        glDeleteTextures(1, [texture_id])
                    except Exception as e:
                        print(f"Error deleting texture {texture_id}: {e}")
            
            self.textures.clear()
            print("All textures cleaned up")
        except Exception as e:
            print(f"Error during texture cleanup: {e}")

    @staticmethod
    def download_textures(base_url="https://www.solarsystemscope.com/textures/", output_dir=None):
        """
        Static helper method to download textures from solarsystemscope.com
        
        This is a placeholder - in a real implementation, you would need to:
        1. Get proper permission from the website
        2. Use appropriate download methods (requests, wget, etc.)
        3. Handle all potential download errors
        
        The solarsystemscope.com textures are licensed under CC BY 4.0,
        so proper attribution is required.
        """
        print("=== Texture Download Helper ===")
        print("The Solar System Scope textures are available at:")
        print("https://www.solarsystemscope.com/textures/")
        print("\nThese textures are licensed under CC BY 4.0, which requires attribution.")
        print("Please download them manually and place them in your textures directory.")
        print("\nRequired textures:")
        print("  - 2k_sun.jpg")
        print("  - 2k_mercury.jpg")
        print("  - 2k_venus_atmosphere.jpg")
        print("  - 2k_earth_daymap.jpg")
        print("  - 2k_mars.jpg")
        print("  - 2k_jupiter.jpg")
        print("  - 2k_saturn.jpg")
        print("  - 2k_uranus.jpg")
        print("  - 2k_neptune.jpg")
        print("  - 2k_moon.jpg")
        print("\nAfter downloading, you may need to rename the files to match")
        print("the names expected by the simulation (e.g., '2k_earth_daymap.jpg' to 'earth.jpg').")
        
        return False  # Indicate that automatic download is not implemented
