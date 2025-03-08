#!/usr/bin/env python3
# create_textures.py - Generate default planet textures


import os
import sys
import numpy as np
import pygame
from constants import TEXTURE_DIR, PLANET_DATA, MOON_DATA

def create_sun_texture(size=1024):
    """Create a procedural sun texture"""
    surface = pygame.Surface((size, size), pygame.SRCALPHA)
    
    # Fill with base orange color
    surface.fill((255, 170, 0))
    
    # Add some noise for texture
    noise = np.random.normal(0, 0.1, (size, size))
    
    # Apply noise and create solar features
    pixels = pygame.surfarray.pixels3d(surface)
    
    # Create granulation pattern
    for y in range(size):
        for x in range(size):
            # Distance from center (0 to 1)
            dx = x / size - 0.5
            dy = y / size - 0.5
            dist = 2 * np.sqrt(dx*dx + dy*dy)
            
            # Limb darkening (edges are darker)
            limb = 1.0 - 0.5 * dist**2
            
            # Add noise
            noise_val = noise[y, x]
            
            # Base color
            r, g, b = pixels[x, y]
            
            # Adjust color
            r = min(255, max(0, int(r * limb + noise_val * 20)))
            g = min(255, max(0, int(g * limb + noise_val * 15)))
            b = min(255, max(0, int((b * 0.7 + 30) * limb + noise_val * 10)))
            
            pixels[x, y] = [r, g, b]
    
    # Create some solar flares
    for _ in range(20):
        x = np.random.randint(0, size)
        y = np.random.randint(0, size)
        radius = np.random.randint(5, 20)
        
        pygame.draw.circle(surface, (255, 220, 50), (x, y), radius)
    
    del pixels
    return surface

def create_earth_texture(size=1024):
    """Create a procedural Earth texture"""
    surface = pygame.Surface((size, size), pygame.SRCALPHA)
    
    # Fill with base blue color (ocean)
    surface.fill((0, 80, 180))
    
    # Add continents
    pixels = pygame.surfarray.pixels3d(surface)
    
    # Create Perlin-like noise for continents
    scale = 4
    octaves = 6
    persistence = 0.5
    
    def noise2d(x, y):
        n = 0
        amplitude = 1.0
        frequency = 1.0
        max_value = 0.0
        
        for _ in range(octaves):
            n += perlin(x * frequency, y * frequency) * amplitude
            max_value += amplitude
            amplitude *= persistence
            frequency *= 2
        
        return n / max_value
    
    def perlin(x, y):
        # Simple gradient noise approximation
        x0 = int(x)
        y0 = int(y)
        x1 = x0 + 1
        y1 = y0 + 1
        
        dx = x - x0
        dy = y - y0
        
        def dot_grid_gradient(ix, iy, x, y):
            # Random gradient
            random = 2920.0 * np.sin(ix * 21942.0 + iy * 171324.0 + 8912.0) * np.cos(ix * 23157.0 * iy * 217832.0 + 9758.0)
            angle = random * 2 * np.pi
            return (x - ix) * np.cos(angle) + (y - iy) * np.sin(angle)
        
        # Interpolate
        def smoothstep(t):
            return t * t * (3 - 2 * t)
        
        sx = smoothstep(dx)
        sy = smoothstep(dy)
        
        n0 = dot_grid_gradient(x0, y0, x, y)
        n1 = dot_grid_gradient(x1, y0, x, y)
        ix0 = n0 + sx * (n1 - n0)
        
        n0 = dot_grid_gradient(x0, y1, x, y)
        n1 = dot_grid_gradient(x1, y1, x, y)
        ix1 = n0 + sx * (n1 - n0)
        
        return ix0 + sy * (ix1 - ix0)
    
    # Generate continent mask
    for y in range(size):
        for x in range(size):
            # Normalized coordinates
            nx = x / size
            ny = y / size
            
            # Noise value
            n = noise2d(nx * scale, ny * scale)
            
            # Only create land where noise is above threshold
            if n > 0.5:
                # Land color (green to brown based on "elevation")
                elevation = (n - 0.5) * 2  # 0 to 1
                
                if elevation < 0.2:
                    # Shallow coastal areas
                    r, g, b = 200, 220, 120
                elif elevation < 0.6:
                    # Forests and plains
                    r, g, b = 30, 150, 30
                else:
                    # Mountains
                    r = int(150 + elevation * 100)
                    g = int(150 + elevation * 50)
                    b = int(150 + elevation * 30)
                    
                pixels[x, y] = [r, g, b]
            
            # Add ice caps at poles
            pole_y = abs(ny - 0.5) * 2  # 0 at equator, 1 at poles
            if pole_y > 0.8:
                ice_factor = (pole_y - 0.8) * 5  # 0 to 1
                r, g, b = pixels[x, y]
                r = int(r * (1 - ice_factor) + 255 * ice_factor)
                g = int(g * (1 - ice_factor) + 255 * ice_factor)
                b = int(b * (1 - ice_factor) + 255 * ice_factor)
                pixels[x, y] = [r, g, b]
    
    del pixels
    return surface

def create_mars_texture(size=1024):
    """Create a procedural Mars texture"""
    surface = pygame.Surface((size, size), pygame.SRCALPHA)
    
    # Fill with base red color
    surface.fill((180, 80, 30))
    
    # Add terrain details
    pixels = pygame.surfarray.pixels3d(surface)
    
    # Create noise pattern
    noise = np.random.normal(0, 0.1, (size, size))
    
    for y in range(size):
        for x in range(size):
            # Normalized coordinates
            nx = x / size
            ny = y / size
            
            # Distance from center
            dx = nx - 0.5
            dy = ny - 0.5
            dist = np.sqrt(dx*dx + dy*dy) * 2  # 0 to 1
            
            # Base color with noise
            r, g, b = pixels[x, y]
            noise_val = noise[y, x]
            
            # Adjust color based on noise and position
            r = min(255, max(0, int(r + noise_val * 30)))
            g = min(255, max(0, int(g + noise_val * 15)))
            b = min(255, max(0, int(b + noise_val * 10)))
            
            pixels[x, y] = [r, g, b]
            
            # Add some darker areas (maria)
            if (nx * 5) % 1 < 0.5 and (ny * 5) % 1 < 0.5 and noise_val < -0.05:
                pixels[x, y] = [r - 40, g - 20, b - 10]
            
            # Add polar caps
            pole_y = abs(ny - 0.5) * 2  # 0 at equator, 1 at poles
            if pole_y > 0.85:
                ice_factor = (pole_y - 0.85) * 6.66  # 0 to 1
                r, g, b = pixels[x, y]
                r = int(r * (1 - ice_factor) + 255 * ice_factor)
                g = int(g * (1 - ice_factor) + 255 * ice_factor)
                b = int(b * (1 - ice_factor) + 255 * ice_factor)
                pixels[x, y] = [r, g, b]
    
    del pixels
    return surface

def create_gas_giant_texture(size=1024, base_color=(200, 150, 50)):
    """Create a procedural gas giant texture with bands"""
    surface = pygame.Surface((size, size), pygame.SRCALPHA)
    
    # Fill with base color
    surface.fill(base_color)
    
    # Add banded structure
    pixels = pygame.surfarray.pixels3d(surface)
    
    # Create noise pattern
    noise = np.random.normal(0, 0.1, (size, size))
    
    # Base color
    r_base, g_base, b_base = base_color
    
    # Create horizontal bands
    num_bands = np.random.randint(10, 20)
    
    for y in range(size):
        # Normalized vertical position
        ny = y / size
        
        # Band pattern
        band_value = np.sin(ny * num_bands * np.pi) * 0.5 + 0.5  # 0 to 1
        
        for x in range(size):
            # Add noise to band pattern
            noise_val = noise[y, x]
            local_band = min(1.0, max(0.0, band_value + noise_val * 0.3))
            
            # Darken or lighten base color based on band
            r = int(r_base * (0.7 + 0.3 * local_band))
            g = int(g_base * (0.7 + 0.3 * local_band))
            b = int(b_base * (0.7 + 0.3 * local_band))
            
            # Add some spot features
            if np.random.random() < 0.0001:
                spot_size = np.random.randint(5, 15)
                center_x, center_y = x, y
                
                for sy in range(max(0, y - spot_size), min(size, y + spot_size)):
                    for sx in range(max(0, x - spot_size), min(size, x + spot_size)):
                        # Distance from spot center
                        spot_dist = np.sqrt((sx - center_x)**2 + (sy - center_y)**2)
                        if spot_dist < spot_size:
                            # Calculate spot effect
                            effect = 1.0 - spot_dist / spot_size
                            
                            # Either darker or lighter spot
                            if np.random.random() < 0.5:
                                spot_factor = 1.5  # Lighter
                            else:
                                spot_factor = 0.5  # Darker
                                
                            # Apply to this pixel
                            if 0 <= sx < size and 0 <= sy < size:
                                curr_r, curr_g, curr_b = pixels[sx, sy]
                                pixels[sx, sy] = [
                                    int(curr_r * (1.0 - effect) + curr_r * spot_factor * effect),
                                    int(curr_g * (1.0 - effect) + curr_g * spot_factor * effect),
                                    int(curr_b * (1.0 - effect) + curr_b * spot_factor * effect)
                                ]
            
            pixels[x, y] = [r, g, b]
    
    del pixels
    return surface

def create_moon_texture(size=512, is_icy=False):
    """Create a procedural moon texture"""
    surface = pygame.Surface((size, size), pygame.SRCALPHA)
    
    # Base color depends on type
    if is_icy:
        surface.fill((220, 220, 230))  # Icy moon
    else:
        surface.fill((150, 150, 150))  # Rocky moon
    
    # Add crater details
    pixels = pygame.surfarray.pixels3d(surface)
    
    # Create noise pattern
    noise = np.random.normal(0, 0.1, (size, size))
    
    for y in range(size):
        for x in range(size):
            # Apply noise to base color
            r, g, b = pixels[x, y]
            noise_val = noise[y, x]
            
            r = min(255, max(0, int(r + noise_val * 20)))
            g = min(255, max(0, int(g + noise_val * 20)))
            b = min(255, max(0, int(b + noise_val * 20)))
            
            pixels[x, y] = [r, g, b]
    
    # Add craters
    num_craters = np.random.randint(50, 200)
    
    for _ in range(num_craters):
        # Random crater position and size
        cx = np.random.randint(0, size)
        cy = np.random.randint(0, size)
        radius = np.random.randint(2, 20)
        
        # Draw crater
        for y in range(max(0, cy - radius), min(size, cy + radius)):
            for x in range(max(0, cx - radius), min(size, cx + radius)):
                # Distance from crater center
                dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                
                if dist < radius:
                    # Crater rim effect
                    rim_effect = abs(dist - radius * 0.8) / (radius * 0.2)
                    rim_effect = min(1.0, max(0.0, rim_effect))
                    
                    # Center is darker, rim is lighter
                    if dist < radius * 0.8:
                        factor = 0.8  # Darker center
                    else:
                        factor = 1.2  # Lighter rim
                    
                    # Apply to this pixel
                    r, g, b = pixels[x, y]
                    pixels[x, y] = [
                        min(255, int(r * factor)),
                        min(255, int(g * factor)),
                        min(255, int(b * factor))
                    ]
    
    del pixels
    return surface

def create_saturn_rings_texture(size=1024):
    """Create a procedural texture for Saturn's rings"""
    surface = pygame.Surface((size, size), pygame.SRCALPHA)
    
    # Start with transparent background
    surface.fill((0, 0, 0, 0))
    
    # Draw rings as concentric circles with varying opacity
    center = (size // 2, size // 2)
    inner_radius = int(size * 0.3)
    outer_radius = int(size * 0.5)
    
    for r in range(inner_radius, outer_radius):
        # Calculate ring properties based on radius
        ring_pos = (r - inner_radius) / (outer_radius - inner_radius)
        
        # Base ring color
        ring_color = (200, 180, 150)
        
        # Vary opacity based on position and some noise
        noise = np.sin(ring_pos * 20) * 0.2 + np.random.random() * 0.1
        opacity = int(200 * (0.6 + noise))
        
        # Add gaps in the rings
        if (ring_pos > 0.3 and ring_pos < 0.4) or (ring_pos > 0.7 and ring_pos < 0.72):
            opacity = int(opacity * 0.3)
        
        # Draw ring circle
        pygame.draw.circle(surface, (*ring_color, opacity), center, r, 1)
    
    return surface

def generate_all_textures():
    """Generate all required textures"""
    # Create texture directory if it doesn't exist
    if not os.path.exists(TEXTURE_DIR):
        os.makedirs(TEXTURE_DIR)
    
    # Generate planet textures
    print("Generating planet textures...")
    
    # Sun
    print("Creating sun.jpg")
    pygame.image.save(create_sun_texture(), os.path.join(TEXTURE_DIR, "sun.jpg"))
    
    # Earth
    print("Creating earth.jpg")
    pygame.image.save(create_earth_texture(), os.path.join(TEXTURE_DIR, "earth.jpg"))
    
    # Mars
    print("Creating mars.jpg")
    pygame.image.save(create_mars_texture(), os.path.join(TEXTURE_DIR, "mars.jpg"))
    
    # Mercury (rocky)
    print("Creating mercury.jpg")
    mercury_texture = create_moon_texture(1024, False)
    # Make it more reddish
    pixels = pygame.surfarray.pixels3d(mercury_texture)
    pixels[:,:,0] = np.minimum(255, pixels[:,:,0] * 1.2)  # Increase red
    pixels[:,:,1] = np.minimum(255, pixels[:,:,1] * 0.9)  # Decrease green
    pixels[:,:,2] = np.minimum(255, pixels[:,:,2] * 0.8)  # Decrease blue
    del pixels
    pygame.image.save(mercury_texture, os.path.join(TEXTURE_DIR, "mercury.jpg"))
    
    # Venus (yellowish)
    print("Creating venus.jpg")
    venus_texture = pygame.Surface((1024, 1024), pygame.SRCALPHA)
    venus_texture.fill((230, 220, 130))
    # Add swirly cloud patterns
    pixels = pygame.surfarray.pixels3d(venus_texture)
    for y in range(1024):
        for x in range(1024):
            nx, ny = x/1024, y/1024
            swirl = np.sin(nx*10 + ny*10 + np.sin(nx*5)*3 + np.sin(ny*5)*2) * 0.5 + 0.5
            pixels[x,y,0] = min(255, int(pixels[x,y,0] * (0.8 + swirl * 0.2)))
            pixels[x,y,1] = min(255, int(pixels[x,y,1] * (0.8 + swirl * 0.2)))
            pixels[x,y,2] = min(255, int(pixels[x,y,2] * (0.8 + swirl * 0.2)))
    del pixels
    pygame.image.save(venus_texture, os.path.join(TEXTURE_DIR, "venus.jpg"))
    
    # Jupiter
    print("Creating jupiter.jpg")
    pygame.image.save(create_gas_giant_texture(1024, (200, 160, 110)), 
                      os.path.join(TEXTURE_DIR, "jupiter.jpg"))
    
    # Saturn
    print("Creating saturn.jpg")
    pygame.image.save(create_gas_giant_texture(1024, (220, 190, 130)), 
                      os.path.join(TEXTURE_DIR, "saturn.jpg"))
    
    # Uranus
    print("Creating uranus.jpg")
    pygame.image.save(create_gas_giant_texture(1024, (150, 190, 220)), 
                      os.path.join(TEXTURE_DIR, "uranus.jpg"))
    
    # Neptune
    print("Creating neptune.jpg")
    pygame.image.save(create_gas_giant_texture(1024, (100, 120, 210)), 
                      os.path.join(TEXTURE_DIR, "neptune.jpg"))
    
    # Moon
    print("Creating moon.jpg")
    pygame.image.save(create_moon_texture(512, False), os.path.join(TEXTURE_DIR, "moon.jpg"))
    
    # Other moons
    print("Creating moon textures...")
    for name, _, _, _, _, _, texture in MOON_DATA:
        if not os.path.exists(os.path.join(TEXTURE_DIR, texture)):
            print(f"Creating {texture}")
            # Determine if icy based on name
            is_icy = any(term in name.lower() for term in ["europa", "enceladus", "ganymede"])
            pygame.image.save(create_moon_texture(512, is_icy), 
                            os.path.join(TEXTURE_DIR, texture))
    
    # Saturn rings
    print("Creating saturn_rings.png")
    pygame.image.save(create_saturn_rings_texture(), 
                     os.path.join(TEXTURE_DIR, "saturn_rings.png"))
    
    print("All textures generated.")

if __name__ == "__main__":
    # Initialize pygame
    pygame.init()
    
    # Set a small display mode to initialize display system
    pygame.display.set_mode((1, 1), pygame.NOFRAME)
    
    try:
        generate_all_textures()
        print(f"Textures saved to {TEXTURE_DIR}")
    except Exception as e:
        print(f"Error generating textures: {e}")
    
    pygame.quit()