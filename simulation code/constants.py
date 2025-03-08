#!/usr/bin/env python3
# constants.py - Shared constants for the gravity simulation

import os

# Display settings
WINDOW_WIDTH = 1920
WINDOW_HEIGHT = 1080
FULLSCREEN = False
MAX_FPS = 60
VSYNC = True

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEXTURE_DIR = os.path.join(BASE_DIR, "textures")
SHADER_DIR = os.path.join(BASE_DIR, "shaders")
FONT_DIR = os.path.join(BASE_DIR, "fonts")

# Physics constants
G = 6.67430e-11  # Gravitational constant (m^3 kg^-1 s^-2)
SCALE_FACTOR = 1e9  # Scale factor for visualization (1 unit = 1 billion meters)
COLLISION_DISTANCE = 5.0  # Distance at which bodies collide (scaled units)
DEFAULT_SIMULATION_SPEED = 3600 * 24  # Default simulation speed (1 day per frame)
MAX_SIMULATION_SPEED = 3600 * 24 * 30  # Maximum simulation speed (1 month per frame)
MIN_SIMULATION_SPEED = 3600  # Minimum simulation speed (1 hour per frame)

# Rendering settings
SPHERE_DETAIL = 32  # Number of segments for sphere rendering
MAX_TRAIL_LENGTH = 500  # Maximum number of points in orbital trails
TRAIL_UPDATE_FREQUENCY = 5  # Update trail every N frames
SHOW_TRAILS = True  # Show orbital trails
SHOW_LABELS = True  # Show planet labels
SHOW_AXES = True  # Show coordinate axes
ANTI_ALIASING = True  # Enable anti-aliasing
MSAA_SAMPLES = 4  # Multi-sample anti-aliasing samples

# Solar system data (name, mass (kg), radius (km), distance from sun (km), color, texture, has_rings)
PLANET_DATA = [
    ("Sun", 1.989e30, 695700, 0, (1.0, 0.7, 0.0), "sun.jpg", False),
    ("Mercury", 3.301e23, 2440, 57.9e6, (0.7, 0.7, 0.7), "mercury.jpg", False),
    ("Venus", 4.867e24, 6052, 108.2e6, (0.9, 0.7, 0.0), "venus.jpg", False),
    ("Earth", 5.972e24, 6371, 149.6e6, (0.0, 0.5, 1.0), "earth.jpg", False),
    ("Mars", 6.417e23, 3390, 227.9e6, (1.0, 0.0, 0.0), "mars.jpg", False),
    ("Jupiter", 1.898e27, 69911, 778.5e6, (0.8, 0.6, 0.0), "jupiter.jpg", True),
    ("Saturn", 5.683e26, 58232, 1434.0e6, (0.9, 0.7, 0.2), "saturn.jpg", True),
    ("Uranus", 8.681e25, 25362, 2871.0e6, (0.5, 0.7, 0.9), "uranus.jpg", True),
    ("Neptune", 1.024e26, 24622, 4495.0e6, (0.0, 0.0, 0.8), "neptune.jpg", True),
]

# Other bodies (name, mass (kg), radius (km), parent, distance (km), color, texture)
MOON_DATA = [
    ("Moon", 7.342e22, 1737, "Earth", 384400, (0.8, 0.8, 0.8), "moon.jpg"),
    ("Phobos", 1.066e16, 11, "Mars", 9376, (0.6, 0.6, 0.6), "phobos.jpg"),
    ("Deimos", 1.48e15, 6, "Mars", 23463, (0.6, 0.6, 0.6), "deimos.jpg"),
    ("Io", 8.932e22, 1822, "Jupiter", 421700, (1.0, 0.9, 0.4), "io.jpg"),
    ("Europa", 4.8e22, 1561, "Jupiter", 671034, (0.9, 0.9, 0.9), "europa.jpg"),
    ("Ganymede", 1.4819e23, 2634, "Jupiter", 1070412, (0.7, 0.7, 0.7), "ganymede.jpg"),
    ("Callisto", 1.076e23, 2410, "Jupiter", 1882709, (0.5, 0.5, 0.5), "callisto.jpg"),
    ("Titan", 1.3455e23, 2575, "Saturn", 1221870, (0.9, 0.7, 0.5), "titan.jpg"),
    ("Enceladus", 1.08e20, 252, "Saturn", 237948, (1.0, 1.0, 1.0), "enceladus.jpg"),
]

# UI settings
UI_FONT = "Arial"
UI_FONT_SIZE = 16
UI_COLOR = (255, 255, 255)
UI_BACKGROUND = (0, 0, 0, 128)
UI_PADDING = 10
UI_LABEL_FONT_SIZE = 12
UI_TITLE_FONT_SIZE = 24