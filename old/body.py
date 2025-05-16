#!/usr/bin/env python3
# body.py - Celestial body class for gravity simulation

import numpy as np
import math
from constants import MAX_TRAIL_LENGTH, SCALE_FACTOR

class Body:
    """Represents a celestial body in the simulation"""

    def __init__(self, name, mass, radius, position, velocity, color, texture_name, has_rings=False):
        """Initialize a celestial body

        Args:
            name (str): Name of the celestial body
            mass (float): Mass in kg
            radius (float): Radius in km
            position (list/np.array): Initial position [x, y, z] in m
            velocity (list/np.array): Initial velocity [vx, vy, vz] in m/s
            color (tuple): RGB color (0.0-1.0)
            texture_name (str): Filename of texture
            has_rings (bool): Whether the body has rings
        """
        self.name = name
        self.mass = mass
        self.radius = radius * 1000  # Convert km to m
        self.position = np.array(position, dtype=np.float64)
        self.velocity = np.array(velocity, dtype=np.float64)
        self.forces = np.zeros(3, dtype=np.float64)
        self.color = color
        self.texture_name = texture_name
        self.texture_id = None  # Set by the renderer
        self.has_rings = has_rings

        # Visual properties
        self.visual_radius = max(0.1, math.log10(self.radius / 1000) / 2)  # Logarithmic scale for visualization
        if has_rings:
            self.ring_inner_radius = 0.6
            self.ring_outer_radius = 1.0

        # Physics properties
        self.orbital_period = 0.0  # Set by simulation

        # Trail for orbital visualization
        self.trail = []
        self.update_trail()  # Initialize trail with current position

        # Additional textures for advanced rendering
        self.normal_map_name = None
        self.specular_map_name = None
        self.clouds_texture_name = None
        self.night_texture_name = None

    def update_velocity(self, dt):
        """Update velocity based on current forces and time step

        Args:
            dt (float): Time step in seconds
        """
        # Check for invalid forces
        if not np.all(np.isfinite(self.forces)):
            return

        # a = F/m
        acceleration = self.forces / self.mass

        # v = v0 + a*dt
        self.velocity += acceleration * dt

    def update_position(self, dt):
        """Update position based on current velocity and time step

        Args:
            dt (float): Time step in seconds
        """
        # Check for invalid velocity
        if not np.all(np.isfinite(self.velocity)):
            return

        # p = p0 + v*dt
        self.position += self.velocity * dt

    def add_force(self, force):
        """Add a force to the body

        Args:
            force (np.array): Force vector to add
        """
        # Check for invalid force
        if not np.all(np.isfinite(force)):
            return

        self.forces += force

    def update_trail(self):
        """Update orbital trail with current position"""
        # Add current position to trail
        self.trail.append(np.copy(self.position))

        # Keep trail at maximum length
        if len(self.trail) > MAX_TRAIL_LENGTH:
            self.trail.pop(0)

    def get_info_text(self):
        """Get formatted information about the body

        Returns:
            list: Lines of information text
        """
        # Convert units for display
        pos_km = self.position / 1000  # m to km
        vel_km = self.velocity / 1000  # m/s to km/s

        info = [
            f"Mass: {self.mass:.2e} kg",
            f"Radius: {self.radius/1000:.1f} km",
            f"Position: ({pos_km[0]:.2e}, {pos_km[1]:.2e}, {pos_km[2]:.2e}) km",
            f"Velocity: ({vel_km[0]:.2f}, {vel_km[1]:.2f}, {vel_km[2]:.2f}) km/s",
            f"Speed: {np.linalg.norm(vel_km):.2f} km/s"
        ]

        # Add orbital period if available
        if self.orbital_period > 0:
            days = self.orbital_period / (24 * 3600)
            if days > 365:
                info.append(f"Orbital Period: {days/365.25:.2f} years")
            else:
                info.append(f"Orbital Period: {days:.2f} days")

        return info