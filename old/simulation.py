#!/usr/bin/env python3
# simulation.py - Simulation and physics calculations (FURTHER REFINED)

import numpy as np
import math
import time
import random
from pygame.locals import *
from body import Body
from constants import *

class Simulation:
    """Handles the physics simulation, collision detection, and body management"""

    def __init__(self):
        """Initialize the physics simulation"""
        self.bodies = []
        self.paused = False
        self.simulation_speed = DEFAULT_SIMULATION_SPEED
        self.selected_body = None
        self.last_update_time = time.time()
        self.frame_count = 0
        self.fps = 0.0
        self.selected_info = []  # Now used for body info ONLY
        self.time_elapsed = 0.0  # Simulation time in seconds
        self.trail_update_counter = 0
        self.show_trails = True  # Control trail visibility
        self.show_labels = True  # Control label visibility


        # Create the solar system
        try:
            self.create_solar_system()
        except Exception as e:
            print(f"Error creating solar system: {e}")
            # Fallback to just creating the sun
            self.create_minimal_system()

    def create_minimal_system(self):
        """Create a minimal system with just the sun as a fallback"""
        sun_data = next((data for data in PLANET_DATA if data[0] == "Sun"), None)
        if sun_data:
            name, mass, radius, distance, color, texture, has_rings = sun_data
            sun = Body(name, mass, radius, [0, 0, 0], [0, 0, 0], color, texture, has_rings)
            self.bodies.append(sun)
            self.selected_body = sun
        else:
            # Ultimate fallback
            sun = Body("Sun", 1.989e30, 695700, [0, 0, 0], [0, 0, 0], (1.0, 0.7, 0.0), "sun.jpg", False)
            self.bodies.append(sun)
            self.selected_body = sun

    def create_solar_system(self):
        """Create the solar system with all planets and major moons"""
        # Add the sun and planets from PLANET_DATA
        planet_objects = {}  # To keep track of planets for moon placement

        for name, mass, radius, distance, color, texture, has_rings in PLANET_DATA:
            # Convert distance from km to m
            distance_m = distance * 1000

            if distance_m == 0:  # The Sun
                position = [0, 0, 0]
                velocity = [0, 0, 0]
            else:
                # Calculate orbital velocity for circular orbit
                # v = sqrt(G * M / r) where M is the sun's mass
                orbital_speed = math.sqrt(G * PLANET_DATA[0][1] / distance_m)

                # Random starting angle
                angle = random.uniform(0, 2 * math.pi)

                # Initial position (x, y, z)
                position = [
                    distance_m * math.cos(angle),
                    0,  # All planets roughly on the ecliptic plane
                    distance_m * math.sin(angle)
                ]

                # Initial velocity perpendicular to position for circular orbit
                velocity = [
                    -orbital_speed * math.sin(angle),
                    0,
                    orbital_speed * math.cos(angle)
                ]

            # Create the body
            body = Body(name, mass, radius, position, velocity, color, texture, has_rings)

            # Add to the simulation
            self.bodies.append(body)

            # Store in planet_objects dict for moons
            planet_objects[name] = body

            # Set the orbital period
            if distance_m > 0:
                body.orbital_period = 2 * math.pi * math.sqrt(distance_m**3 / (G * PLANET_DATA[0][1]))

        # Add moons
        for name, mass, radius, parent_name, distance, color, texture in MOON_DATA:
            if parent_name in planet_objects:
                parent = planet_objects[parent_name]

                # Convert distance from km to m
                distance_m = distance * 1000

                # Calculate orbital velocity for circular orbit around parent
                orbital_speed = math.sqrt(G * parent.mass / distance_m)

                # Random starting angle
                angle = random.uniform(0, 2 * math.pi)

                # Position relative to parent
                rel_position = [
                    distance_m * math.cos(angle),
                    0,
                    distance_m * math.sin(angle)
                ]

                # Initial velocity relative to parent
                rel_velocity = [
                    -orbital_speed * math.sin(angle),
                    0,
                    orbital_speed * math.cos(angle)
                ]

                # Absolute position and velocity (parent + relative)
                position = parent.position + np.array(rel_position)
                velocity = parent.velocity + np.array(rel_velocity)

                # Create the moon
                body = Body(name, mass, radius, position, velocity, color, texture)

                # Set the orbital period (around parent)
                body.orbital_period = 2 * math.pi * math.sqrt(distance_m**3 / (G * parent.mass))

                # Add to the simulation
                self.bodies.append(body)

        # Set the initially selected body to Earth
        for body in self.bodies:
            if body.name == "Earth":
                self.selected_body = body
                break

        # If Earth wasn't found, use the first body
        if not self.selected_body and self.bodies:
            self.selected_body = self.bodies[0]

    def update(self):
        """Update the simulation by one time step"""
        # Update performance metrics (FPS calculation)
        self.frame_count += 1
        current_time = time.time()
        if current_time - self.last_update_time > 1.0:
            self.fps = self.frame_count / (current_time - self.last_update_time)
            self.frame_count = 0
            self.last_update_time = current_time

        # If paused, don't update physics
        if self.paused:
            return

        try:
            # Adaptive time stepping (simplified for now)
            dt = self.simulation_speed

            # Calculate all forces between bodies
            self.calculate_forces()

            # Update velocities and positions
            for body in self.bodies:
                body.update_velocity(dt)
                body.update_position(dt)

            # Update trails less frequently
            self.trail_update_counter += 1
            if self.trail_update_counter >= TRAIL_UPDATE_FREQUENCY:
                self.trail_update_counter = 0
                for body in self.bodies:
                    if self.show_trails:
                       body.update_trail()
                    else:  # Clear trails if not showing
                        body.trail = []

            # Check for collisions
            self.handle_collisions()

            # Update elapsed time
            self.time_elapsed += dt

            # Update selected body info if needed
            if self.selected_body:
                self.selected_info = self.selected_body.get_info_text() #This is correct
        except Exception as e:
            print(f"Error during simulation update: {e}")

    def calculate_forces(self):
        """Calculate gravitational forces between all bodies"""
        # Reset forces on all bodies
        for body in self.bodies:
            body.forces = np.zeros(3, dtype=np.float64)

        # Calculate forces between all pairs of bodies
        for i, body1 in enumerate(self.bodies):
            for j, body2 in enumerate(self.bodies):
                if i != j:  # Skip self-interactions
                    try:
                        force = self.calculate_gravity(body1, body2)
                        body1.add_force(force)
                    except Exception as e:
                        print(f"Error calculating force between {body1.name} and {body2.name}: {e}")

    def calculate_gravity(self, body1, body2):
        """Calculate the gravitational force between two bodies"""
        # Vector from body1 to body2
        r_vec = body2.position - body1.position

        # Distance between bodies
        r = np.linalg.norm(r_vec)

        # Avoid division by zero or very small values
        if r < 1e-10:
            return np.zeros(3, dtype=np.float64)

        # Gravitational force magnitude: F = G * (m1 * m2) / r^2
        f_mag = G * body1.mass * body2.mass / (r * r)

        # Direction of force (unit vector)
        r_hat = r_vec / r

        # Force vector
        force = f_mag * r_hat

        return force

    def handle_collisions(self):
        """Detect and handle collisions between bodies"""
        # Create a list of collision pairs to avoid modifying the list while iterating
        bodies_to_remove = []
        bodies_to_add = []

        for i in range(len(self.bodies)):
            for j in range(i + 1, len(self.bodies)):
                body1 = self.bodies[i]
                body2 = self.bodies[j]

                # Check if both bodies exist:
                if body1 not in self.bodies or body2 not in self.bodies:
                    continue

                # Distance between bodies
                r_vec = body2.position - body1.position
                r = np.linalg.norm(r_vec)

                # Check for collision
                if r < (body1.radius + body2.radius) / SCALE_FACTOR:  # Check against actual radii
                   # print(f"Collision detected between {body1.name} and {body2.name}!")

                    # Merge bodies
                    try:
                        new_body = self.merge_bodies(body1, body2)
                        bodies_to_add.append(new_body)
                        bodies_to_remove.extend([body1, body2])
                    except Exception as e:
                        print(f"Error merging {body1.name} and {body2.name}: {e}")

        # Remove and add bodies after the loop
        for body in bodies_to_remove:
            if body in self.bodies:  # Double-check to avoid errors
                self.bodies.remove(body)


        for body in bodies_to_add:
            self.bodies.append(body)
            # If the selected body was removed, select the new body
            if self.selected_body in bodies_to_remove:
                self.selected_body = body

    def merge_bodies(self, body1, body2):
        """Merge two bodies, conserving momentum."""

        # Conservation of momentum: m1*v1 + m2*v2 = (m1+m2)*v_new
        total_mass = body1.mass + body2.mass
        new_velocity = (body1.mass * body1.velocity + body2.mass * body2.velocity) / total_mass

        # Weighted average position (center of mass)
        new_position = (body1.mass * body1.position + body2.mass * body2.position) / total_mass

        # Combine visual radius,  volume scales with radius cubed
        new_radius = (body1.radius**3 + body2.radius**3)**(1/3)

        # Choose properties of the larger body
        if body1.mass > body2.mass:
            new_name = body1.name
            new_color = body1.color
            new_texture = body1.texture_name
            new_has_rings = body1.has_rings
        else:
            new_name = body2.name
            new_color = body2.color
            new_texture = body2.texture_name
            new_has_rings = body2.has_rings

        # Create the new merged body
        merged_body = Body(new_name, total_mass, new_radius / 1000, new_position, new_velocity,
                          new_color, new_texture, new_has_rings)  # radius in km for the constructor

        return merged_body

    def handle_key(self, key):
        """Handle keyboard input for the simulation"""
        if key == K_SPACE:
            self.paused = not self.paused
        elif key == K_EQUALS or key == K_PLUS:
            # Increase simulation speed
            self.simulation_speed = min(self.simulation_speed * 2, MAX_SIMULATION_SPEED)
        elif key == K_MINUS:
            # Decrease simulation speed
            self.simulation_speed = max(self.simulation_speed / 2, MIN_SIMULATION_SPEED)
        elif key == K_t:
            # Toggle trails
            self.show_trails = not self.show_trails  # Use the class attribute
        elif key == K_l:
            # Toggle labels
            self.show_labels = not self.show_labels # Use the class attribute
        elif key == K_r:
            # Reset simulation
            self.bodies.clear()
            try:
                self.create_solar_system()
            except Exception as e:
                print(f"Error resetting simulation: {e}")
                self.create_minimal_system()
        elif key == K_TAB:
            # Cycle selected body
            if self.bodies:
                try:
                    current_index = self.bodies.index(self.selected_body) if self.selected_body else -1
                    next_index = (current_index + 1) % len(self.bodies)
                    self.selected_body = self.bodies[next_index]
                except ValueError:
                    # If selected body not in list, reset to first body
                    if self.bodies:
                        self.selected_body = self.bodies[0]

    def get_info_text(self):
        """Get overall simulation information (FPS, time, etc.)"""
        days = self.time_elapsed / (24 * 3600)
        years = days / 365.25

        if years > 1:
            time_str = f"{years:.2f} years"
        else:
            time_str = f"{days:.2f} days"

        info = [
            f"Bodies: {len(self.bodies)}",
            f"FPS: {self.fps:.1f}",
            f"Simulation Speed: {self.simulation_speed/3600:.1f} hours/frame",
            f"Elapsed Time: {time_str}",
            f"Status: {'Paused' if self.paused else 'Running'}"
        ]

        return info
    