#!/usr/bin/env python3
# ui.py - User interface for the gravity simulation with improved reliability

import pygame
import os
import numpy as np
from constants import *
from OpenGL.GL import *
from OpenGL.GLU import *
import traceback

class UI:
    """Handles UI rendering and interaction"""
    
    def __init__(self, simulation):
        """Initialize the UI system"""
        self.simulation = simulation
        self.text_renderer = TextRenderer()
        
        try:
            # Load fonts
            self.init_fonts()
            
            # Create UI components
            self.info_panel = InfoPanel(simulation, self.text_renderer)
            self.planet_labels = PlanetLabels(simulation, self.text_renderer)
            
            self.initialization_successful = True
        except Exception as e:
            print(f"Error initializing UI: {e}")
            traceback.print_exc()
            self.initialization_successful = False
    
    def init_fonts(self):
        """Initialize fonts"""
        try:
            # Create font directory if it doesn't exist
            if not os.path.exists(FONT_DIR):
                os.makedirs(FONT_DIR)
                
            # Initialize pygame font
            if not pygame.font.get_init():
                pygame.font.init()
                
            # Check if initialization was successful
            if not pygame.font.get_init():
                raise RuntimeError("Failed to initialize pygame font module")
            
            # Standard font
            self.text_renderer.add_font("standard", UI_FONT, UI_FONT_SIZE)
            
            # Header font
            self.text_renderer.add_font("title", UI_FONT, UI_TITLE_FONT_SIZE)
            
            # Label font
            self.text_renderer.add_font("label", UI_FONT, UI_LABEL_FONT_SIZE)
            
            # Monospace font for data
            self.text_renderer.add_font("mono", "Courier New", UI_FONT_SIZE)
            
            # Check if we have at least one font loaded
            if not self.text_renderer.fonts:
                raise RuntimeError("Failed to load any fonts")
                
        except Exception as e:
            print(f"Error initializing fonts: {e}")
            # Create a fallback font
            try:
                default_font = pygame.font.SysFont(None, UI_FONT_SIZE)  # Default system font
                self.text_renderer.fonts["standard"] = default_font
                self.text_renderer.fonts["title"] = default_font
                self.text_renderer.fonts["label"] = default_font
                self.text_renderer.fonts["mono"] = default_font
                print("Using fallback system font")
            except:
                print("Critical error: Could not create any fonts")
    
    def render(self, screen):
        """Render the UI elements"""
        if not self.initialization_successful:
            self.render_error_message(screen)
            return
            
        try:
            # Clear screen for UI rendering
            screen_overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
            
            # Render info panel
            self.info_panel.render(screen_overlay)
            
            # Render planet labels
            self.planet_labels.render(screen_overlay)
            
            # Render help text
            self.render_help_text(screen_overlay)
            
            # Render selected body info
            self.render_selected_info(screen_overlay)
            
            # Blit the overlay onto the screen
            screen.blit(screen_overlay, (0, 0))
        except Exception as e:
            print(f"Error rendering UI: {e}")
            self.render_error_message(screen)
    
    def render_error_message(self, screen):
        """Render a simple error message when UI rendering fails"""
        try:
            # Create a basic surface
            error_surface = pygame.Surface((400, 100), pygame.SRCALPHA)
            error_surface.fill((50, 0, 0, 200))
            
            # Try to render text if font system is working
            if hasattr(self, 'text_renderer') and self.text_renderer.fonts:
                font = next(iter(self.text_renderer.fonts.values()))
                text = font.render("UI Rendering Error - See Console", True, (255, 255, 255))
                error_surface.blit(text, (20, 40))
            
            # Position in center of screen
            screen.blit(error_surface, (WINDOW_WIDTH//2 - 200, WINDOW_HEIGHT//2 - 50))
        except:
            # If everything fails, we can't render anything
            pass
    
    def render_help_text(self, surface):
        """Render help text"""
        try:
            help_text = [
                "Controls:",
                "Space: Pause/Resume",
                "Tab: Cycle selected body",
                "+/-: Change simulation speed",
                "T: Toggle trails",
                "L: Toggle labels",
                "R: Reset simulation",
                "Mouse drag: Rotate view",
                "Mouse wheel: Zoom",
                "Middle mouse: Pan",
                "F: Toggle free camera mode",
                "WASD/QE: Free camera movement",
                "C: Reset camera",
                "Esc: Exit"
            ]
            
            x = WINDOW_WIDTH - 250
            y = WINDOW_HEIGHT - 20 * len(help_text) - UI_PADDING
            
            # Ensure text is on screen
            if y < 0:
                y = 0
                
            # Draw background panel
            pygame.draw.rect(surface, UI_BACKGROUND, 
                            (x - UI_PADDING, y - UI_PADDING, 
                             230 + UI_PADDING * 2, len(help_text) * 20 + UI_PADDING * 2),
                            0, 10)
            
            # Draw help text
            for line in help_text:
                text_surface = self.text_renderer.render_text(line, "standard", UI_COLOR)
                if text_surface:
                    surface.blit(text_surface, (x, y))
                y += 20
        except Exception as e:
            print(f"Error rendering help text: {e}")
    
    def render_selected_info(self, surface):
        """Render information about the selected body"""
        try:
            if not self.simulation.selected_body:
                return
                
            body = self.simulation.selected_body
            info = body.get_info_text()
            
            # Validate info list
            if not info or not isinstance(info, list):
                info = [f"Name: {body.name}"]
            
            # Render in the top right corner
            x = WINDOW_WIDTH - 300
            y = UI_PADDING
            
            # Draw background panel
            pygame.draw.rect(surface, UI_BACKGROUND, 
                            (x - UI_PADDING, y - UI_PADDING, 
                             290 + UI_PADDING * 2, (len(info) + 1) * 25 + UI_PADDING * 2),
                            0, 10)
            
            # Draw title
            title_surface = self.text_renderer.render_text(body.name, "title", UI_COLOR)
            if title_surface:
                surface.blit(title_surface, (x, y))
            y += 30
            
            # Draw info lines
            for line in info:
                text_surface = self.text_renderer.render_text(line, "standard", UI_COLOR)
                if text_surface:
                    surface.blit(text_surface, (x, y))
                y += 25
        except Exception as e:
            print(f"Error rendering selected info: {e}")


class TextRenderer:
    """Handles text rendering and caching"""
    
    def __init__(self):
        """Initialize the text renderer"""
        self.fonts = {}
        self.cache = {}
        self.max_cache_size = 1000  # Limit cache size
        self.fallback_font = None  # Emergency fallback font
        
        # Create an emergency fallback font that should work on any system
        try:
            self.fallback_font = pygame.font.SysFont(None, 16)  # Default system font
        except:
            print("Critical error: Could not create fallback font")
    
    def add_font(self, name, font_name, size):
        """Add a font to the renderer"""
        try:
            # Try to load the specified font
            try:
                # Try as a system font first
                font = pygame.font.SysFont(font_name, size)
                if font is None:
                    raise ValueError("SysFont returned None")
            except:
                try:
                    # Try as a file font
                    font_path = os.path.join(FONT_DIR, f"{font_name}.ttf")
                    if os.path.exists(font_path):
                        font = pygame.font.Font(font_path, size)
                    else:
                        raise FileNotFoundError(f"Font file not found: {font_path}")
                except:
                    # Fall back to default font
                    if self.fallback_font:
                        font = self.fallback_font
                    else:
                        font = pygame.font.SysFont(None, size)
            
            # Double-check that we have a valid font
            if font is None:
                raise RuntimeError("Failed to create a valid font object")
                
            self.fonts[name] = font
        except Exception as e:
            print(f"Error loading font {font_name}: {e}")
            # Use fallback font
            if self.fallback_font:
                self.fonts[name] = self.fallback_font
                print(f"Using fallback font for {name}")
    
    def render_text(self, text, font_name, color):
        """Render text to a surface, with caching"""
        try:
            # Cache key
            key = (text, font_name, color)
            
            # Check if already in cache
            if key in self.cache:
                return self.cache[key]
            
            # Ensure text is a string
            if not isinstance(text, str):
                text = str(text)
            
            # Get font
            font = self.fonts.get(font_name)
            
            # Fallback to any available font
            if font is None:
                if self.fonts:
                    # Use first available font
                    font = next(iter(self.fonts.values()))
                elif self.fallback_font:
                    # Use emergency fallback
                    font = self.fallback_font
                else:
                    # Cannot render text
                    print(f"No fonts available to render text: {text}")
                    return None
            
            # Render text
            try:
                text_surface = font.render(text, True, color)
            except Exception as e:
                # Try to render a simpler version
                try:
                    text_surface = font.render("Error rendering text", True, (255, 0, 0))
                except:
                    # Truly cannot render
                    return None
            
            # Cache the result
            if len(self.cache) >= self.max_cache_size:
                # Remove a random item if cache is full
                self.cache.pop(next(iter(self.cache.keys())))
                
            self.cache[key] = text_surface
            
            return text_surface
        except Exception as e:
            print(f"Error rendering text '{text}': {e}")
            return None


class InfoPanel:
    """Displays simulation information"""
    
    def __init__(self, simulation, text_renderer):
        """Initialize the info panel"""
        self.simulation = simulation
        self.text_renderer = text_renderer
    
    def render(self, surface):
        """Render the info panel"""
        try:
            info = self.simulation.get_info_text()
            
            # Validate info
            if not info or not isinstance(info, list):
                info = ["Simulation Info"]
            
            x = UI_PADDING
            y = UI_PADDING
            
            # Draw background panel
            pygame.draw.rect(surface, UI_BACKGROUND, 
                            (x - UI_PADDING, y - UI_PADDING, 
                             250 + UI_PADDING * 2, len(info) * 25 + UI_PADDING * 2),
                            0, 10)
            
            # Draw info lines
            for line in info:
                text_surface = self.text_renderer.render_text(line, "standard", UI_COLOR)
                if text_surface:
                    surface.blit(text_surface, (x, y))
                y += 25
        except Exception as e:
            print(f"Error rendering info panel: {e}")


class PlanetLabels:
    """Handles rendering of planet labels in 3D space"""
    
    def __init__(self, simulation, text_renderer):
        """Initialize the planet label system"""
        self.simulation = simulation
        self.text_renderer = text_renderer
    
    def render(self, surface):
        """Render planet labels"""
        if not SHOW_LABELS:
            return
            
        try:
            for body in self.simulation.bodies:
                # Skip if position is invalid
                if not np.all(np.isfinite(body.position)):
                    continue
                    
                # Project 3D position to 2D screen coordinates
                screen_pos = self.project_to_screen(body.position / SCALE_FACTOR)
                
                if screen_pos:
                    x, y = screen_pos
                    
                    # Only render if on screen with padding
                    padding = 50
                    if (padding <= x < WINDOW_WIDTH - padding and 
                        padding <= y < WINDOW_HEIGHT - padding):
                        # Render body name
                        text_surface = self.text_renderer.render_text(body.name, "label", UI_COLOR)
                        if text_surface:
                            text_width, text_height = text_surface.get_size()
                            
                            # Center text on screen position
                            surface.blit(text_surface, (x - text_width // 2, y - text_height - 5))
        except Exception as e:
            print(f"Error rendering planet labels: {e}")
    
    def project_to_screen(self, position):
        """Project a 3D world position to 2D screen coordinates"""
        try:
            # Get current matrices
            model_view = glGetDoublev(GL_MODELVIEW_MATRIX)
            projection = glGetDoublev(GL_PROJECTION_MATRIX)
            viewport = glGetIntegerv(GL_VIEWPORT)
            
            # Validate inputs
            if not np.all(np.isfinite(position)):
                return None
                
            if not np.all(np.isfinite(model_view)) or not np.all(np.isfinite(projection)):
                return None
            
            # Project point
            screen_x, screen_y, screen_z = gluProject(
                position[0], position[1], position[2],
                model_view, projection, viewport
            )
            
            # Convert from OpenGL coordinates to Pygame coordinates
            screen_y = viewport[3] - screen_y
            
            # Return 2D position if point is in front of camera
            if 0.0 <= screen_z <= 1.0:
                return (int(screen_x), int(screen_y))
            else:
                return None
                
        except Exception as e:
            print(f"Error projecting point: {e}")
            return None
        