#!/usr/bin/env python3
# renderer.py - OpenGL rendering for gravity simulation

import numpy as np
import pygame
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GL import shaders
import os
import math
import traceback

# --- Constants (Replace with your actual constants) ---
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
SCALE_FACTOR = 1  # Adjust as needed
SPHERE_DETAIL = 30
SHOW_TRAILS = True
SHOW_AXES = True
SHOW_LABELS = False  # Placeholder for label rendering
TEXTURE_DIR = "textures"  # Directory for textures
SHADER_DIR = "shaders"  # Directory for shaders

# Placeholder logger (replace with a real logging system)
class Logger:
    def error(self, message):
        print(f"ERROR: {message}")
    def warning(self, message):
        print(f"WARNING: {message}")

logger = Logger()

class TextureManager:
    """Manages loading and binding of textures"""

    def __init__(self):
        """Initialize the texture manager"""
        self.textures = {}

        # Create texture directory if it doesn't exist
        if not os.path.exists(TEXTURE_DIR):
            os.makedirs(TEXTURE_DIR)

        # Default texture (used if requested texture not found)
        self.default_texture = self.create_default_texture()

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
            # Create a fallback texture ID that's just blank
            try:
                texture_id = glGenTextures(1)
                glBindTexture(GL_TEXTURE_2D, texture_id)
                data = np.zeros((4, 4, 4), dtype=np.uint8)
                data[:,:] = [255, 255, 255, 255]
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 4, 4, 0, GL_RGBA, GL_UNSIGNED_BYTE, data)
                return texture_id
            except:
                print("Critical error creating fallback texture")
                return 0

    def load_texture(self, filename):
        """Load a texture from file"""
        if filename in self.textures:
            return self.textures[filename]

        try:
            # Full path to texture file
            filepath = os.path.join(TEXTURE_DIR, filename)

            # Check if file exists
            if not os.path.exists(filepath):
                print(f"Texture file not found: {filepath}")
                return self.default_texture

            # Load image using pygame
            image = pygame.image.load(filepath)

            # Convert to RGBA if not already
            if image.get_bytesize() == 3:  # RGB
                image = image.convert_alpha()

            image_data = pygame.image.tostring(image, "RGBA", 1)
            width, height = image.get_size()

            # Generate OpenGL texture
            texture_id = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, texture_id)

            # Set texture parameters
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)

            # Upload texture data
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, image_data)

            # Generate mipmaps
            glGenerateMipmap(GL_TEXTURE_2D)

            # Store and return texture ID
            self.textures[filename] = texture_id
            return texture_id

        except Exception as e:
            print(f"Error loading texture {filename}: {e}")
            traceback.print_exc()
            return self.default_texture

    def cleanup(self):
        """Delete all textures"""
        try:
            for texture_id in self.textures.values():
                if texture_id:
                    glDeleteTextures(1, [texture_id])

            if self.default_texture:
                glDeleteTextures(1, [self.default_texture])

            self.textures = {}
        except Exception as e:
            print(f"Error cleaning up textures: {e}")


class ShaderManager:
    """Manages loading and compiling of GLSL shaders"""

    def __init__(self):
        """Initialize the shader manager"""
        self.shaders = {}

        # Create shader directory if it doesn't exist
        if not os.path.exists(SHADER_DIR):
            os.makedirs(SHADER_DIR)

        # Create default shaders
        self.create_default_shaders()

    def create_default_shaders(self):
        """Create default shaders if not found on disk"""
        try:
            # Planet shader
            if not self.load_shader_from_files("planet", "planet_vertex.glsl", "planet_fragment.glsl"):
                self.create_default_planet_shader()

            # Trail shader
            if not self.load_shader_from_files("trail", "trail_vertex.glsl", "trail_fragment.glsl"):
                self.create_default_trail_shader()

            # Ring shader
            if not self.load_shader_from_files("ring", "ring_vertex.glsl", "ring_fragment.glsl"):
                self.create_default_ring_shader()
        except Exception as e:
            print(f"Error creating default shaders: {e}")
            traceback.print_exc()

    def load_shader_from_files(self, name, vertex_file, fragment_file):
        """Try to load shader from files, return True if successful"""
        try:
            # Construct full paths
            vertex_path = os.path.join(SHADER_DIR, vertex_file)
            fragment_path = os.path.join(SHADER_DIR, fragment_file)

            # Check if files exist
            if not os.path.exists(vertex_path) or not os.path.exists(fragment_path):
                return False

            # Read shader source
            with open(vertex_path, 'r') as f:
                vertex_src = f.read()

            with open(fragment_path, 'r') as f:
                fragment_src = f.read()

            # Compile shaders
            shader_program = self.compile_shader(vertex_src, fragment_src)
            if shader_program:
                self.shaders[name] = shader_program
                return True
            return False
        except Exception as e:
            print(f"Error loading shader files for {name}: {e}")
            return False

    def compile_shader(self, vertex_src, fragment_src):
        """Compile and link shader program"""
        try:
            vertex_shader = shaders.compileShader(vertex_src, GL_VERTEX_SHADER)
            fragment_shader = shaders.compileShader(fragment_src, GL_FRAGMENT_SHADER)
            shader_program = shaders.compileProgram(vertex_shader, fragment_shader)
            return shader_program
        except Exception as e:
            print(f"Error compiling shader: {e}")
            return 0

    def create_default_planet_shader(self):
        """Create the default planet shader"""
        # Vertex shader
        vertex_src = """
        #version 330 core

        layout(location = 0) in vec3 position;
        layout(location = 1) in vec3 normal;
        layout(location = 2) in vec2 texCoord;

        out vec3 fragNormal;
        out vec3 fragPosition;
        out vec2 fragTexCoord;

        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;

        void main() {
            fragPosition = vec3(model * vec4(position, 1.0));
            fragNormal = mat3(transpose(inverse(model))) * normal;
            fragTexCoord = texCoord;

            gl_Position = projection * view * model * vec4(position, 1.0);
        }
        """

        # Fragment shader
        fragment_src = """
        #version 330 core

        in vec3 fragNormal;
        in vec3 fragPosition;
        in vec2 fragTexCoord;

        out vec4 fragColor;

        uniform vec3 lightPos;
        uniform vec3 viewPos;
        uniform vec3 baseColor;
        uniform sampler2D texSampler;
        uniform bool useTexture;

        void main() {
            // Ambient
            float ambientStrength = 0.3;
            vec3 ambient = ambientStrength * vec3(1.0, 1.0, 1.0);

            // Diffuse
            vec3 norm = normalize(fragNormal);
            vec3 lightDir = normalize(lightPos - fragPosition);
            float diff = max(dot(norm, lightDir), 0.0);
            vec3 diffuse = diff * vec3(1.0, 1.0, 1.0);

            // Specular
            float specularStrength = 0.5;
            vec3 viewDir = normalize(viewPos - fragPosition);
            vec3 reflectDir = reflect(-lightDir, norm);
            float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
            vec3 specular = specularStrength * spec * vec3(1.0, 1.0, 1.0);

            // Get color
            vec3 color;
            if (useTexture) {
                color = texture(texSampler, fragTexCoord).rgb;
            } else {
                color = baseColor;
            }

            // Combine
            vec3 result = (ambient + diffuse + specular) * color;
            fragColor = vec4(result, 1.0);
        }
        """

        # Compile shaders
        try:
            shader_program = self.compile_shader(vertex_src, fragment_src)
            if shader_program:
                self.shaders["planet"] = shader_program
        except Exception as e:
            print(f"Error creating planet shader: {e}")

    def create_default_trail_shader(self):
        """Create the default trail shader"""
        # Vertex shader
        vertex_src = """
        #version 330 core

        layout(location = 0) in vec3 position;

        uniform mat4 view;
        uniform mat4 projection;

        void main() {
            gl_Position = projection * view * vec4(position, 1.0);
        }
        """

        # Fragment shader
        fragment_src = """
        #version 330 core

        out vec4 fragColor;

        uniform vec3 trailColor;

        void main() {
            fragColor = vec4(trailColor, 0.5);
        }
        """

        # Compile shaders
        try:
            shader_program = self.compile_shader(vertex_src, fragment_src)
            if shader_program:
                self.shaders["trail"] = shader_program
        except Exception as e:
            print(f"Error creating trail shader: {e}")

    def create_default_ring_shader(self):
        """Create the default ring shader"""
        # Vertex shader
        vertex_src = """
        #version 330 core

        layout(location = 0) in vec3 position;
        layout(location = 1) in vec2 texCoord;

        out vec2 fragTexCoord;

        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;

        void main() {
            fragTexCoord = texCoord;
            gl_Position = projection * view * model * vec4(position, 1.0);
        }
        """

        # Fragment shader
        fragment_src = """
        #version 330 core

        in vec2 fragTexCoord;

        out vec4 fragColor;

        uniform sampler2D texSampler;
        uniform vec3 ringColor;

        void main() {
            // Calculate distance from center (0.5, 0.5)
            vec2 center = vec2(0.5, 0.5);
            float dist = distance(fragTexCoord, center) * 2.0;

            // Ring pattern
            float ring = smoothstep(0.8, 0.85, dist) * smoothstep(1.0, 0.95, dist);

            // Use texture if available, otherwise procedural ring
            vec4 texColor = texture(texSampler, fragTexCoord);
            float alpha = texColor.a > 0.0 ? texColor.a : ring * 0.7;

            // Final color
            vec3 color = texColor.rgb;
            if (texColor.a == 0.0) {
                color = ringColor;
            }

            fragColor = vec4(color, alpha);
        }
        """

        # Compile shaders
        try:
            shader_program = self.compile_shader(vertex_src, fragment_src)
            if shader_program:
                self.shaders["ring"] = shader_program
        except Exception as e:
            print(f"Error creating ring shader: {e}")

    def get_shader(self, name):
        """Get a shader by name"""
        if name in self.shaders:
            return self.shaders[name]
        else:
            print(f"Shader {name} not found")
            # Return a valid shader program if possible
            if self.shaders:
                return next(iter(self.shaders.values()))
            return 0

    def cleanup(self):
        """Delete all shaders"""
        try:
            for shader in self.shaders.values():
                if shader:
                    glDeleteProgram(shader)
            self.shaders = {}
        except Exception as e:
            print(f"Error cleaning up shaders: {e}")


class Renderer:
    """Handles OpenGL rendering of the simulation"""

    def __init__(self, simulation, camera):
        """Initialize the renderer"""
        self.simulation = simulation
        self.camera = camera
        self.vaos = {}  # Track VAOs for cleanup
        self.vbos = {}  # Track VBOs for cleanup

        # Initialize OpenGL
        try:
            self.setup_opengl()

            # Create texture and shader managers
            self.texture_manager = TextureManager()
            self.shader_manager = ShaderManager()

            # Create mesh data
            self.sphere_vao, self.sphere_vertex_count = self.create_sphere(SPHERE_DETAIL, SPHERE_DETAIL)
            self.ring_vao, self.ring_vertex_count = self.create_ring(SPHERE_DETAIL)

            # Assign textures to bodies
            self.assign_textures()

            self.initialization_successful = True
        except Exception as e:
            print(f"Error initializing renderer: {e}")
            traceback.print_exc()
            self.initialization_successful = False

    def setup_opengl(self):
        """Set up OpenGL state"""
        # Enable depth testing
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)

        # Enable blending for transparency
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # Enable face culling
        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)

        # Set clear color (deep space black with a hint of blue)
        glClearColor(0.0, 0.0, 0.05, 1.0)

        # Set up projection matrix (perspective)
        # Note: We set up the projection matrix here, but we also *get*
        # the current projection matrix during rendering to handle any
        # changes (e.g., window resizing).  This combines setup with
        # flexibility.
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()  # Reset the projection matrix
        gluPerspective(45, WINDOW_WIDTH / WINDOW_HEIGHT, 0.1, 10000.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()


    def create_sphere(self, stacks, slices):
        """Create a sphere mesh"""
        try:
            vertices = []
            normals = []
            tex_coords = []

            # Generate vertices, normals, and texture coordinates
            for stack in range(stacks + 1):
                phi = stack * math.pi / stacks
                for slice in range(slices + 1):
                    theta = slice * 2 * math.pi / slices

                    # Vertex position
                    x = math.sin(phi) * math.cos(theta)
                    y = math.cos(phi)
                    z = math.sin(phi) * math.sin(theta)
                    vertices.extend([x, y, z])

                    # Normal (same as position for a unit sphere)
                    normals.extend([x, y, z])

                    # Texture coordinates
                    s = 1.0 - slice / slices
                    t = stack / stacks
                    tex_coords.extend([s, t])

            # Generate indices
            indices = []
            for stack in range(stacks):
                for slice in range(slices):
                    # Calculate indices for two triangles per quad
                    p1 = stack * (slices + 1) + slice
                    p2 = p1 + 1
                    p3 = p1 + (slices + 1)
                    p4 = p3 + 1

                    # First triangle
                    indices.extend([p1, p3, p2])
                    # Second triangle
                    indices.extend([p2, p3, p4])

            # Convert to numpy arrays
            vertices = np.array(vertices, dtype=np.float32)
            normals = np.array(normals, dtype=np.float32)
            tex_coords = np.array(tex_coords, dtype=np.float32)
            indices = np.array(indices, dtype=np.uint32)

            # Create buffers
            vao = glGenVertexArrays(1)
            self.vaos['sphere'] = vao
            glBindVertexArray(vao)

            # Vertex positions
            vbo = glGenBuffers(1)
            self.vbos['sphere_pos'] = vbo
            glBindBuffer(GL_ARRAY_BUFFER, vbo)
            glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
            glEnableVertexAttribArray(0)

            # Normals
            nbo = glGenBuffers(1)
            self.vbos['sphere_norm'] = nbo
            glBindBuffer(GL_ARRAY_BUFFER, nbo)
            glBufferData(GL_ARRAY_BUFFER, normals.nbytes, normals, GL_STATIC_DRAW)
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, None)
            glEnableVertexAttribArray(1)

            # Texture coordinates
            tbo = glGenBuffers(1)
            self.vbos['sphere_tex'] = tbo
            glBindBuffer(GL_ARRAY_BUFFER, tbo)
            glBufferData(GL_ARRAY_BUFFER, tex_coords.nbytes, tex_coords, GL_STATIC_DRAW)
            glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 0, None)
            glEnableVertexAttribArray(2)

            # Element buffer
            ebo = glGenBuffers(1)
            self.vbos['sphere_elem'] = ebo
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

            # Unbind VAO
            glBindVertexArray(0)

            return vao, len(indices)
        except Exception as e:
            print(f"Error creating sphere mesh: {e}")
            traceback.print_exc()
            return 0, 0

    def create_ring(self, segments):
        """Create a ring mesh"""
        try:
            vertices = []
            tex_coords = []

            inner_radius = 0.6
            outer_radius = 1.0

            # Generate vertices and texture coordinates
            for i in range(segments + 1):
                theta = i * 2 * math.pi / segments

                # Inner vertex
                x_in = inner_radius * math.cos(theta)
                z_in = inner_radius * math.sin(theta)
                vertices.extend([x_in, 0, z_in])

                # Texture coordinate (inner)
                tx_in = 0.5 + 0.5 * inner_radius * math.cos(theta)
                ty_in = 0.5 + 0.5 * inner_radius * math.sin(theta)
                tex_coords.extend([tx_in, ty_in])

                # Outer vertex
                x_out = outer_radius * math.cos(theta)
                z_out = outer_radius * math.sin(theta)
                vertices.extend([x_out, 0, z_out])

                # Texture coordinate (outer)
                tx_out = 0.5 + 0.5 * outer_radius * math.cos(theta)
                ty_out = 0.5 + 0.5 * outer_radius * math.sin(theta)
                tex_coords.extend([tx_out, ty_out])

            # Convert to numpy arrays
            vertices = np.array(vertices, dtype=np.float32)
            tex_coords = np.array(tex_coords, dtype=np.float32)

            # Create buffers
            vao = glGenVertexArrays(1)
            self.vaos['ring'] = vao
            glBindVertexArray(vao)

            # Vertex positions
            vbo = glGenBuffers(1)
            self.vbos['ring_pos'] = vbo
            glBindBuffer(GL_ARRAY_BUFFER, vbo)
            glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
            glEnableVertexAttribArray(0)

            # Texture coordinates
            tbo = glGenBuffers(1)
            self.vbos['ring_tex'] = tbo
            glBindBuffer(GL_ARRAY_BUFFER, tbo)
            glBufferData(GL_ARRAY_BUFFER, tex_coords.nbytes, tex_coords, GL_STATIC_DRAW)
            glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, None)
            glEnableVertexAttribArray(1)

            # Unbind VAO
            glBindVertexArray(0)

            return vao, len(vertices) // 3  # Number of vertices, not indices
        except Exception as e:
            print(f"Error creating ring mesh: {e}")
            traceback.print_exc()
            return 0, 0

    def assign_textures(self):
        """Assign textures to bodies in the simulation"""
        for body in self.simulation.bodies:
            body.texture_id = self.texture_manager.load_texture(body.texture_name)

    def render(self):
        """Render the scene"""
        if not self.initialization_successful:
            return

        try:
            # Clear buffers
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            # Set up view matrix
            glLoadIdentity()
            self.camera.setup_view()

            # Get view and projection matrices
            view_matrix = glGetFloatv(GL_MODELVIEW_MATRIX)
            glMatrixMode(GL_PROJECTION)
            projection_matrix = glGetFloatv(GL_PROJECTION_MATRIX)
            glMatrixMode(GL_MODELVIEW)

            # Draw celestial bodies
            self.draw_bodies(view_matrix, projection_matrix)

            # Draw orbital trails
            if SHOW_TRAILS:
                self.draw_trails(view_matrix, projection_matrix)

            # Draw coordinate axes if enabled
            if SHOW_AXES:
                self.draw_axes()

        except Exception as e:
            print(f"Error in render method: {e}")
            traceback.print_exc()


    def _draw_body_standard(self, body, view_matrix, projection_matrix):
        """Standard rendering for a single celestial body (used as fallback)"""
        # Skip if body has invalid position
        if not np.all(np.isfinite(body.position)):
            return

        shader = self.shader_manager.get_shader("planet")
        if not shader:
            return
        try:
            glUseProgram(shader)  # Make sure to use the shader

            # Set model matrix
            model_matrix = np.identity(4, dtype=np.float32)

            # Apply translation
            model_matrix[0, 3] = body.position[0] / SCALE_FACTOR
            model_matrix[1, 3] = body.position[1] / SCALE_FACTOR
            model_matrix[2, 3] = body.position[2] / SCALE_FACTOR

            # Apply scaling
            model_matrix[0, 0] = body.visual_radius
            model_matrix[1, 1] = body.visual_radius
            model_matrix[2, 2] = body.visual_radius

            # Set uniforms
            model_loc = glGetUniformLocation(shader, "model")
            base_color_loc = glGetUniformLocation(shader, "baseColor")
            use_texture_loc = glGetUniformLocation(shader, "useTexture")

            glUniformMatrix4fv(model_loc, 1, GL_FALSE, model_matrix)
            glUniform3f(base_color_loc, body.color[0], body.color[1], body.color[2])

            # Bind texture if available
            if body.texture_id:
                glActiveTexture(GL_TEXTURE0)
                glBindTexture(GL_TEXTURE_2D, body.texture_id)
                glUniform1i(glGetUniformLocation(shader, "texSampler"), 0)
                glUniform1i(use_texture_loc, 1)
            else:
                glUniform1i(use_texture_loc, 0)

            # Draw sphere
            glDrawElements(GL_TRIANGLES, self.sphere_vertex_count, GL_UNSIGNED_INT, None)

            # Draw rings if body has them
            if body.has_rings:
                self.draw_rings(body, model_matrix, view_matrix, projection_matrix)
        except Exception as e:
            logger.error(f"Error in _draw_body_standard for {body.name}: {e}")
            traceback.print_exc()
        finally:
            glUseProgram(0)  # Always unbind the shader

    def draw_bodies(self, view_matrix, projection_matrix):
        """Draw all celestial bodies, with fallback to standard rendering."""

        # Skip if not initialized
        if not hasattr(self, 'initialization_successful') or not self.initialization_successful:
            return

        # --- Advanced Rendering Attempt (Optional) ---
        try:
            import renderer_high_res  # Try importing high-res renderer

            successful_renders = 0
            for body in self.simulation.bodies:
                if not np.all(np.isfinite(body.position)):
                    continue

                try:
                    success = renderer_high_res.draw_body_with_advanced_shader(
                        self, body, view_matrix, projection_matrix
                    )
                    if success:
                        successful_renders += 1
                    else:
                        self._draw_body_standard(body, view_matrix, projection_matrix)
                except Exception as e:
                    logger.error(f"Error rendering {body.name} with advanced shader: {e}")
                    traceback.print_exc()
                    self._draw_body_standard(body, view_matrix, projection_matrix)
        except ImportError as e:
            logger.warning("High-res renderer not available, using standard rendering.")
            for body in self.simulation.bodies:
                if not np.all(np.isfinite(body.position)):
                    continue
                self._draw_body_standard(body, view_matrix, projection_matrix)
        except Exception as e:
            logger.error(f"Unexpected error in draw_bodies: {e}")
            traceback.print_exc()
            for body in self.simulation.bodies:
                if not np.all(np.isfinite(body.position)):
                    continue
                self._draw_body_standard(body, view_matrix, projection_matrix)