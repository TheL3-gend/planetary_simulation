#!/usr/bin/env python3
# renderer_high_res.py - High-resolution texture support for the renderer

import os
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from constants import *
from texture_loader import TextureLoader
from texture_mapping import get_texture_filename, get_normal_map, get_specular_map
import high_res_texture_integration as hires

def initialize_renderer(renderer):
    """
    Initialize the renderer with high-resolution texture support
    
    Args:
        renderer: The renderer object to initialize
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Check for advanced shaders
        has_advanced_shaders = hires.ensure_advanced_shaders()
        
        # Replace texture manager with enhanced version
        if hasattr(renderer, 'texture_manager'):
            # Clean up old texture manager if it exists
            try:
                renderer.texture_manager.cleanup()
            except:
                pass
        
        # Create enhanced texture loader
        renderer.texture_loader = TextureLoader(TEXTURE_DIR)
        
        # Check for high-resolution textures
        has_textures, missing, has_high_res = hires.check_textures()
        
        if not has_textures:
            print("Warning: Some essential textures are missing: ", ", ".join(missing))
            hires.print_texture_instructions()
        
        if has_high_res:
            print("Using high-resolution planet textures")
        
        # Try to automatically rename textures if needed
        hires.rename_textures_if_needed()
        
        # If advanced shaders are available, use them
        if has_advanced_shaders:
            renderer.use_advanced_shaders = True
            load_advanced_shaders(renderer)
        else:
            renderer.use_advanced_shaders = False
        
        # Assign high-resolution textures to bodies
        assign_high_res_textures(renderer)
        
        return True
    except Exception as e:
        print(f"Error initializing renderer with high-resolution textures: {e}")
        return False

def load_advanced_shaders(renderer):
    """
    Load advanced shaders for high-resolution textures
    
    Args:
        renderer: The renderer object
    """
    if not hasattr(renderer, 'shader_manager'):
        return
    
    try:
        # Load advanced planet shader
        vertex_path = os.path.join(SHADER_DIR, "advanced_planet_vertex.glsl")
        fragment_path = os.path.join(SHADER_DIR, "advanced_planet_fragment.glsl")
        
        if os.path.exists(vertex_path) and os.path.exists(fragment_path):
            # Load vertex shader
            with open(vertex_path, 'r') as f:
                vertex_src = f.read()
                
            # Load fragment shader
            with open(fragment_path, 'r') as f:
                fragment_src = f.read()
                
            # Compile shaders
            shader_program = renderer.shader_manager.compile_shader(vertex_src, fragment_src)
            if shader_program:
                renderer.shader_manager.shaders["advanced_planet"] = shader_program
                print("Advanced planet shader loaded successfully")
            else:
                print("Failed to compile advanced planet shader")
        else:
            print("Advanced planet shader files not found")
    except Exception as e:
        print(f"Error loading advanced shaders: {e}")

def assign_high_res_textures(renderer):
    """
    Assign high-resolution textures to bodies in the simulation
    
    Args:
        renderer: The renderer object
    """
    if not hasattr(renderer, 'simulation') or not hasattr(renderer, 'texture_loader'):
        return
    
    try:
        # Update texture names in celestial bodies to use high-res versions if available
        hires.setup_high_res_textures(renderer.simulation)
        
        # Load textures for each body
        for body in renderer.simulation.bodies:
            # Load main texture
            body.texture_id = renderer.texture_loader.load_texture(body.texture_name)
            
            # Load normal map if available
            if hasattr(body, 'normal_map_name') and body.normal_map_name:
                body.normal_map_id = renderer.texture_loader.load_texture(body.normal_map_name)
            else:
                body.normal_map_id = None
                
            # Load specular map if available
            if hasattr(body, 'specular_map_name') and body.specular_map_name:
                body.specular_map_id = renderer.texture_loader.load_texture(body.specular_map_name)
            else:
                body.specular_map_id = None
                
            # Load cloud texture if available
            if hasattr(body, 'clouds_texture_name') and body.clouds_texture_name:
                body.clouds_texture_id = renderer.texture_loader.load_texture(body.clouds_texture_name)
            else:
                body.clouds_texture_id = None
                
            # Load night texture if available
            if hasattr(body, 'night_texture_name') and body.night_texture_name:
                body.night_texture_id = renderer.texture_loader.load_texture(body.night_texture_name)
            else:
                body.night_texture_id = None
                
        print("All textures assigned successfully")
    except Exception as e:
        print(f"Error assigning high-resolution textures: {e}")

def create_enhanced_sphere(renderer, stacks, slices):
    """
    Create an enhanced sphere mesh with tangent and bitangent data for normal mapping
    
    Args:
        renderer: The renderer object
        stacks: Number of horizontal divisions
        slices: Number of vertical divisions
        
    Returns:
        Tuple of (vao, vertex_count)
    """
    try:
        vertices = []
        normals = []
        tex_coords = []
        
        # Generate vertices, normals, and texture coordinates
        for stack in range(stacks + 1):
            phi = stack * np.pi / stacks
            for slice in range(slices + 1):
                theta = slice * 2 * np.pi / slices
                
                # Vertex position
                x = np.sin(phi) * np.cos(theta)
                y = np.cos(phi)
                z = np.sin(phi) * np.sin(theta)
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
        
        # Generate tangents and bitangents for normal mapping
        tangents, bitangents = hires.generate_tangents_bitangents(
            vertices, tex_coords, normals, indices)
        
        # Convert to numpy arrays
        vertices = np.array(vertices, dtype=np.float32)
        normals = np.array(normals, dtype=np.float32)
        tex_coords = np.array(tex_coords, dtype=np.float32)
        tangents = np.array(tangents, dtype=np.float32)
        bitangents = np.array(bitangents, dtype=np.float32)
        indices = np.array(indices, dtype=np.uint32)
        
        # Create buffers
        vao = glGenVertexArrays(1)
        renderer.vaos['enhanced_sphere'] = vao
        glBindVertexArray(vao)
        
        # Vertex positions
        vbo = glGenBuffers(1)
        renderer.vbos['enhanced_sphere_pos'] = vbo
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(0)
        
        # Normals
        nbo = glGenBuffers(1)
        renderer.vbos['enhanced_sphere_norm'] = nbo
        glBindBuffer(GL_ARRAY_BUFFER, nbo)
        glBufferData(GL_ARRAY_BUFFER, normals.nbytes, normals, GL_STATIC_DRAW)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(1)
        
        # Texture coordinates
        tbo = glGenBuffers(1)
        renderer.vbos['enhanced_sphere_tex'] = tbo
        glBindBuffer(GL_ARRAY_BUFFER, tbo)
        glBufferData(GL_ARRAY_BUFFER, tex_coords.nbytes, tex_coords, GL_STATIC_DRAW)
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(2)
        
        # Tangents
        tan_bo = glGenBuffers(1)
        renderer.vbos['enhanced_sphere_tan'] = tan_bo
        glBindBuffer(GL_ARRAY_BUFFER, tan_bo)
        glBufferData(GL_ARRAY_BUFFER, tangents.nbytes, tangents, GL_STATIC_DRAW)
        glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(3)
        
        # Bitangents
        bitan_bo = glGenBuffers(1)
        renderer.vbos['enhanced_sphere_bitan'] = bitan_bo
        glBindBuffer(GL_ARRAY_BUFFER, bitan_bo)
        glBufferData(GL_ARRAY_BUFFER, bitangents.nbytes, bitangents, GL_STATIC_DRAW)
        glVertexAttribPointer(4, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(4)
        
        # Element buffer
        ebo = glGenBuffers(1)
        renderer.vbos['enhanced_sphere_elem'] = ebo
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
        
        # Unbind VAO
        glBindVertexArray(0)
        
        return vao, len(indices)
    except Exception as e:
        print(f"Error creating enhanced sphere mesh: {e}")
        return 0, 0

def draw_body_with_advanced_shader(renderer, body, view_matrix, projection_matrix):
    """
    Draw a celestial body using the advanced shader with normal mapping and special effects
    
    Args:
        renderer: The renderer object
        body: The celestial body to draw
        view_matrix: The view matrix
        projection_matrix: The projection matrix
    """
    if not renderer.use_advanced_shaders:
        return False
        
    try:
        # Use advanced planet shader
        shader = renderer.shader_manager.get_shader("advanced_planet")
        if not shader:
            return False
            
        glUseProgram(shader)
        
        # Set uniform values
        light_pos_loc = glGetUniformLocation(shader, "lightPos")
        view_pos_loc = glGetUniformLocation(shader, "viewPos")
        
        # Find the sun's position
        sun_pos = np.array([0, 0, 0])  # Default if sun not found
        for b in renderer.simulation.bodies:
            if b.name == "Sun":
                sun_pos = b.position / SCALE_FACTOR
                break
        
        glUniform3f(light_pos_loc, sun_pos[0], sun_pos[1], sun_pos[2])
        
        # Calculate view position
        eye_pos = renderer.camera.target - np.array([
            renderer.camera.distance * np.sin(renderer.camera.x_angle) * np.cos(renderer.camera.y_angle),
            renderer.camera.distance * np.sin(renderer.camera.y_angle),
            renderer.camera.distance * np.cos(renderer.camera.x_angle) * np.cos(renderer.camera.y_angle)
        ])
        glUniform3f(view_pos_loc, eye_pos[0], eye_pos[1], eye_pos[2])
        
        # Set matrices
        proj_loc = glGetUniformLocation(shader, "projection")
        view_loc = glGetUniformLocation(shader, "view")
        model_loc = glGetUniformLocation(shader, "model")
        
        glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection_matrix)
        glUniformMatrix4fv(view_loc, 1, GL_FALSE, view_matrix)
        
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
        
        glUniformMatrix4fv(model_loc, 1, GL_FALSE, model_matrix)
        
        # Set material properties
        base_color_loc = glGetUniformLocation(shader, "baseColor")
        ambient_str_loc = glGetUniformLocation(shader, "ambientStrength")
        specular_str_loc = glGetUniformLocation(shader, "specularStrength")
        shininess_loc = glGetUniformLocation(shader, "shininess")
        
        glUniform3f(base_color_loc, body.color[0], body.color[1], body.color[2])
        glUniform1f(ambient_str_loc, 0.3)
        glUniform1f(specular_str_loc, 0.5)
        glUniform1f(shininess_loc, 32.0)
        
        # Set time for animations
        time_loc = glGetUniformLocation(shader, "time")
        glUniform1f(time_loc, float(pygame.time.get_ticks()) / 1000.0)
        
        # Set feature toggles
        use_texture_loc = glGetUniformLocation(shader, "useTexture")
        use_normal_map_loc = glGetUniformLocation(shader, "useNormalMap")
        use_specular_map_loc = glGetUniformLocation(shader, "useSpecularMap")
        use_night_map_loc = glGetUniformLocation(shader, "useNightMap")
        use_clouds_map_loc = glGetUniformLocation(shader, "useCloudsMap")
        is_atmospheric_loc = glGetUniformLocation(shader, "isAtmospheric")
        
        # Bind diffuse texture
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, body.texture_id)
        glUniform1i(glGetUniformLocation(shader, "diffuseMap"), 0)
        glUniform1i(use_texture_loc, 1)
        
        # Bind normal map if available
        if hasattr(body, 'normal_map_id') and body.normal_map_id:
            glActiveTexture(GL_TEXTURE1)
            glBindTexture(GL_TEXTURE_2D, body.normal_map_id)
            glUniform1i(glGetUniformLocation(shader, "normalMap"), 1)
            glUniform1i(use_normal_map_loc, 1)
        else:
            glUniform1i(use_normal_map_loc, 0)
            
        # Bind specular map if available
        if hasattr(body, 'specular_map_id') and body.specular_map_id:
            glActiveTexture(GL_TEXTURE2)
            glBindTexture(GL_TEXTURE_2D, body.specular_map_id)
            glUniform1i(glGetUniformLocation(shader, "specularMap"), 2)
            glUniform1i(use_specular_map_loc, 1)
        else:
            glUniform1i(use_specular_map_loc, 0)
            
        # Bind night map if available (Earth)
        if hasattr(body, 'night_texture_id') and body.night_texture_id:
            glActiveTexture(GL_TEXTURE3)
            glBindTexture(GL_TEXTURE_2D, body.night_texture_id)
            glUniform1i(glGetUniformLocation(shader, "nightMap"), 3)
            glUniform1i(use_night_map_loc, 1)
        else:
            glUniform1i(use_night_map_loc, 0)
            
        # Bind clouds map if available (Earth, Jupiter)
        if hasattr(body, 'clouds_texture_id') and body.clouds_texture_id:
            glActiveTexture(GL_TEXTURE4)
            glBindTexture(GL_TEXTURE_2D, body.clouds_texture_id)
            glUniform1i(glGetUniformLocation(shader, "cloudsMap"), 4)
            glUniform1i(use_clouds_map_loc, 1)
        else:
            glUniform1i(use_clouds_map_loc, 0)
            
        # Set atmospheric flag for gas giants and Venus
        is_atmospheric = body.name in ["Venus", "Jupiter", "Saturn", "Uranus", "Neptune"]
        glUniform1i(is_atmospheric_loc, 1 if is_atmospheric else 0)
        
        # Use enhanced sphere if available, otherwise fall back to regular sphere
        if 'enhanced_sphere' in renderer.vaos:
            glBindVertexArray(renderer.vaos['enhanced_sphere'])
            glDrawElements(GL_TRIANGLES, renderer.enhanced_sphere_vertex_count, GL_UNSIGNED_INT, None)
        else:
            glBindVertexArray(renderer.vaos['sphere'])
            glDrawElements(GL_TRIANGLES, renderer.sphere_vertex_count, GL_UNSIGNED_INT, None)
        
        # Unbind
        glBindVertexArray(0)
        glUseProgram(0)
        
        # Draw rings if body has them
        if body.has_rings:
            renderer.draw_rings(body, model_matrix, view_matrix, projection_matrix)
            
        return True
    except Exception as e:
        print(f"Error drawing body with advanced shader: {e}")
        return False
