#!/usr/bin/env python3
# renderer_high_res.py - High-resolution rendering extensions with safety measures

import os
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
import logging
from constants import SHADER_DIR, TEXTURE_DIR
import texture_mapping

# Configure logging
logger = logging.getLogger("RendererHighRes")

def initialize_renderer(renderer):
    """
    Initialize the renderer with high-resolution extensions
    
    Args:
        renderer: The renderer object to initialize
        
    Returns:
        bool: True if successful, False otherwise
    """
    assert renderer is not None, "Renderer cannot be None"
    
    try:
        # Save original draw_bodies method for fallback
        if not hasattr(renderer, '_draw_body_standard'):
            renderer._draw_body_standard = renderer.draw_bodies
        
        # Add new functionality
        renderer.use_advanced_shaders = ensure_advanced_shaders()
        renderer.enhanced_sphere_vertex_count = 0
        
        # Check if we have high-resolution textures
        has_textures, missing, has_high_res = texture_mapping.check_textures()
        
        if missing:
            logger.warning(f"Missing textures: {', '.join(missing)}")
            
        if has_high_res:
            logger.info("Using high-resolution textures")
            
        # Apply enhanced textures to simulation bodies
        texture_mapping.apply_texture_mappings(renderer.simulation.bodies)
        
        # Create enhanced resources if using advanced shaders
        if renderer.use_advanced_shaders:
            logger.info("Initializing advanced rendering features")
            # Add enhanced sphere for normal mapping
            enhanced_vao, vertex_count = create_enhanced_sphere(renderer, 32, 32)
            if enhanced_vao:
                renderer.vaos['enhanced_sphere'] = enhanced_vao
                renderer.enhanced_sphere_vertex_count = vertex_count
            
        logger.info("High-resolution renderer initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize high-resolution renderer: {e}")
        return False

def ensure_advanced_shaders():
    """
    Ensure advanced shaders are available
    
    Returns:
        bool: True if advanced shaders are available, False otherwise
    """
    try:
        # Check if shader directory exists
        if not os.path.exists(SHADER_DIR):
            os.makedirs(SHADER_DIR)
        
        # Required shader files
        vertex_path = os.path.join(SHADER_DIR, "advanced_planet_vertex.glsl")
        fragment_path = os.path.join(SHADER_DIR, "advanced_planet_fragment.glsl")
        
        # Check if both files exist
        if os.path.exists(vertex_path) and os.path.exists(fragment_path):
            return True
            
        # Look for alternative locations (the .txt versions)
        alt_vertex = os.path.join(SHADER_DIR, "advanced-planet-shader.txt")
        alt_fragment = os.path.join(SHADER_DIR, "advanced-planet-fragment.txt")
        
        if os.path.exists(alt_vertex) and os.path.exists(alt_fragment):
            # Copy to the correct locations
            try:
                with open(alt_vertex, 'r') as src, open(vertex_path, 'w') as dst:
                    dst.write(src.read())
                with open(alt_fragment, 'r') as src, open(fragment_path, 'w') as dst:
                    dst.write(src.read())
                logger.info("Copied advanced shaders to standard locations")
                return True
            except Exception as e:
                logger.error(f"Failed to copy shaders: {e}")
        
        logger.warning("Advanced shaders not available")
        return False
    except Exception as e:
        logger.error(f"Error checking advanced shaders: {e}")
        return False

def create_enhanced_sphere(renderer, stacks, slices):
    """
    Create an enhanced sphere mesh with tangent and bitangent data
    
    Args:
        renderer: The renderer object
        stacks (int): Number of horizontal divisions
        slices (int): Number of vertical divisions
        
    Returns:
        tuple: (vao, vertex_count)
    """
    assert renderer is not None, "Renderer cannot be None"
    assert isinstance(stacks, int) and stacks > 0, "Stacks must be a positive integer"
    assert isinstance(slices, int) and slices > 0, "Slices must be a positive integer"
    
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
        tangents, bitangents = generate_tangents_bitangents(
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
        logger.error(f"Error creating enhanced sphere: {e}")
        return 0, 0

def generate_tangents_bitangents(vertices, uvs, normals, indices):
    """
    Generate tangent and bitangent vectors for normal mapping
    
    Args:
        vertices (list): List of vertex positions
        uvs (list): List of texture coordinates
        normals (list): List of normal vectors
        indices (list): List of triangle indices
        
    Returns:
        tuple: (tangents, bitangents)
    """
    assert len(vertices) % 3 == 0, "Vertices must be a multiple of 3"
    assert len(uvs) % 2 == 0, "UVs must be a multiple of 2"
    assert len(normals) % 3 == 0, "Normals must be a multiple of 3"
    assert len(indices) % 3 == 0, "Indices must be a multiple of 3"
    
    # Initialize arrays
    num_vertices = len(vertices) // 3
    tangents = np.zeros((num_vertices, 3), dtype=np.float32)
    bitangents = np.zeros((num_vertices, 3), dtype=np.float32)
    
    # Process each triangle
    for i in range(0, len(indices), 3):
        # Ensure we don't exceed array bounds
        if i + 2 >= len(indices):
            break
            
        # Get indices for the triangle
        i0 = indices[i]
        i1 = indices[i+1]
        i2 = indices[i+2]
        
        # Validate indices
        if i0 >= num_vertices or i1 >= num_vertices or i2 >= num_vertices:
            continue
        
        # Get vertex positions
        v0 = np.array([vertices[i0*3], vertices[i0*3+1], vertices[i0*3+2]])
        v1 = np.array([vertices[i1*3], vertices[i1*3+1], vertices[i1*3+2]])
        v2 = np.array([vertices[i2*3], vertices[i2*3+1], vertices[i2*3+2]])
        
        # Get texture coordinates
        uv0 = np.array([uvs[i0*2], uvs[i0*2+1]])
        uv1 = np.array([uvs[i1*2], uvs[i1*2+1]])
        uv2 = np.array([uvs[i2*2], uvs[i2*2+1]])
        
        # Calculate edges
        e1 = v1 - v0
        e2 = v2 - v0
        
        # Calculate UV deltas
        deltaUV1 = uv1 - uv0
        deltaUV2 = uv2 - uv0
        
        # Calculate tangent and bitangent
        try:
            # Handle zero denominator
            denom = (deltaUV1[0] * deltaUV2[1] - deltaUV1[1] * deltaUV2[0])
            if abs(denom) < 1e-6:
                # Use arbitrary perpendicular vectors
                n = np.array([normals[i0*3], normals[i0*3+1], normals[i0*3+2]])
                t = np.zeros(3)
                if abs(n[0]) < abs(n[1]):
                    if abs(n[0]) < abs(n[2]):
                        t[0] = 1.0
                    else:
                        t[2] = 1.0
                else:
                    if abs(n[1]) < abs(n[2]):
                        t[1] = 1.0
                    else:
                        t[2] = 1.0
                t = np.cross(n, t)
                t = t / np.linalg.norm(t)
                b = np.cross(n, t)
            else:
                r = 1.0 / denom
                tangent = (e1 * deltaUV2[1] - e2 * deltaUV1[1]) * r
                bitangent = (e2 * deltaUV1[0] - e1 * deltaUV2[0]) * r
                
                # Normalize
                t_len = np.linalg.norm(tangent)
                b_len = np.linalg.norm(bitangent)
                
                t = tangent / t_len if t_len > 1e-6 else np.array([1.0, 0.0, 0.0])
                b = bitangent / b_len if b_len > 1e-6 else np.array([0.0, 1.0, 0.0])
            
            # Add to arrays
            tangents[i0] += t
            tangents[i1] += t
            tangents[i2] += t
            
            bitangents[i0] += b
            bitangents[i1] += b
            bitangents[i2] += b
        except Exception as e:
            logger.debug(f"Error calculating tangents for triangle {i//3}: {e}")
    
    # Normalize each tangent and bitangent
    for i in range(num_vertices):
        t_norm = np.linalg.norm(tangents[i])
        b_norm = np.linalg.norm(bitangents[i])
        
        tangents[i] = tangents[i] / t_norm if t_norm > 1e-6 else np.array([1.0, 0.0, 0.0])
        bitangents[i] = bitangents[i] / b_norm if b_norm > 1e-6 else np.array([0.0, 1.0, 0.0])
    
    # Flatten arrays for OpenGL
    return tangents.flatten(), bitangents.flatten()

def draw_body_with_advanced_shader(renderer, body, view_matrix, projection_matrix):
    """
    Draw a celestial body using the advanced shader
    
    Args:
        renderer: The renderer object
        body: The celestial body to draw
        view_matrix: The view matrix
        projection_matrix: The projection matrix
        
    Returns:
        bool: True if successful, False otherwise
    """
    assert renderer is not None, "Renderer cannot be None"
    assert body is not None, "Body cannot be None"
    assert view_matrix is not None, "View matrix cannot be None"
    assert projection_matrix is not None, "Projection matrix cannot be None"
    
    # Skip if advanced rendering is disabled
    if not hasattr(renderer, 'use_advanced_shaders') or not renderer.use_advanced_shaders:
        return False
        
    # Skip if body has invalid position
    if not np.all(np.isfinite(body.position)):
        return False
        
    try:
        # Get advanced shader
        shader = renderer.shader_manager.get_shader("advanced_planet")
        if not shader:
            return False
            
        glUseProgram(shader)
        
        # Set uniform values
        light_pos_loc = glGetUniformLocation(shader, "lightPos")
        view_pos_loc = glGetUniformLocation(shader, "viewPos")
        
        # Find the sun's position (default to origin if not found)
        sun_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        for b in renderer.simulation.bodies:
            if b.name == "Sun":
                sun_pos = b.position / SCALE_FACTOR
                break
        
        # Set light and view positions
        glUniform3f(light_pos_loc, sun_pos[0], sun_pos[1], sun_pos[2])
        
        # Calculate camera position
        camera = renderer.camera
        eye_pos = camera.target - np.array([
            camera.distance * np.sin(camera.x_angle) * np.cos(camera.y_angle),
            camera.distance * np.sin(camera.y_angle),
            camera.distance * np.cos(camera.x_angle) * np.cos(camera.y_angle)
        ])
        glUniform3f(view_pos_loc, eye_pos[0], eye_pos[1], eye_pos[2])
        
        # Set matrices
        glUniformMatrix4fv(glGetUniformLocation(shader, "projection"), 1, GL_FALSE, projection_matrix)
        glUniformMatrix4fv(glGetUniformLocation(shader, "view"), 1, GL_FALSE, view_matrix)
        
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
        
        glUniformMatrix4fv(glGetUniformLocation(shader, "model"), 1, GL_FALSE, model_matrix)
        
        # Set material properties
        glUniform3f(glGetUniformLocation(shader, "baseColor"), 
                    body.color[0], body.color[1], body.color[2])
        glUniform1f(glGetUniformLocation(shader, "ambientStrength"), 0.3)
        glUniform1f(glGetUniformLocation(shader, "specularStrength"), 0.5)
        glUniform1f(glGetUniformLocation(shader, "shininess"), 32.0)
        
        # Set time for animations
        import pygame
        current_time = pygame.time.get_ticks() / 1000.0
        glUniform1f(glGetUniformLocation(shader, "time"), current_time)
        
        # Set texture flags with safe defaults
        glUniform1i(glGetUniformLocation(shader, "useTexture"), 0)
        glUniform1i(glGetUniformLocation(shader, "useNormalMap"), 0)
        glUniform1i(glGetUniformLocation(shader, "useSpecularMap"), 0)
        glUniform1i(glGetUniformLocation(shader, "useNightMap"), 0)
        glUniform1i(glGetUniformLocation(shader, "useCloudsMap"), 0)
        
        # Set atmospheric flag for gas giants and Venus
        is_atmospheric = body.name in ["Venus", "Jupiter", "Saturn", "Uranus", "Neptune"]
        glUniform1i(glGetUniformLocation(shader, "isAtmospheric"), 1 if is_atmospheric else 0)
        
        # Bind diffuse texture if available
        if hasattr(body, 'texture_id') and body.texture_id:
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, body.texture_id)
            glUniform1i(glGetUniformLocation(shader, "diffuseMap"), 0)
            glUniform1i(glGetUniformLocation(shader, "useTexture"), 1)
        
        # Bind normal map if available
        if hasattr(body, 'normal_map_id') and body.normal_map_id:
            glActiveTexture(GL_TEXTURE1)
            glBindTexture(GL_TEXTURE_2D, body.normal_map_id)
            glUniform1i(glGetUniformLocation(shader, "normalMap"), 1)
            glUniform1i(glGetUniformLocation(shader, "useNormalMap"), 1)
            
        # Bind specular map if available
        if hasattr(body, 'specular_map_id') and body.specular_map_id:
            glActiveTexture(GL_TEXTURE2)
            glBindTexture(GL_TEXTURE_2D, body.specular_map_id)
            glUniform1i(glGetUniformLocation(shader, "specularMap"), 2)
            glUniform1i(glGetUniformLocation(shader, "useSpecularMap"), 1)
            
        # Bind night map if available
        if hasattr(body, 'night_texture_id') and body.night_texture_id:
            glActiveTexture(GL_TEXTURE3)
            glBindTexture(GL_TEXTURE_2D, body.night_texture_id)
            glUniform1i(glGetUniformLocation(shader, "nightMap"), 3)
            glUniform1i(glGetUniformLocation(shader, "useNightMap"), 1)
            
        # Bind clouds map if available
        if hasattr(body, 'clouds_texture_id') and body.clouds_texture_id:
            glActiveTexture(GL_TEXTURE4)
            glBindTexture(GL_TEXTURE_2D, body.clouds_texture_id)
            glUniform1i(glGetUniformLocation(shader, "cloudsMap"), 4)
            glUniform1i(glGetUniformLocation(shader, "useCloudsMap"), 1)
            
        # Draw with enhanced sphere if available, fall back to regular sphere
        if hasattr(renderer, 'enhanced_sphere_vertex_count') and \
           renderer.enhanced_sphere_vertex_count > 0 and \
           'enhanced_sphere' in renderer.vaos:
            glBindVertexArray(renderer.vaos['enhanced_sphere'])
            glDrawElements(GL_TRIANGLES, renderer.enhanced_sphere_vertex_count, GL_UNSIGNED_INT, None)
        elif 'sphere' in renderer.vaos:
            glBindVertexArray(renderer.vaos['sphere'])
            glDrawElements(GL_TRIANGLES, renderer.sphere_vertex_count, GL_UNSIGNED_INT, None)
        else:
            logger.error("No sphere mesh available for rendering")
            glUseProgram(0)
            return False
        
        # Clean up state
        glBindVertexArray(0)
        glUseProgram(0)
        
        # Draw rings if body has them
        if body.has_rings:
            renderer.draw_rings(body, model_matrix, view_matrix, projection_matrix)
            
        return True
    except Exception as e:
        logger.error(f"Error drawing body with advanced shader: {e}")
        # Ensure OpenGL state is clean
        glBindVertexArray(0)
        glUseProgram(0)
        return False