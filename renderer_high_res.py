#!/usr/bin/env python3
# renderer_high_res.py - High-resolution rendering extension for planetary simulation (FIXED VERSION)

import os
import numpy as np
import pygame
from OpenGL.GL import *
from OpenGL.GLU import *
import logging
from constants import *
from texture_loader import TextureLoader
import debug_utils

# Initialize logging
logger = logging.getLogger("GravitySim.HighRes")

def initialize_renderer(renderer):
    """
    Initialize high-resolution rendering features
    
    Args:
        renderer: The main renderer object
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        debug_utils.debug_print("Initializing high-resolution renderer")
        
        # Check if the renderer is properly initialized
        if not hasattr(renderer, 'initialization_successful') or not renderer.initialization_successful:
            logger.warning("Cannot initialize high-res renderer: base renderer not initialized")
            return False
        
        # Add high-res flag
        renderer.use_advanced_shaders = False
        
        # Load advanced shaders
        if load_advanced_shaders(renderer):
            debug_utils.debug_print("Advanced shaders loaded successfully")
            renderer.use_advanced_shaders = True
        else:
            debug_utils.debug_print("Using standard shaders (advanced shaders not available)")
            return False
        
        # Create enhanced sphere mesh
        debug_utils.debug_print("Creating enhanced sphere mesh")
        vao, count = create_enhanced_sphere(renderer, SPHERE_DETAIL, SPHERE_DETAIL)
        if vao and count > 0:
            debug_utils.debug_print(f"Enhanced sphere created with {count} vertices")
            renderer.vaos['enhanced_sphere'] = vao
            renderer.enhanced_sphere_vertex_count = count
        else:
            debug_utils.debug_print("Failed to create enhanced sphere, using standard sphere")
            return False
        
        # Check for high-resolution textures
        if os.path.exists(os.path.join(TEXTURE_DIR, "2k_earth_daymap.jpg")):
            debug_utils.debug_print("High-resolution textures found")
        else:
            debug_utils.debug_print("High-resolution textures not found")
        
        # Load special textures for planets
        assign_special_textures(renderer)
        
        debug_utils.debug_print("High-resolution renderer initialization complete")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize high-resolution renderer: {e}")
        import traceback
        traceback.print_exc()
        # Revert any changes made
        renderer.use_advanced_shaders = False
        return False

def load_advanced_shaders(renderer):
    """
    Load advanced shaders for high-resolution textures
    
    Args:
        renderer: The renderer object
        
    Returns:
        bool: True if successful, False otherwise
    """
    if not hasattr(renderer, 'shader_manager'):
        logger.error("Renderer has no shader manager")
        return False
    
    try:
        # Try different files that might contain advanced shaders
        filenames = [
            ("advanced_planet_vertex.glsl", "advanced_planet_fragment.glsl"),
            ("advanced-planet-shader.txt", "advanced-planet-fragment.txt")
        ]
        
        for vertex_file, fragment_file in filenames:
            # Check if files exist
            vertex_path = os.path.join(SHADER_DIR, vertex_file)
            fragment_path = os.path.join(SHADER_DIR, fragment_file)
            
            if os.path.exists(vertex_path) and os.path.exists(fragment_path):
                debug_utils.debug_print(f"Found shader files: {vertex_file}, {fragment_file}")
                
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
                    debug_utils.debug_print(f"Advanced planet shader compiled: {shader_program}")
                    return True
        
        # If no files found or compilation failed, use embedded shaders
        debug_utils.debug_print("Using embedded advanced shaders")
        
        # Basic vertex shader that supports normal mapping
        vertex_src = """
        #version 330 core
        
        layout(location = 0) in vec3 position;
        layout(location = 1) in vec3 normal;
        layout(location = 2) in vec2 texCoord;
        layout(location = 3) in vec3 tangent;
        layout(location = 4) in vec3 bitangent;
        
        out vec3 fragNormal;
        out vec3 fragPosition;
        out vec2 fragTexCoord;
        out mat3 TBN;
        
        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;
        uniform bool useNormalMap;
        
        void main() {
            // Calculate fragment position in world space
            fragPosition = vec3(model * vec4(position, 1.0));
            
            // Calculate normal in world space
            mat3 normalMatrix = transpose(inverse(mat3(model)));
            fragNormal = normalize(normalMatrix * normal);
            
            // Pass through texture coordinates
            fragTexCoord = texCoord;
            
            // Calculate TBN matrix for normal mapping
            if (useNormalMap) {
                vec3 T = normalize(normalMatrix * tangent);
                vec3 B = normalize(normalMatrix * bitangent);
                vec3 N = fragNormal;
                
                // Create TBN matrix
                TBN = mat3(T, B, N);
            } else {
                // Dummy TBN matrix
                TBN = mat3(1.0);
            }
            
            // Calculate final vertex position
            gl_Position = projection * view * model * vec4(position, 1.0);
        }
        """
        
        # Basic fragment shader with advanced features
        fragment_src = """
        #version 330 core
        
        in vec3 fragNormal;
        in vec3 fragPosition;
        in vec2 fragTexCoord;
        in mat3 TBN;
        
        out vec4 fragColor;
        
        uniform vec3 lightPos;
        uniform vec3 viewPos;
        uniform vec3 baseColor;
        uniform float time;
        
        uniform sampler2D diffuseMap;
        uniform sampler2D normalMap;
        uniform sampler2D specularMap;
        uniform sampler2D nightMap;
        uniform sampler2D cloudsMap;
        
        uniform bool useTexture;
        uniform bool useNormalMap;
        uniform bool useSpecularMap;
        uniform bool useNightMap;
        uniform bool useCloudsMap;
        uniform bool isAtmospheric;
        
        void main() {
            // Base color
            vec3 color;
            if (useTexture) {
                color = texture(diffuseMap, fragTexCoord).rgb;
            } else {
                color = baseColor;
            }
            
            // Normal mapping
            vec3 normal;
            if (useNormalMap) {
                // Sample normal map and transform to world space
                vec3 normalColor = texture(normalMap, fragTexCoord).rgb * 2.0 - 1.0;
                normal = normalize(TBN * normalColor);
            } else {
                normal = normalize(fragNormal);
            }
            
            // Basic lighting
            vec3 lightDir = normalize(lightPos - fragPosition);
            float diff = max(dot(normal, lightDir), 0.0);
            
            // Ambient component
            float ambientStrength = 0.2;
            vec3 ambient = ambientStrength * color;
            
            // Diffuse component
            vec3 diffuse = diff * color;
            
            // Specular component
            vec3 viewDir = normalize(viewPos - fragPosition);
            vec3 reflectDir = reflect(-lightDir, normal);
            float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
            float specularStrength = 0.5;
            if (useSpecularMap) {
                specularStrength *= texture(specularMap, fragTexCoord).r;
            }
            vec3 specular = specularStrength * spec * vec3(1.0);
            
            // Combine components
            vec3 result = ambient + diffuse + specular;
            
            // Night lights for Earth
            if (useNightMap) {
                float nightFactor = 1.0 - max(dot(normal, lightDir), 0.0);
                nightFactor = smoothstep(0.1, 0.5, nightFactor);
                vec3 nightColor = texture(nightMap, fragTexCoord).rgb;
                result = mix(result, nightColor, nightFactor * 0.7);
            }
            
            // Cloud layer
            if (useCloudsMap) {
                // Animate clouds slightly
                vec2 cloudCoord = fragTexCoord;
                cloudCoord.x = mod(cloudCoord.x + time * 0.005, 1.0);
                vec4 cloudColor = texture(cloudsMap, cloudCoord);
                
                // Blend clouds based on light direction
                float cloudLighting = max(dot(normal, lightDir), 0.1);
                vec3 litClouds = cloudColor.rgb * cloudLighting;
                
                // Mix clouds with surface
                result = mix(result, litClouds, cloudColor.a * 0.5);
            }
            
            // Atmospheric effects for gas giants
            if (isAtmospheric) {
                // Edge glow effect
                float fresnel = pow(1.0 - max(dot(normal, viewDir), 0.0), 3.0);
                vec3 atmosphereColor = mix(color, vec3(0.7, 0.8, 1.0), 0.5);
                result = mix(result, atmosphereColor, fresnel * 0.4);
            }
            
            // Final color
            fragColor = vec4(result, 1.0);
        }
        """
        
        # Compile embedded shaders
        shader_program = renderer.shader_manager.compile_shader(vertex_src, fragment_src)
        if shader_program:
            renderer.shader_manager.shaders["advanced_planet"] = shader_program
            debug_utils.debug_print("Embedded advanced shader compiled successfully")
            return True
            
        logger.warning("All attempts to create advanced shader failed")
        return False
    except Exception as e:
        logger.error(f"Error loading advanced shaders: {e}")
        return False

def assign_special_textures(renderer):
    """
    Assign special textures (normal maps, specular maps, etc.) to bodies
    
    Args:
        renderer: The renderer object
    """
    if not hasattr(renderer, 'simulation') or not hasattr(renderer, 'texture_manager'):
        return
    
    try:
        for body in renderer.simulation.bodies:
            # Earth special textures
            if body.name == "Earth":
                # Normal map
                normal_map = os.path.join(TEXTURE_DIR, "2k_earth_normal_map.jpg")
                if os.path.exists(normal_map):
                    body.normal_map_id = renderer.texture_manager.load_texture("2k_earth_normal_map.jpg")
                
                # Night lights
                night_map = os.path.join(TEXTURE_DIR, "2k_earth_nightmap.jpg")
                if os.path.exists(night_map):
                    body.night_texture_id = renderer.texture_manager.load_texture("2k_earth_nightmap.jpg")
                
                # Clouds
                clouds_map = os.path.join(TEXTURE_DIR, "2k_earth_clouds.jpg")
                if os.path.exists(clouds_map):
                    body.clouds_texture_id = renderer.texture_manager.load_texture("2k_earth_clouds.jpg")
                
                # Specular map
                spec_map = os.path.join(TEXTURE_DIR, "2k_earth_specular_map.jpg")
                if os.path.exists(spec_map):
                    body.specular_map_id = renderer.texture_manager.load_texture("2k_earth_specular_map.jpg")
            
            # Moon special textures
            elif body.name == "Moon":
                normal_map = os.path.join(TEXTURE_DIR, "2k_moon_normal_map.jpg")
                if os.path.exists(normal_map):
                    body.normal_map_id = renderer.texture_manager.load_texture("2k_moon_normal_map.jpg")
            
            # Mars special textures
            elif body.name == "Mars":
                normal_map = os.path.join(TEXTURE_DIR, "2k_mars_normal_map.jpg")
                if os.path.exists(normal_map):
                    body.normal_map_id = renderer.texture_manager.load_texture("2k_mars_normal_map.jpg")
            
            # Jupiter special textures
            elif body.name == "Jupiter":
                clouds_map = os.path.join(TEXTURE_DIR, "2k_jupiter_clouds.jpg")
                if os.path.exists(clouds_map):
                    body.clouds_texture_id = renderer.texture_manager.load_texture("2k_jupiter_clouds.jpg")
            
    except Exception as e:
        logger.error(f"Error assigning special textures: {e}")

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
        
        # Generate tangents and bitangents using a simplified approach
        tangents = []
        bitangents = []
        
        # Simple approximation for tangents and bitangents
        for i in range(0, len(vertices), 3):
            # Extract vertex position
            vx, vy, vz = vertices[i], vertices[i+1], vertices[i+2]
            
            # Calculate tangent and bitangent
            # For a sphere, the tangent is roughly in the direction of decreasing theta
            # The bitangent is roughly in the direction of decreasing phi
            t_length = np.sqrt(vx*vx + vz*vz)
            if t_length > 0.001:
                tx, ty, tz = -vz/t_length, 0, vx/t_length
            else:
                tx, ty, tz = 1, 0, 0
                
            tangents.extend([tx, ty, tz])
            
            # Bitangent = normal × tangent
            nx, ny, nz = normals[i], normals[i+1], normals[i+2]
            bx = ny*tz - nz*ty
            by = nz*tx - nx*tz
            bz = nx*ty - ny*tx
            
            b_length = np.sqrt(bx*bx + by*by + bz*bz)
            if b_length > 0.001:
                bx, by, bz = bx/b_length, by/b_length, bz/b_length
            else:
                bx, by, bz = 0, 1, 0
                
            bitangents.extend([bx, by, bz])
        
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
        
        debug_utils.check_gl_errors("create_enhanced_sphere")
        
        return vao, len(indices)
    except Exception as e:
        logger.error(f"Error creating enhanced sphere mesh: {e}")
        import traceback
        traceback.print_exc()
        return 0, 0

def draw_body_with_advanced_shader(renderer, body, view_matrix, projection_matrix):
    """
    Draw a celestial body using the advanced shader with normal mapping and special effects
    
    Args:
        renderer: The renderer object
        body: The celestial body to draw
        view_matrix: The view matrix
        projection_matrix: The projection matrix
        
    Returns:
        bool: True if drawn successfully, False otherwise
    """
    if not hasattr(renderer, 'use_advanced_shaders') or not renderer.use_advanced_shaders:
        return False
        
    try:
        # Get shader
        shader = renderer.shader_manager.get_shader("advanced_planet")
        if not shader:
            debug_utils.debug_print(f"Advanced shader not available for {body.name}")
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
        
        # Set light position
        if light_pos_loc != -1:
            glUniform3f(light_pos_loc, sun_pos[0], sun_pos[1], sun_pos[2])
        
        # Calculate view position
        eye_pos = renderer.camera.target - np.array([
            renderer.camera.distance * np.sin(renderer.camera.x_angle) * np.cos(renderer.camera.y_angle),
            renderer.camera.distance * np.sin(renderer.camera.y_angle),
            renderer.camera.distance * np.cos(renderer.camera.x_angle) * np.cos(renderer.camera.y_angle)
        ])
        
        # Set view position
        if view_pos_loc != -1:
            glUniform3f(view_pos_loc, eye_pos[0], eye_pos[1], eye_pos[2])
        
        # Set matrices
        proj_loc = glGetUniformLocation(shader, "projection")
        view_loc = glGetUniformLocation(shader, "view")
        model_loc = glGetUniformLocation(shader, "model")
        
        if proj_loc != -1:
            glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection_matrix)
        if view_loc != -1:
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
        
        if model_loc != -1:
            glUniformMatrix4fv(model_loc, 1, GL_FALSE, model_matrix)
        
        # Set material properties
        base_color_loc = glGetUniformLocation(shader, "baseColor")
        ambient_str_loc = glGetUniformLocation(shader, "ambientStrength")
        specular_str_loc = glGetUniformLocation(shader, "specularStrength")
        shininess_loc = glGetUniformLocation(shader, "shininess")
        
        if base_color_loc != -1:
            glUniform3f(base_color_loc, body.color[0], body.color[1], body.color[2])
        if ambient_str_loc != -1:
            glUniform1f(ambient_str_loc, 0.3)
        if specular_str_loc != -1:
            glUniform1f(specular_str_loc, 0.5)
        if shininess_loc != -1:
            glUniform1f(shininess_loc, 32.0)
        
        # Set time for animations
        time_loc = glGetUniformLocation(shader, "time")
        if time_loc != -1:
            if pygame.get_init():
                glUniform1f(time_loc, float(pygame.time.get_ticks()) / 1000.0)
            else:
                glUniform1f(time_loc, 0.0)
        
        # Set feature toggles
        use_texture_loc = glGetUniformLocation(shader, "useTexture")
        use_normal_map_loc = glGetUniformLocation(shader, "useNormalMap")
        use_specular_map_loc = glGetUniformLocation(shader, "useSpecularMap")
        use_night_map_loc = glGetUniformLocation(shader, "useNightMap")
        use_clouds_map_loc = glGetUniformLocation(shader, "useCloudsMap")
        is_atmospheric_loc = glGetUniformLocation(shader, "isAtmospheric")
        
        # Bind diffuse texture
        if hasattr(body, 'texture_id') and body.texture_id:
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, body.texture_id)
            
            tex_loc = glGetUniformLocation(shader, "diffuseMap")
            if tex_loc != -1:
                glUniform1i(tex_loc, 0)
                
            if use_texture_loc != -1:
                glUniform1i(use_texture_loc, 1)
        else:
            if use_texture_loc != -1:
                glUniform1i(use_texture_loc, 0)
        
        # Bind normal map if available
        if hasattr(body, 'normal_map_id') and body.normal_map_id:
            glActiveTexture(GL_TEXTURE1)
            glBindTexture(GL_TEXTURE_2D, body.normal_map_id)
            
            normal_loc = glGetUniformLocation(shader, "normalMap")
            if normal_loc != -1:
                glUniform1i(normal_loc, 1)
                
            if use_normal_map_loc != -1:
                glUniform1i(use_normal_map_loc, 1)
        else:
            if use_normal_map_loc != -1:
                glUniform1i(use_normal_map_loc, 0)
            
        # Bind specular map if available
        if hasattr(body, 'specular_map_id') and body.specular_map_id:
            glActiveTexture(GL_TEXTURE2)
            glBindTexture(GL_TEXTURE_2D, body.specular_map_id)
            
            spec_loc = glGetUniformLocation(shader, "specularMap")
            if spec_loc != -1:
                glUniform1i(spec_loc, 2)
                
            if use_specular_map_loc != -1:
                glUniform1i(use_specular_map_loc, 1)
        else:
            if use_specular_map_loc != -1:
                glUniform1i(use_specular_map_loc, 0)
            
        # Bind night map if available (Earth)
        if hasattr(body, 'night_texture_id') and body.night_texture_id:
            glActiveTexture(GL_TEXTURE3)
            glBindTexture(GL_TEXTURE_2D, body.night_texture_id)
            
            night_loc = glGetUniformLocation(shader, "nightMap")
            if night_loc != -1:
                glUniform1i(night_loc, 3)
                
            if use_night_map_loc != -1:
                glUniform1i(use_night_map_loc, 1)
        else:
            if use_night_map_loc != -1:
                glUniform1i(use_night_map_loc, 0)
            
        # Bind clouds map if available (Earth, Jupiter)
        if hasattr(body, 'clouds_texture_id') and body.clouds_texture_id:
            glActiveTexture(GL_TEXTURE4)
            glBindTexture(GL_TEXTURE_2D, body.clouds_texture_id)
            
            clouds_loc = glGetUniformLocation(shader, "cloudsMap")
            if clouds_loc != -1:
                glUniform1i(clouds_loc, 4)
                
            if use_clouds_map_loc != -1:
                glUniform1i(use_clouds_map_loc, 1)
        else:
            if use_clouds_map_loc != -1:
                glUniform1i(use_clouds_map_loc, 0)
            
        # Set atmospheric flag for gas giants and Venus
        is_atmospheric = body.name in ["Venus", "Jupiter", "Saturn", "Uranus", "Neptune"]
        if is_atmospheric_loc != -1:
            glUniform1i(is_atmospheric_loc, 1 if is_atmospheric else 0)
        
        # Draw body with enhanced sphere if available
        if 'enhanced_sphere' in renderer.vaos and hasattr(renderer, 'enhanced_sphere_vertex_count'):
            glBindVertexArray(renderer.vaos['enhanced_sphere'])
            glDrawElements(GL_TRIANGLES, renderer.enhanced_sphere_vertex_count, GL_UNSIGNED_INT, None)
            glBindVertexArray(0)
        # Fallback to regular sphere
        elif hasattr(renderer, 'sphere_vao') and renderer.sphere_vao:
            glBindVertexArray(renderer.sphere_vao)
            glDrawElements(GL_TRIANGLES, renderer.sphere_vertex_count, GL_UNSIGNED_INT, None)
            glBindVertexArray(0)
        else:
            debug_utils.debug_print(f"No sphere mesh available for {body.name}")
            glUseProgram(0)
            return False
        
        # Unbind shader
        glUseProgram(0)
        
        # Draw rings if body has them
        if hasattr(body, 'has_rings') and body.has_rings:
            renderer.draw_rings(body, model_matrix, view_matrix, projection_matrix)
        
        debug_utils.check_gl_errors(f"draw_body_with_advanced_shader for {body.name}")
        return True
    except Exception as e:
        logger.error(f"Error drawing body with advanced shader: {e}")
        import traceback
        traceback.print_exc()
        glUseProgram(0)
        return False