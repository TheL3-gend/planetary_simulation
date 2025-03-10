#!/usr/bin/env python3
# fix_simulation.py - Utility script to fix simulation issues

import os
import sys
import logging
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("FixSimulation")

def check_python_version():
    """Check if Python version is sufficient"""
    min_version = (3, 6)
    current = sys.version_info[:2]
    
    if current < min_version:
        logger.error(f"Python {min_version[0]}.{min_version[1]} or newer required")
        return False
    
    logger.info(f"Python version: {sys.version.split()[0]}")
    return True

def check_required_modules():
    """Check if required modules are installed"""
    required_modules = ["pygame", "numpy", "OpenGL", "OpenGL.GL", "OpenGL.GLU"]
    missing_modules = []
    
    for module in required_modules:
        try:
            __import__(module.split(".")[0])
        except ImportError:
            missing_modules.append(module.split(".")[0])
    
    if missing_modules:
        logger.error(f"Missing modules: {', '.join(missing_modules)}")
        print("Install with:")
        print(f"pip install {' '.join(missing_modules)}")
        return False
    
    logger.info("All required modules installed")
    return True

def check_directory_structure():
    """Check and create necessary directories"""
    required_dirs = ["textures", "shaders", "fonts"]
    
    for directory in required_dirs:
        if not os.path.exists(directory):
            try:
                os.makedirs(directory)
                logger.info(f"Created directory: {directory}")
            except Exception as e:
                logger.error(f"Could not create directory {directory}: {e}")
                return False
    
    logger.info("Directory structure verified")
    return True

def check_shader_files():
    """Check if shader files exist"""
    required_shaders = [
        "planet_vertex.glsl", "planet_fragment.glsl",
        "ring_vertex.glsl", "ring_fragment.glsl",
        "trail_vertex.glsl", "trail_fragment.glsl"
    ]
    
    missing_shaders = []
    
    for shader in required_shaders:
        if not os.path.exists(os.path.join("shaders", shader)):
            missing_shaders.append(shader)
    
    if missing_shaders:
        logger.warning(f"Missing shader files: {', '.join(missing_shaders)}")
        if input("Would you like to create default shader files? (y/n): ").lower() == 'y':
            create_default_shaders(missing_shaders)
        else:
            logger.warning("Shaders not created, simulation may not work correctly")
            return False
    
    logger.info("Shader files verified")
    return True

def check_texture_files():
    """Check if essential texture files exist"""
    essential_textures = [
        "sun.jpg", "earth.jpg", "mercury.jpg", "venus.jpg", "mars.jpg",
        "jupiter.jpg", "saturn.jpg", "uranus.jpg", "neptune.jpg", "moon.jpg"
    ]
    
    missing_textures = []
    
    for texture in essential_textures:
        if not os.path.exists(os.path.join("textures", texture)):
            missing_textures.append(texture)
    
    if missing_textures:
        logger.warning(f"Missing texture files: {', '.join(missing_textures)}")
        print("You can generate textures by running create_textures.py")
        print("Or download them from NASA or other sources")
        return False
    
    logger.info("Texture files verified")
    return True

def create_default_shaders(missing_shaders):
    """Create default GLSL shader files"""
    # Basic planet vertex shader
    planet_vertex = """#version 330 core
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
}"""

    # Basic planet fragment shader
    planet_fragment = """#version 330 core
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
}"""

    # Ring vertex shader
    ring_vertex = """#version 330 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec2 texCoord;

out vec2 fragTexCoord;
out vec3 fragPosition;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main() {
    fragPosition = vec3(model * vec4(position, 1.0));
    fragTexCoord = texCoord;
    gl_Position = projection * view * model * vec4(position, 1.0);
}"""

    # Ring fragment shader
    ring_fragment = """#version 330 core
in vec2 fragTexCoord;
in vec3 fragPosition;

out vec4 fragColor;

uniform sampler2D texSampler;
uniform vec3 ringColor;
uniform vec3 lightPos;

void main() {
    // Calculate distance from center
    vec2 center = vec2(0.5, 0.5);
    float dist = distance(fragTexCoord, center) * 2.0;
    
    // Ring pattern
    float alpha = 0.7 * smoothstep(0.8, 0.85, dist) * smoothstep(1.0, 0.95, dist);
    
    // Use texture if available
    vec4 texColor = texture(texSampler, fragTexCoord);
    if (texColor.a > 0.0) {
        fragColor = texColor;
    } else {
        fragColor = vec4(ringColor, alpha);
    }
}"""

    # Trail vertex shader
    trail_vertex = """#version 330 core
layout(location = 0) in vec3 position;

uniform mat4 view;
uniform mat4 projection;

void main() {
    gl_Position = projection * view * vec4(position, 1.0);
}"""

    # Trail fragment shader
    trail_fragment = """#version 330 core
out vec4 fragColor;

uniform vec3 trailColor;

void main() {
    fragColor = vec4(trailColor, 0.5);
}"""

    # Map of filenames to shader code
    shader_code = {
        "planet_vertex.glsl": planet_vertex,
        "planet_fragment.glsl": planet_fragment,
        "ring_vertex.glsl": ring_vertex,
        "ring_fragment.glsl": ring_fragment,
        "trail_vertex.glsl": trail_vertex,
        "trail_fragment.glsl": trail_fragment
    }
    
    # Create each missing shader
    for shader in missing_shaders:
        if shader in shader_code:
            try:
                with open(os.path.join("shaders", shader), 'w') as f:
                    f.write(shader_code[shader])
                logger.info(f"Created shader: {shader}")
            except Exception as e:
                logger.error(f"Could not create shader {shader}: {e}")
        else:
            logger.error(f"No default code for shader: {shader}")

def main():
    """Main function to fix simulation issues"""
    print("=== Planetary Simulation Fix Utility ===")
    print("This script will check and fix common issues with the simulation")
    
    issues_found = False
    
    # Check Python version
    if not check_python_version():
        issues_found = True
        print("Please upgrade Python to continue")
        return 1
    
    # Check required modules
    if not check_required_modules():
        issues_found = True
        print("Please install missing modules to continue")
        return 1
    
    # Check directory structure
    if not check_directory_structure():
        issues_found = True
    
    # Check shader files
    if not check_shader_files():
        issues_found = True
    
    # Check texture files
    if not check_texture_files():
        issues_found = True
    
    if issues_found:
        print("\nIssues were found that may affect the simulation")
        print("Please address them before running the simulation")
    else:
        print("\nNo issues were found! The simulation should work correctly")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())