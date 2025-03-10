#!/usr/bin/env python3
# setup.py - Setup script for gravity simulation with improved error handling

import os
import sys
import subprocess
import platform
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("Setup")

def create_directories():
    """Create required directories if they don't exist"""
    directories = ["textures", "shaders", "fonts"]
    
    for directory in directories:
        if not os.path.exists(directory):
            try:
                logger.info(f"Creating {directory} directory...")
                os.makedirs(directory)
            except OSError as e:
                logger.error(f"Failed to create {directory} directory: {e}")
                return False
    
    return True

def check_requirements():
    """Check if required packages are installed"""
    required_packages = ["pygame", "PyOpenGL", "PyOpenGL_accelerate", "numpy"]
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✓ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            logger.warning(f"✗ {package} is not installed")
    
    if missing_packages:
        logger.warning(f"Missing packages: {', '.join(missing_packages)}")
        
        print("\nSome required packages are missing. Install them with:")
        print(f"pip install {' '.join(missing_packages)}")
        
        try:
            install = input("Install missing packages now? (y/n): ").strip().lower()
            if install == 'y':
                try:
                    logger.info("Installing missing packages...")
                    subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
                    logger.info("Packages installed successfully!")
                    
                    # Verify installation
                    still_missing = []
                    for package in missing_packages:
                        try:
                            __import__(package)
                        except ImportError:
                            still_missing.append(package)
                    
                    if still_missing:
                        logger.error(f"Some packages are still missing: {', '.join(still_missing)}")
                        return False
                    
                    return True
                except subprocess.CalledProcessError as e:
                    logger.error(f"Failed to install packages: {e}")
                    print("Please install them manually.")
                    return False
            else:
                logger.warning("Packages must be installed to run the simulation")
                return False
        except KeyboardInterrupt:
            logger.warning("Setup interrupted")
            return False
    
    return True

def check_opengl_version():
    """Check if OpenGL version is sufficient"""
    try:
        import pygame
        import OpenGL.GL as gl
        
        # Initialize pygame and create a temporary OpenGL context
        if not pygame.get_init():
            pygame.init()
            
        display_flags = pygame.OPENGL | pygame.DOUBLEBUF
        if hasattr(pygame, 'HIDDEN'):
            display_flags |= pygame.HIDDEN
            
        pygame.display.set_mode((1, 1), display_flags)
        
        # Get OpenGL version
        try:
            version_string = gl.glGetString(gl.GL_VERSION).decode('utf-8')
            renderer_string = gl.glGetString(gl.GL_RENDERER).decode('utf-8')
            
            logger.info(f"OpenGL Version: {version_string}")
            logger.info(f"OpenGL Renderer: {renderer_string}")
            
            # Extract version numbers
            import re
            version_match = re.search(r'(\d+)\.(\d+)', version_string)
            if version_match:
                major, minor = map(int, version_match.groups())
                
                if major < 2 or (major == 2 and minor < 1):
                    logger.error(f"OpenGL {major}.{minor} is too old. Version 2.1 or higher required.")
                    return False
                    
                if major < 3 or (major == 3 and minor < 3):
                    logger.warning(f"OpenGL {major}.{minor} is supported, but 3.3+ is recommended for best performance")
            else:
                logger.warning(f"Could not parse OpenGL version: {version_string}")
        except Exception as e:
            logger.error(f"Error getting OpenGL information: {e}")
            
        # Clean up
        pygame.quit()
        return True
        
    except ImportError as e:
        logger.error(f"Could not import required packages to check OpenGL: {e}")
        return False
    except Exception as e:
        logger.error(f"Error checking OpenGL version: {e}")
        return False

def copy_shaders():
    """Copy shader files to the shaders directory"""
    shader_files = [
        "planet_vertex.glsl", "planet_fragment.glsl",
        "ring_vertex.glsl", "ring_fragment.glsl",
        "trail_vertex.glsl", "trail_fragment.glsl"
    ]
    
    # Create shader directory if needed
    if not os.path.exists("shaders"):
        try:
            os.makedirs("shaders")
        except OSError as e:
            logger.error(f"Failed to create shader directory: {e}")
            return False
    
    # Check if shader files exist
    missing_shaders = []
    for shader in shader_files:
        if not os.path.exists(os.path.join("shaders", shader)):
            missing_shaders.append(shader)
    
    if missing_shaders:
        logger.warning(f"Missing shader files: {', '.join(missing_shaders)}")
        
        # Look for example shaders
        example_dirs = [
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "example_shaders"),
            "example_shaders",
            "."
        ]
        
        files_copied = 0
        for shader in missing_shaders:
            shader_found = False
            
            # Try each possible location
            for example_dir in example_dirs:
                example_path = os.path.join(example_dir, shader)
                if os.path.exists(example_path):
                    try:
                        with open(example_path, 'r') as f_in, open(os.path.join("shaders", shader), 'w') as f_out:
                            f_out.write(f_in.read())
                        logger.info(f"Copied {shader}")
                        files_copied += 1
                        shader_found = True
                        break
                    except Exception as e:
                        logger.error(f"Error copying {shader}: {e}")
            
            if not shader_found:
                logger.error(f"Could not find {shader} to copy")
                    
        if files_copied < len(missing_shaders):
            # Create default shaders for any that couldn't be found
            logger.warning(f"Creating default shaders for missing files")
            return create_default_shaders(missing_shaders)
    
    return True

def create_default_shaders(missing_shaders):
    """Create default shaders for any missing ones"""
    # Default shader code
    default_shaders = {
        "planet_vertex.glsl": """#version 330 core
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
}""",
        "planet_fragment.glsl": """#version 330 core
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
}""",
        "ring_vertex.glsl": """#version 330 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec2 texCoord;

out vec2 fragTexCoord;
out vec3 fragPosition;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main() {
    fragTexCoord = texCoord;
    fragPosition = vec3(model * vec4(position, 1.0));
    gl_Position = projection * view * model * vec4(position, 1.0);
}""",
        "ring_fragment.glsl": """#version 330 core
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
}""",
        "trail_vertex.glsl": """#version 330 core
layout(location = 0) in vec3 position;

uniform mat4 view;
uniform mat4 projection;

void main() {
    gl_Position = projection * view * vec4(position, 1.0);
}""",
        "trail_fragment.glsl": """#version 330 core
out vec4 fragColor;

uniform vec3 trailColor;

void main() {
    fragColor = vec4(trailColor, 0.5);
}"""
    }
    
    try:
        # Create each missing shader
        for shader in missing_shaders:
            if shader in default_shaders:
                shader_path = os.path.join("shaders", shader)
                with open(shader_path, 'w') as f:
                    f.write(default_shaders[shader])
                logger.info(f"Created default {shader}")
            else:
                logger.error(f"No default code available for {shader}")
        
        return True
    except Exception as e:
        logger.error(f"Error creating default shaders: {e}")
        return False

def generate_textures():
    """Generate procedural textures for planets if they don't exist"""
    logger.info("Checking for planet textures...")
    
    # Essential textures to check
    texture_files = [
        "sun.jpg", "mercury.jpg", "venus.jpg", "earth.jpg", "mars.jpg",
        "jupiter.jpg", "saturn.jpg", "uranus.jpg", "neptune.jpg", "moon.jpg"
    ]
    
    # Count missing textures
    missing_textures = []
    for texture in texture_files:
        if not os.path.exists(os.path.join("textures", texture)):
            missing_textures.append(texture)
    
    if missing_textures:
        logger.warning(f"Missing texture files: {', '.join(missing_textures)}")
        
        print("\nWould you like to generate procedural textures? (This may take a moment)")
        generate = input("Generate textures now? (y/n): ").strip().lower()
        
        if generate == 'y':
            try:
                logger.info("Initializing pygame for texture generation...")
                import pygame
                pygame.init()
                pygame.display.set_mode((1, 1), pygame.NOFRAME)
                
                logger.info("Generating textures...")
                import create_textures
                create_textures.generate_all_textures()
                
                # Verify textures were created
                still_missing = []
                for texture in missing_textures:
                    if not os.path.exists(os.path.join("textures", texture)):
                        still_missing.append(texture)
                
                if still_missing:
                    logger.warning(f"Some textures are still missing: {', '.join(still_missing)}")
                else:
                    logger.info("All textures generated successfully!")
                
                pygame.quit()
            except Exception as e:
                logger.error(f"Error generating textures: {e}")
                logger.warning("You can download planet textures from NASA's website or other sources")
                return False
    else:
        logger.info("All required textures found!")
    
    return True

def main():
    """Main setup function"""
    print("=== Gravity Simulation Setup ===")
    
    try:
        # Create required directories
        if not create_directories():
            logger.error("Failed to create required directories")
            return 1
        
        # Check required packages
        if not check_requirements():
            logger.error("Some required packages are missing")
            return 1
        
        # Check OpenGL version
        if not check_opengl_version():
            logger.warning("OpenGL version may not be sufficient")
            # Continue anyway, it might still work
        
        # Copy shader files if needed
        if not copy_shaders():
            logger.error("Failed to set up shader files")
            return 1
        
        # Generate textures if needed
        if not generate_textures():
            logger.warning("Texture generation incomplete")
            # Continue anyway, the program will use fallback textures
        
        logger.info("Setup complete! Run the simulation with: python main.py")
        return 0
    
    except KeyboardInterrupt:
        logger.warning("Setup interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error during setup: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())