#!/usr/bin/env python3
# setup.py - Setup script for gravity simulation

import os
import sys
import subprocess
import platform

def create_directories():
    """Create required directories if they don't exist"""
    directories = ["textures", "shaders", "fonts"]
    
    for directory in directories:
        if not os.path.exists(directory):
            print(f"Creating {directory} directory...")
            os.makedirs(directory)

def check_requirements():
    """Check if required packages are installed"""
    required_packages = ["pygame", "PyOpenGL", "PyOpenGL_accelerate", "numpy"]
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package} is not installed")
    
    if missing_packages:
        print("\nSome required packages are missing. Install them with:")
        print(f"pip install {' '.join(missing_packages)}")
        
        install = input("Install missing packages now? (y/n): ")
        if install.lower() == 'y':
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
                print("Packages installed successfully!")
            except subprocess.CalledProcessError:
                print("Failed to install packages. Please install them manually.")
                return False
    
    return True

def check_opengl_version():
    """Check OpenGL version"""
    try:
        import pygame
        import OpenGL.GL as gl
        
        # Initialize pygame and create an OpenGL context
        pygame.init()
        pygame.display.set_mode((1, 1), pygame.OPENGL | pygame.DOUBLEBUF | pygame.HIDDEN)
        
        # Get OpenGL version
        version = gl.glGetString(gl.GL_VERSION).decode('utf-8')
        print(f"OpenGL Version: {version}")
        
        # Check if version is sufficient
        major, minor = map(int, version.split(' ')[0].split('.')[:2])
        if major < 3 or (major == 3 and minor < 3):
            print("Warning: OpenGL 3.3 or higher is recommended for best performance")
        
        pygame.quit()
        
    except Exception as e:
        print(f"Error checking OpenGL version: {e}")
        print("Warning: Unable to determine OpenGL version")

def copy_shaders():
    """Copy shader files to shaders directory"""
    shader_files = [
        "planet_vertex.glsl", "planet_fragment.glsl",
        "ring_vertex.glsl", "ring_fragment.glsl",
        "trail_vertex.glsl", "trail_fragment.glsl"
    ]
    
    if not os.path.exists("shaders"):
        os.makedirs("shaders")
    
    # Check if shader files exist in current directory
    missing_shaders = []
    for shader in shader_files:
        if not os.path.exists(os.path.join("shaders", shader)):
            missing_shaders.append(shader)
    
    if missing_shaders:
        print(f"Some shader files are missing: {', '.join(missing_shaders)}")
        print("Please ensure all shader files are in the 'shaders' directory.")
        
        # Copy from examples if available
        example_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "example_shaders")
        if os.path.exists(example_dir):
            print("Copying example shaders...")
            for shader in missing_shaders:
                example_path = os.path.join(example_dir, shader)
                if os.path.exists(example_path):
                    with open(example_path, 'r') as f_in, open(os.path.join("shaders", shader), 'w') as f_out:
                        f_out.write(f_in.read())
                    print(f"Copied {shader}")

def generate_textures():
    """Generate procedural textures for planets if they don't exist"""
    print("\nChecking for planet textures...")
    textures_path = "textures"
    
    if not os.path.exists(textures_path):
        os.makedirs(textures_path)
    
    # Check if textures exist
    texture_files = [
        "sun.jpg", "mercury.jpg", "venus.jpg", "earth.jpg", "mars.jpg",
        "jupiter.jpg", "saturn.jpg", "uranus.jpg", "neptune.jpg", "moon.jpg"
    ]
    
    missing_textures = []
    for texture in texture_files:
        if not os.path.exists(os.path.join(textures_path, texture)):
            missing_textures.append(texture)
    
    if missing_textures:
        print(f"Missing texture files: {', '.join(missing_textures)}")
        print("Would you like to generate procedural textures? (This may take a moment)")
        
        generate = input("Generate textures now? (y/n): ")
        if generate.lower() == 'y':
            try:
                print("Generating textures...")
                import create_textures
                create_textures.generate_all_textures()
                print("Textures generated successfully!")
            except Exception as e:
                print(f"Error generating textures: {e}")
                print("You can download planet textures from NASA's website or other sources.")
    else:
        print("All required textures found!")

def main():
    """Main setup function"""
    print("=== Gravity Simulation Setup ===")
    
    # Create required directories
    create_directories()
    
    # Check if required packages are installed
    if not check_requirements():
        print("Setup incomplete due to missing requirements.")
        return
    
    # Check OpenGL version
    check_opengl_version()
    
    # Copy shader files if needed
    copy_shaders()
    
    # Generate textures if needed
    generate_textures()
    
    print("\nSetup complete! Run the simulation with: python main.py")

if __name__ == "__main__":
    main()