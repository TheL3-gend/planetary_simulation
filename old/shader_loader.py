#!/usr/bin/env python3
# shader_loader.py - Utilities for loading and compiling GLSL shaders

import os
from OpenGL.GL import *
from OpenGL.GL import shaders
from constants import SHADER_DIR

def load_shader(vertex_file, fragment_file):
    """Load and compile a shader program from vertex and fragment shader files"""
    try:
        # Full paths to shader files
        vertex_path = os.path.join(SHADER_DIR, vertex_file)
        fragment_path = os.path.join(SHADER_DIR, fragment_file)
        
        
        # Check if files exist
        if not os.path.exists(vertex_path):
            raise FileNotFoundError(f"Vertex shader file not found: {vertex_path}")
        if not os.path.exists(fragment_path):
            raise FileNotFoundError(f"Fragment shader file not found: {fragment_path}")
        
        # Read shader source code
        with open(vertex_path, 'r') as f:
            vertex_src = f.read()
        
        with open(fragment_path, 'r') as f:
            fragment_src = f.read()
        
        # Compile shaders
        vertex_shader = shaders.compileShader(vertex_src, GL_VERTEX_SHADER)
        fragment_shader = shaders.compileShader(fragment_src, GL_FRAGMENT_SHADER)
        
        # Link shaders into a program
        shader_program = shaders.compileProgram(vertex_shader, fragment_shader)
        
        return shader_program
        
    except Exception as e:
        print(f"Error loading shader {vertex_file}/{fragment_file}: {e}")
        return 0

def load_shader_from_strings(vertex_src, fragment_src):
    """Load and compile a shader program from vertex and fragment shader strings"""
    try:
        # Compile shaders
        vertex_shader = shaders.compileShader(vertex_src, GL_VERTEX_SHADER)
        fragment_shader = shaders.compileShader(fragment_src, GL_FRAGMENT_SHADER)
        
        # Link shaders into a program
        shader_program = shaders.compileProgram(vertex_shader, fragment_shader)
        
        return shader_program
        
    except Exception as e:
        print(f"Error compiling shader from strings: {e}")
        return 0

def set_uniform_matrix4fv(shader_program, name, value):
    """Set a uniform mat4 value in a shader program"""
    try:
        location = glGetUniformLocation(shader_program, name)
        glUniformMatrix4fv(location, 1, GL_FALSE, value)
    except Exception as e:
        print(f"Error setting uniform {name}: {e}")

def set_uniform_3f(shader_program, name, x, y, z):
    """Set a uniform vec3 value in a shader program"""
    try:
        location = glGetUniformLocation(shader_program, name)
        glUniform3f(location, x, y, z)
    except Exception as e:
        print(f"Error setting uniform {name}: {e}")

def set_uniform_1f(shader_program, name, value):
    """Set a uniform float value in a shader program"""
    try:
        location = glGetUniformLocation(shader_program, name)
        glUniform1f(location, value)
    except Exception as e:
        print(f"Error setting uniform {name}: {e}")

def set_uniform_1i(shader_program, name, value):
    """Set a uniform int value in a shader program"""
    try:
        location = glGetUniformLocation(shader_program, name)
        glUniform1i(location, value)
    except Exception as e:
        print(f"Error setting uniform {name}: {e}")

def set_uniform_1b(shader_program, name, value):
    """Set a uniform bool value in a shader program"""
    try:
        location = glGetUniformLocation(shader_program, name)
        glUniform1i(location, 1 if value else 0)
    except Exception as e:
        print(f"Error setting uniform {name}: {e}")