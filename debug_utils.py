#!/usr/bin/env python3
# debug_utils.py - Debugging utilities for the gravity simulation

import os
import sys
import traceback
import time
from OpenGL.GL import *

# Global variables to track state
_gl_initialized = False
_debug_mode = True  # Set to True to enable debug output, False for production
_debug_count = 0
_debug_start_time = 0
_frame_count = 0
_last_frame_time = 0
_fps = 0

def initialize_debug():
    """Initialize debug systems"""
    global _debug_start_time, _last_frame_time
    _debug_start_time = time.time()
    _last_frame_time = _debug_start_time
    debug_print("Debug system initialized")

def debug_print(message, level="INFO"):
    """Print a debug message if debug mode is enabled"""
    global _debug_count
    if _debug_mode:
        _debug_count += 1
        elapsed = time.time() - _debug_start_time
        print(f"[{elapsed:.3f}s] [{level}] #{_debug_count}: {message}")

def mark_gl_initialized():
    """Mark OpenGL as initialized"""
    global _gl_initialized
    _gl_initialized = True
    debug_print("OpenGL marked as initialized")

def is_gl_initialized():
    """Check if OpenGL is initialized"""
    return _gl_initialized

def track_frame():
    """Track frame rate information"""
    global _frame_count, _last_frame_time, _fps
    current_time = time.time()
    _frame_count += 1
    if current_time - _last_frame_time >= 1.0:
        _fps = _frame_count / (current_time - _last_frame_time)
        _frame_count = 0
        _last_frame_time = current_time
        debug_print(f"FPS: {_fps:.1f}")
    return _fps

def check_gl_errors(where=""):
    """Check for OpenGL errors and print them"""
    if not _debug_mode:
        return False
    
    error = glGetError()
    if error != GL_NO_ERROR:
        error_str = {
            GL_INVALID_ENUM: "GL_INVALID_ENUM",
            GL_INVALID_VALUE: "GL_INVALID_VALUE",
            GL_INVALID_OPERATION: "GL_INVALID_OPERATION",
            GL_STACK_OVERFLOW: "GL_STACK_OVERFLOW",
            GL_STACK_UNDERFLOW: "GL_STACK_UNDERFLOW",
            GL_OUT_OF_MEMORY: "GL_OUT_OF_MEMORY"
        }.get(error, f"Unknown error code: {error}")
        
        debug_print(f"OpenGL error at {where}: {error_str}", "ERROR")
        return True
    return False

def print_gl_info():
    """Print OpenGL driver information"""
    try:
        version = glGetString(GL_VERSION).decode('utf-8')
        vendor = glGetString(GL_VENDOR).decode('utf-8')
        renderer = glGetString(GL_RENDERER).decode('utf-8')
        
        debug_print(f"OpenGL Version: {version}")
        debug_print(f"OpenGL Vendor: {vendor}")
        debug_print(f"OpenGL Renderer: {renderer}")
        
        # Check if shaders are supported
        glsl_version = glGetString(GL_SHADING_LANGUAGE_VERSION)
        if glsl_version:
            debug_print(f"GLSL Version: {glsl_version.decode('utf-8')}")
        else:
            debug_print("GLSL not supported!", "WARNING")
    except:
        debug_print("Could not obtain OpenGL information", "ERROR")
        traceback.print_exc()

def check_shader_compilation(shader):
    """Check if a shader compiled successfully"""
    if not _debug_mode:
        return True
    
    status = glGetShaderiv(shader, GL_COMPILE_STATUS)
    if status != GL_TRUE:
        log = glGetShaderInfoLog(shader)
        if log:
            debug_print(f"Shader compilation failed: {log.decode('utf-8')}", "ERROR")
        else:
            debug_print("Shader compilation failed with no log", "ERROR")
        return False
    return True

def check_program_linking(program):
    """Check if a shader program linked successfully"""
    if not _debug_mode:
        return True
        
    status = glGetProgramiv(program, GL_LINK_STATUS)
    if status != GL_TRUE:
        log = glGetProgramInfoLog(program)
        if log:
            debug_print(f"Program linking failed: {log.decode('utf-8')}", "ERROR")
        else:
            debug_print("Program linking failed with no log", "ERROR")
        return False
    return True

def validate_program(program):
    """Validate a shader program"""
    if not _debug_mode:
        return True
        
    glValidateProgram(program)
    status = glGetProgramiv(program, GL_VALIDATE_STATUS)
    if status != GL_TRUE:
        log = glGetProgramInfoLog(program)
        if log:
            debug_print(f"Program validation failed: {log.decode('utf-8')}", "ERROR")
        else:
            debug_print("Program validation failed with no log", "ERROR")
        return False
    return True