#!/usr/bin/env python3
# debug_utils.py - Debugging utilities for OpenGL and general error checking
# This file should be placed in the same directory as main.py

import logging
from OpenGL.GL import *

# Configure logging
logger = logging.getLogger("GravitySim.Debug")

# Debug flag
_debug_mode = False

def initialize_debug():
    """Initialize debug system"""
    global _debug_mode
    _debug_mode = True
    logger.info("Debug system initialized")

def debug_print(message, level="INFO"):
    """Print debug message based on level"""
    if not _debug_mode and level == "DEBUG":
        return
        
    if level == "ERROR":
        logger.error(message)
    elif level == "WARNING":
        logger.warning(message)
    else:
        logger.info(message)

def mark_gl_initialized():
    """Mark that OpenGL has been initialized"""
    debug_print("OpenGL initialization completed", "INFO")

def print_gl_info():
    """Print OpenGL version and capabilities"""
    try:
        version = glGetString(GL_VERSION).decode('utf-8')
        vendor = glGetString(GL_VENDOR).decode('utf-8')
        renderer = glGetString(GL_RENDERER).decode('utf-8')
        shading_version = glGetString(GL_SHADING_LANGUAGE_VERSION).decode('utf-8')
        
        debug_print(f"OpenGL Version: {version}", "INFO")
        debug_print(f"OpenGL Vendor: {vendor}", "INFO")
        debug_print(f"OpenGL Renderer: {renderer}", "INFO")
        debug_print(f"GLSL Version: {shading_version}", "INFO")
        
        # Check for extensions
        extensions = []
        for i in range(glGetIntegerv(GL_NUM_EXTENSIONS)):
            ext = glGetStringi(GL_EXTENSIONS, i).decode('utf-8')
            extensions.append(ext)
            
        if _debug_mode:
            debug_print(f"OpenGL Extensions: {', '.join(extensions)}", "DEBUG")
    except Exception as e:
        debug_print(f"Error getting OpenGL info: {e}", "ERROR")

def check_gl_errors(where=""):
    """Check for OpenGL errors and log them"""
    error = glGetError()
    if error != GL_NO_ERROR:
        error_str = {
            GL_INVALID_ENUM: "GL_INVALID_ENUM",
            GL_INVALID_VALUE: "GL_INVALID_VALUE",
            GL_INVALID_OPERATION: "GL_INVALID_OPERATION",
            GL_STACK_OVERFLOW: "GL_STACK_OVERFLOW",
            GL_STACK_UNDERFLOW: "GL_STACK_UNDERFLOW",
            GL_OUT_OF_MEMORY: "GL_OUT_OF_MEMORY",
            GL_INVALID_FRAMEBUFFER_OPERATION: "GL_INVALID_FRAMEBUFFER_OPERATION",
            GL_CONTEXT_LOST: "GL_CONTEXT_LOST"
        }.get(error, f"Unknown error code: {error}")
        
        debug_print(f"OpenGL error at {where}: {error_str}", "ERROR")
        return True
    return False

def check_shader_compilation(shader):
    """Check if a shader compiled successfully"""
    status = glGetShaderiv(shader, GL_COMPILE_STATUS)
    if status != GL_TRUE:
        # Get shader info log
        log = glGetShaderInfoLog(shader).decode('utf-8')
        debug_print(f"Shader compilation failed: {log}", "ERROR")
        return False
    return True

def check_program_linking(program):
    """Check if a shader program linked successfully"""
    status = glGetProgramiv(program, GL_LINK_STATUS)
    if status != GL_TRUE:
        # Get program info log
        log = glGetProgramInfoLog(program).decode('utf-8')
        debug_print(f"Program linking failed: {log}", "ERROR")
        return False
    return True

def validate_program(program):
    """Validate a shader program"""
    glValidateProgram(program)
    status = glGetProgramiv(program, GL_VALIDATE_STATUS)
    if status != GL_TRUE:
        # Get validation info log
        log = glGetProgramInfoLog(program).decode('utf-8')
        debug_print(f"Program validation warning: {log}", "WARNING")
        return False
    return True

def check_memory_usage():
    """Check system memory usage (platform-dependent)"""
    try:
        import psutil
        mem = psutil.virtual_memory()
        debug_print(f"Memory usage: {mem.percent}% (Available: {mem.available / (1024*1024):.1f} MB)", "INFO")
    except ImportError:
        debug_print("psutil module not available for memory checking", "DEBUG")
    except Exception as e:
        debug_print(f"Error checking memory: {e}", "ERROR")

def check_framerate(dt):
    """Monitor framerate"""
    if dt > 0:
        fps = 1.0 / dt
        if fps < 30:
            debug_print(f"Low framerate: {fps:.1f} FPS", "WARNING")