import glfw
from OpenGL.GL import *

if not glfw.init():
    raise RuntimeError("glfw.init failed")

window = glfw.create_window(800, 600, "Test", None, None)
