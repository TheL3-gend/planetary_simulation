#version 330 core

// Input vertex data
layout(location = 0) in vec3 position;

// Uniforms
uniform mat4 view;
uniform mat4 projection;

void main() {
    // Simple pass-through with view and projection transforms
    gl_Position = projection * view * vec4(position, 1.0);
}