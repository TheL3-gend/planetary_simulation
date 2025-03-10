#version 330 core

// Input vertex data
layout(location = 0) in vec3 position;
layout(location = 1) in vec2 texCoord;

// Output data to fragment shader
out vec2 fragTexCoord;
out vec3 fragPosition;

// Uniforms
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main() {
    // Pass through texture coordinates
    fragTexCoord = texCoord;
    
    // Calculate fragment position in world space
    fragPosition = vec3(model * vec4(position, 1.0));
    
    // Calculate final vertex position
    gl_Position = projection * view * model * vec4(position, 1.0);
}