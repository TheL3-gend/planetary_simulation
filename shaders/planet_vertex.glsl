#version 330 core

// Input vertex data
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 texCoord;

// Output data to fragment shader
out vec3 fragNormal;
out vec3 fragPosition;
out vec2 fragTexCoord;

// Uniforms
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main() {
    // Calculate fragment position in world space
    fragPosition = vec3(model * vec4(position, 1.0));
    
    // Calculate normal in world space (without non-uniform scaling)
    fragNormal = mat3(transpose(inverse(model))) * normal;
    
    // Pass through texture coordinates
    fragTexCoord = texCoord;
    
    // Calculate final vertex position
    gl_Position = projection * view * model * vec4(position, 1.0);
}