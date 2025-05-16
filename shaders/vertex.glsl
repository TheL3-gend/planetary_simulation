#version 330 core
layout(location = 0) in vec3 aPos;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out float lightIntensity;

void main() {
    gl_Position = projection * view * model * vec4(aPos, 1.0);
    vec3 worldPos = (model * vec4(aPos, 1.0)).xyz;
    // For a sphere centered at origin, aPos is the normal if radius is 1
    // If radius is not 1, normal is normalize(aPos)
    vec3 normal = normalize(aPos); // Assuming aPos are vertices of a unit sphere scaled by model
    vec3 dirToCenter = normalize(-worldPos); // Light source at world origin
    lightIntensity = max(dot(normal, dirToCenter), 0.15); // Ambient light component
}