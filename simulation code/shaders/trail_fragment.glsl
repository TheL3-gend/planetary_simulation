#version 330 core

// Output color
out vec4 fragColor;

// Uniforms
uniform vec3 trailColor;
uniform float trailFade; // Value from 0.0 to 1.0 for position in trail

void main() {
    // Set color with fade effect
    float alpha = 0.5 * trailFade;
    fragColor = vec4(trailColor, alpha);
}