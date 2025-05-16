#version 330 core
in float lightIntensity;
out vec4 FragColor;

uniform vec4 objectColor;
uniform bool isGrid;
uniform bool GLOW;

void main() {
    if(isGrid) {
        FragColor = objectColor;
    } else if(GLOW) {
        // Be careful with extreme multipliers, can lead to clamping issues or unexpected behavior
        // depending on framebuffer format and post-processing (if any).
        // For a simple glow, a slightly brighter color and potentially a bloom shader pass is better.
        // This large multiplier is probably for a specific artistic effect.
        FragColor = vec4(objectColor.rgb * 2.0, objectColor.a); // Reduced multiplier
    } else {
        float fade = smoothstep(0.0, 1.0, lightIntensity); // Adjusted smoothstep range
        FragColor = vec4(objectColor.rgb * fade, objectColor.a);
    }
}