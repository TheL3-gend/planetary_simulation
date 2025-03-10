#version 330 core

// Input data from vertex shader
in vec2 fragTexCoord;
in vec3 fragPosition;

// Output color
out vec4 fragColor;

// Uniforms
uniform sampler2D texSampler;
uniform vec3 ringColor;
uniform vec3 lightPos;
uniform vec3 viewPos;

// Distance calculations
uniform float innerRadius;
uniform float outerRadius;

void main() {
    // Calculate distance from center (0.5, 0.5)
    vec2 center = vec2(0.5, 0.5);
    float dist = distance(fragTexCoord, center) * 2.0;
    
    // Calculate ring density based on distance from center
    float normalized_dist = (dist - innerRadius) / (outerRadius - innerRadius);
    
    // Ring pattern with multiple bands
    float bands = sin(normalized_dist * 30.0) * 0.5 + 0.5;
    
    // Procedural ring transparency pattern
    float alpha;
    
    // Check if texture provided
    vec4 texColor = texture(texSampler, fragTexCoord);
    
    if (texColor.a > 0.0) {
        // Use texture if available
        alpha = texColor.a;
    } else {
        // Create procedural rings with gaps
        alpha = 0.7;
        
        // Create Cassini Division (main gap in Saturn's rings)
        if (normalized_dist > 0.4 && normalized_dist < 0.5) {
            alpha *= 0.3;
        }
        
        // Create smaller gaps
        if (normalized_dist > 0.7 && normalized_dist < 0.72) {
            alpha *= 0.5;
        }
        
        if (normalized_dist > 0.85 && normalized_dist < 0.86) {
            alpha *= 0.7;
        }
        
        // Fade at edges
        alpha *= smoothstep(0.0, 0.1, normalized_dist) * smoothstep(1.0, 0.9, normalized_dist);
        
        // Modulate with bands
        alpha *= 0.8 + bands * 0.2;
    }
    
    // Lighting calculations
    vec3 ringNormal = vec3(0.0, 1.0, 0.0);  // Rings are flat on the y-plane
    
    // Light direction
    vec3 lightDir = normalize(lightPos - fragPosition);
    
    // View direction
    vec3 viewDir = normalize(viewPos - fragPosition);
    
    // Calculate lighting factor based on angle between light and ring normal
    float lightFactor = max(abs(dot(ringNormal, lightDir)), 0.2);
    
    // Calculate color based on lighting
    vec3 color;
    
    if (texColor.a > 0.0) {
        color = texColor.rgb * lightFactor;
    } else {
        color = ringColor * (0.5 + bands * 0.5) * lightFactor;
    }
    
    // Set final color with calculated alpha
    fragColor = vec4(color, alpha);
}