#version 330 core

// Input data from vertex shader
in vec3 fragNormal;
in vec3 fragPosition;
in vec2 fragTexCoord;

// Output color
out vec4 fragColor;

// Uniforms
uniform vec3 lightPos;
uniform vec3 viewPos;
uniform vec3 baseColor;
uniform sampler2D texSampler;
uniform bool useTexture;

// Lighting constants
const float ambientStrength = 0.3;
const float specularStrength = 0.5;
const float shininess = 32.0;

void main() {
    // Ambient lighting
    vec3 ambient = ambientStrength * vec3(1.0, 1.0, 1.0);
    
    // Diffuse lighting
    vec3 norm = normalize(fragNormal);
    vec3 lightDir = normalize(lightPos - fragPosition);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * vec3(1.0, 1.0, 1.0);
    
    // Specular lighting
    vec3 viewDir = normalize(viewPos - fragPosition);
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), shininess);
    vec3 specular = specularStrength * spec * vec3(1.0, 1.0, 1.0);
    
    // Get base color (from texture or uniform)
    vec3 color;
    if (useTexture) {
        color = texture(texSampler, fragTexCoord).rgb;
    } else {
        color = baseColor;
    }
    
    // Combine lighting components
    vec3 result = (ambient + diffuse + specular) * color;
    
    // Final color with full opacity
    fragColor = vec4(result, 1.0);
}