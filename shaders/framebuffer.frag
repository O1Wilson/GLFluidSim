#version 430 core
out vec4 FragColor;

in vec2 TexCoords;

uniform sampler2D screenTexture;

void main() {
    float d = texture(screenTexture, TexCoords).r;
    float n = fract(sin(dot(TexCoords * 256.0, vec2(12.9898,78.233))) * 43758.5453);

    d += (n - 0.5) * 0.03;
    d = clamp(d, 0.0, 1.0);

    vec3 smokeColor = mix(vec3(0.1, 0.1, 0.1), vec3(0.9, 0.9, 0.9), d);

    float alpha = clamp(d * 1.5, 0.0, 1.0);

    FragColor = vec4(smokeColor, alpha);
}