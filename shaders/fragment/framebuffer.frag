#version 460 core
out vec4 FragColor;

in vec2 TexCoords;

uniform sampler2D screenTexture;

void main() {
    float d = texture(screenTexture, TexCoords).r;
    float n = fract(sin(dot(TexCoords * 256.0, vec2(12.9898,78.233))) * 43758.5453);

    d += (n - 0.5) * 0.03;
    d = clamp(d, 0.0, 1.0);

    vec3 color;
if (d < 0.3)
    color = mix(vec3(0.1, 0.1, 0.1), vec3(0.2, 0.5, 0.8), smoothstep(0.0, 0.3, d));
else if (d < 0.7)
    color = mix(vec3(0.1, 0.2, 0.4), vec3(0.9, 0.6, 0.3), smoothstep(0.3, 0.7, d));
else
    color = mix(vec3(0.9, 0.9, 0.9), vec3(1.0, 1.0, 1.0), smoothstep(0.7, 1.0, d));

vec3 smokeColor = color;

    float alpha = clamp(d * 1.5, 0.0, 1.0);

    FragColor = vec4(smokeColor, alpha);
}