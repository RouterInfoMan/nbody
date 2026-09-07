#version 430 core

// Soft star sprite: wide halo plus tight core, drawn additively.

in vec3 vertexColor;
out vec4 FragColor;

void main() {
    vec2 coord = (gl_PointCoord - vec2(0.5)) * 2.0;
    float d2 = dot(coord, coord);
    if (d2 > 1.0) discard;

    float halo = exp(-d2 * 4.5);
    float core = exp(-d2 * 40.0);
    float falloff = halo + 0.75 * core;

    FragColor = vec4(vertexColor * falloff, 1.0);
}
