#version 430 core

// Soft star sprite: a wide halo plus a tight core. Rendered additively into an
// HDR target, so overlapping sprites accumulate and dense regions blow past
// 1.0 for the bloom pass to pick up.

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
