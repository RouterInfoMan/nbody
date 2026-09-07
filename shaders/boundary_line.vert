#version 430 core

// Draws the containment wall as a line loop from gl_VertexID.

uniform mat4 projection;
uniform int shape;        // 0 circle, 1 box
uniform int segments;
uniform float radius;

void main() {
    vec2 p;
    if (shape == 0) {
        float t = 6.28318530718 * float(gl_VertexID) / float(segments);
        p = vec2(cos(t), sin(t)) * radius;
    } else {
        // 0,1,2,3 -> the box corners in order.
        int c = gl_VertexID & 3;
        p = vec2((c == 1 || c == 2) ? radius : -radius,
                 (c >= 2) ? radius : -radius);
    }
    gl_Position = projection * vec4(p, 0.0, 1.0);
}
