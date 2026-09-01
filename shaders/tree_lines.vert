#version 430 core

// Draws box outlines straight out of an SSBO: no vertex buffer, no index
// buffer. Eight vertices per box trace its four edges.

layout(std430, binding = 0) readonly buffer BoxBuffer { vec4 boxes[]; };
layout(std430, binding = 1) readonly buffer DrawCmd   { uint cmd[]; };

uniform mat4 projection;
uniform int capacity;

out float vShade;

void main() {
    uint node = uint(gl_VertexID) >> 3u;
    uint v = uint(gl_VertexID) & 7u;

    if (node >= uint(capacity)) {
        // The collector counts every visible box but only stores what fits, so
        // the tail of an overflowing draw is pushed outside clip space.
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        vShade = 0.0;
        return;
    }

    vec4 bb = boxes[node];
    vec2 lo = bb.xy;
    vec2 hi = bb.zw;

    // 0,1,1,2,2,3,3,0 -- consecutive pairs walk the perimeter.
    uint corner = ((v + 1u) >> 1u) & 3u;
    vec2 p = vec2((corner == 1u || corner == 2u) ? hi.x : lo.x,
                  (corner >= 2u) ? hi.y : lo.y);

    gl_Position = projection * vec4(p, 0.0, 1.0);

    // Smaller cells are deeper; fading them keeps the coarse structure legible
    // instead of drowning it in the finest level.
    float rootE = uintBitsToFloat(cmd[4]);
    float rel = max(hi.x - lo.x, hi.y - lo.y) / max(rootE, 1e-6);
    vShade = clamp(0.35 + 0.65 * pow(rel, 0.35), 0.0, 1.0);
}
