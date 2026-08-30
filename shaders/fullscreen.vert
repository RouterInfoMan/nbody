#version 430 core

// Single oversized triangle covering the viewport; needs no vertex buffer,
// just a bound (empty) VAO and glDrawArrays(GL_TRIANGLES, 0, 3).

out vec2 uv;

void main() {
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    uv = p;
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
