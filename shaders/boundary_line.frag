#version 430 core

out vec4 FragColor;

uniform vec3 lineColor;
uniform float lineAlpha;

void main() {
    FragColor = vec4(lineColor, lineAlpha);
}
