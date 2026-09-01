#version 430 core

in float vShade;
out vec4 FragColor;

uniform vec3 lineColor;
uniform float lineAlpha;

void main() {
    FragColor = vec4(lineColor * vShade, lineAlpha * vShade);
}
