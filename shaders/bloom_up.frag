#version 430 core

// 3x3 tent upsample, written with additive blending so each mip accumulates
// into the one above it. Progressive upsampling like this gives a wide, smooth
// glow far more cheaply than one large-radius blur.

in vec2 uv;
out vec4 FragColor;

uniform sampler2D source;
uniform vec2 texelSize;
uniform float radius;

void main() {
    vec2 t = texelSize * radius;

    vec3 sum = texture(source, uv + vec2(-t.x,  t.y)).rgb;
    sum += texture(source, uv + vec2(0.0,  t.y)).rgb * 2.0;
    sum += texture(source, uv + vec2( t.x,  t.y)).rgb;
    sum += texture(source, uv + vec2(-t.x, 0.0)).rgb * 2.0;
    sum += texture(source, uv).rgb * 4.0;
    sum += texture(source, uv + vec2( t.x, 0.0)).rgb * 2.0;
    sum += texture(source, uv + vec2(-t.x, -t.y)).rgb;
    sum += texture(source, uv + vec2(0.0, -t.y)).rgb * 2.0;
    sum += texture(source, uv + vec2( t.x, -t.y)).rgb;

    FragColor = vec4(sum * (1.0 / 16.0), 1.0);
}
