#version 430 core

// 13-tap downsample (Jimenez 2014); stable under repeated halving.

in vec2 uv;
out vec4 FragColor;

uniform sampler2D source;
uniform vec2 texelSize;
uniform int firstPass;      // apply the bright-pass only when reading the scene
uniform float threshold;
uniform float knee;

vec3 prefilter(vec3 c) {
    if (firstPass == 0) return c;
    float br = max(c.r, max(c.g, c.b));
    float soft = clamp(br - threshold + knee, 0.0, 2.0 * knee);
    soft = soft * soft / (4.0 * knee + 1e-4);
    float contrib = max(soft, br - threshold) / max(br, 1e-4);
    return c * contrib;
}

void main() {
    vec2 t = texelSize;

    vec3 a = texture(source, uv + vec2(-2.0 * t.x,  2.0 * t.y)).rgb;
    vec3 b = texture(source, uv + vec2( 0.0,        2.0 * t.y)).rgb;
    vec3 c = texture(source, uv + vec2( 2.0 * t.x,  2.0 * t.y)).rgb;
    vec3 d = texture(source, uv + vec2(-2.0 * t.x,  0.0)).rgb;
    vec3 e = texture(source, uv).rgb;
    vec3 f = texture(source, uv + vec2( 2.0 * t.x,  0.0)).rgb;
    vec3 g = texture(source, uv + vec2(-2.0 * t.x, -2.0 * t.y)).rgb;
    vec3 h = texture(source, uv + vec2( 0.0,       -2.0 * t.y)).rgb;
    vec3 i = texture(source, uv + vec2( 2.0 * t.x, -2.0 * t.y)).rgb;

    vec3 j = texture(source, uv + vec2(-t.x,  t.y)).rgb;
    vec3 k = texture(source, uv + vec2( t.x,  t.y)).rgb;
    vec3 l = texture(source, uv + vec2(-t.x, -t.y)).rgb;
    vec3 m = texture(source, uv + vec2( t.x, -t.y)).rgb;

    vec3 sum = e * 0.125;
    sum += (a + c + g + i) * 0.03125;
    sum += (b + d + f + h) * 0.0625;
    sum += (j + k + l + m) * 0.125;

    FragColor = vec4(prefilter(sum), 1.0);
}
