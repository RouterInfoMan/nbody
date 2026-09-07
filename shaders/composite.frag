#version 430 core

in vec2 uv;
out vec4 FragColor;

uniform sampler2D scene;
uniform sampler2D bloom;
uniform float exposure;
uniform float bloomStrength;
uniform float vignette;
uniform float saturation;
uniform float huePreserve;

float luminance(vec3 c) {
    return dot(c, vec3(0.2126, 0.7152, 0.0722));
}

// Narkowicz's ACES fit.
float acesCurve(float x) {
    const float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

vec3 acesPerChannel(vec3 x) {
    const float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

// Per-channel tone mapping desaturates; blend in a luminance-only curve.
vec3 toneMap(vec3 c) {
    vec3 perChannel = acesPerChannel(c);

    float l = luminance(c);
    vec3 luminanceOnly = c * (acesCurve(l) / max(l, 1e-5));

    return clamp(mix(perChannel, luminanceOnly, huePreserve), 0.0, 1.0);
}

void main() {
    vec3 color = texture(scene, uv).rgb;
    color += texture(bloom, uv).rgb * bloomStrength;

    color *= exposure;

    if (saturation != 1.0) {
        float l = luminance(color);
        color = max(mix(vec3(l), color, saturation), vec3(0.0));
    }

    color = toneMap(color);

    if (vignette > 0.0) {
        vec2 d = uv - 0.5;
        float v = 1.0 - vignette * dot(d, d) * 2.0;
        color *= clamp(v, 0.0, 1.0);
    }

    // Linear to sRGB.
    color = pow(color, vec3(1.0 / 2.2));
    FragColor = vec4(color, 1.0);
}
