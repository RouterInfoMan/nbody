#version 430 core

// Combines the HDR scene with the bloom chain, applies exposure and a filmic
// tone curve, and writes sRGB.

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

// Applying a tone curve per channel compresses the largest channel hardest,
// which drags every bright pixel toward white -- and with additive particle
// blending almost the whole image is bright, so the scene loses most of its
// colour exactly where there is most of it. Tone mapping luminance alone and
// rescaling chroma keeps hue and saturation intact but can push channels out
// of gamut, so the two are blended.
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

    // Saturation is applied in linear HDR, before the tone curve, so the
    // curve sees the intended colour rather than correcting a clipped one.
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
