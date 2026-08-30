#include "galaxy_preset.h"
#include <algorithm>
#include <cmath>
#include <random>
#include <glm/gtc/constants.hpp>

GalaxyPreset::GalaxyPreset(const GalaxyParams& params) : params(params) {}

GalaxyPreset::GalaxyPreset(int num_particles, float radius) {
    params.count = num_particles;
    params.outer_radius = radius;
}

int GalaxyPreset::getParticleCount() const {
    return params.count + (params.central_mass > 0.0f ? 1 : 0);
}

void GalaxyPreset::apply(std::vector<Particle>& particles) {
    particles.clear();
    particles.reserve(getParticleCount());

    const float inner = std::max(params.inner_radius, 1e-3f);
    const float outer = std::max(params.outer_radius, inner * 1.001f);
    const float disc_mass = params.particle_mass * float(params.count);

    std::mt19937 gen(params.seed);
    std::uniform_real_distribution<float> angle_dist(0.0f, 2.0f * glm::pi<float>());
    std::uniform_real_distribution<float> unit_dist(0.0f, 1.0f);
    std::normal_distribution<float> jitter(0.0f, 1.0f);

    if (params.central_mass > 0.0f)
        particles.emplace_back(glm::vec2(0.0f), glm::vec2(0.0f), params.central_mass);

    // Inverse-CDF sampling of sigma(r) ~ r^-falloff. With q = 2 - falloff the
    // enclosed-count fraction is exactly the uniform deviate that produced r,
    // which is also what the enclosed disc mass below needs.
    const float q = 2.0f - params.density_falloff;
    const bool scale_free = std::abs(q) < 1e-4f;
    const float inner_q = scale_free ? 0.0f : std::pow(inner, q);
    const float outer_q = scale_free ? 0.0f : std::pow(outer, q);
    const float log_ratio = std::log(outer / inner);

    for (int i = 0; i < params.count; i++) {
        const float u = unit_dist(gen);
        const float r = scale_free
            ? inner * std::exp(u * log_ratio)
            : std::pow(inner_q + u * (outer_q - inner_q), 1.0f / q);

        const float angle = angle_dist(gen);
        const glm::vec2 pos(r * std::cos(angle), r * std::sin(angle));

        // Balance against everything interior: the central mass plus the disc
        // inside this radius. Using the central mass alone leaves the outer
        // disc badly under-supported once it carries comparable mass.
        const float enclosed = params.central_mass + disc_mass * u;
        const float v_circ = std::sqrt(params.G * enclosed / r);
        const float speed = v_circ * params.spin
                          * (1.0f + params.velocity_dispersion * jitter(gen));

        const glm::vec2 vel(-speed * std::sin(angle), speed * std::cos(angle));
        particles.emplace_back(pos, vel, params.particle_mass);
    }
}
