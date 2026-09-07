#include "collapse_preset.h"
#include <algorithm>
#include <cmath>
#include <random>
#include <glm/gtc/constants.hpp>

CollapsePreset::CollapsePreset(const CollapseParams& params) : params(params) {}

CollapsePreset::CollapsePreset(int num_particles, float radius) {
    params.count = num_particles;
    params.radius = radius;
}

int CollapsePreset::getParticleCount() const {
    return params.count + (params.central_mass > 0.0f ? 1 : 0);
}

void CollapsePreset::apply(std::vector<Particle>& particles) {
    particles.clear();
    particles.reserve(getParticleCount());

    const float radius = std::max(params.radius, 1e-3f);
    const float total_mass = params.particle_mass * float(params.count);

    std::mt19937 gen(params.seed);
    std::uniform_real_distribution<float> angle_dist(0.0f, 2.0f * glm::pi<float>());
    std::uniform_real_distribution<float> unit_dist(0.0f, 1.0f);
    std::normal_distribution<float> jitter(0.0f, 1.0f);

    if (params.central_mass > 0.0f)
        particles.emplace_back(glm::vec2(0.0f), glm::vec2(0.0f), params.central_mass);

    const bool needs_speed = params.rotation != 0.0f || params.velocity_dispersion > 0.0f;

    for (int i = 0; i < params.count; i++) {
        // sqrt keeps surface density uniform across the disc.
        const float u = unit_dist(gen);
        const float r = radius * std::sqrt(u);
        const float angle = angle_dist(gen);
        const glm::vec2 pos(r * std::cos(angle), r * std::sin(angle));

        glm::vec2 vel(0.0f);
        if (needs_speed && r > 1e-6f) {
            const float enclosed = params.central_mass + total_mass * u;
            const float v_circ = std::sqrt(params.G * enclosed / r);

            vel = glm::vec2(-std::sin(angle), std::cos(angle)) * (v_circ * params.rotation);
            if (params.velocity_dispersion > 0.0f) {
                const float sigma = v_circ * params.velocity_dispersion;
                vel += glm::vec2(jitter(gen), jitter(gen)) * sigma;
            }
        }

        particles.emplace_back(pos, vel, params.particle_mass);
    }
}
