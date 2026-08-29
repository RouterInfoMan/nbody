#include "collapse_preset.h"
#include <cmath>
#include <random>
#include <glm/gtc/constants.hpp>

CollapsePreset::CollapsePreset(int num_particles, float radius)
    : num_particles(num_particles), radius(radius) {}

void CollapsePreset::apply(std::vector<Particle>& particles) {
    particles.clear();
    particles.reserve(num_particles);

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> angle_dist(0.0f, 2.0f * glm::pi<float>());
    std::uniform_real_distribution<float> radius_dist(0.0f, 1.0f);

    for (int i = 0; i < num_particles; i++) {
        float angle = angle_dist(gen);
        float r = radius * std::sqrt(radius_dist(gen));

        glm::vec2 pos(r * std::cos(angle), r * std::sin(angle));
        particles.emplace_back(pos, glm::vec2(0.0f), 1.0f);
    }
}

int CollapsePreset::getParticleCount() const {
    return num_particles;
}
