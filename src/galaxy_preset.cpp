#include "galaxy_preset.h"
#include <cmath>
#include <random>
#include <glm/gtc/constants.hpp>

GalaxyPreset::GalaxyPreset(int num_particles, float radius)
    : num_particles(num_particles), radius(radius) {}

void GalaxyPreset::apply(std::vector<Particle>& particles) {
    particles.clear();
    particles.reserve(num_particles + 1);

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> angle_dist(0.0f, 2.0f * glm::pi<float>());
    std::uniform_real_distribution<float> unit_dist(0.0f, 1.0f);
    std::normal_distribution<float> perturb(1.0f, 0.02f);

    float bh_mass = 10000.0f;
    float inner_r = 10.0f;

    particles.emplace_back(glm::vec2(0.0f), glm::vec2(0.0f), bh_mass);

    float G = 1.0f;
    for (int i = 0; i < num_particles; i++) {
        float angle = angle_dist(gen);
        float t = std::sqrt(unit_dist(gen));
        float r = inner_r + t * (radius - inner_r);

        glm::vec2 pos(r * std::cos(angle), r * std::sin(angle));

        float orbital_speed = std::sqrt(G * bh_mass / r) * perturb(gen);
        glm::vec2 vel(-orbital_speed * std::sin(angle),
                       orbital_speed * std::cos(angle));

        particles.emplace_back(pos, vel, 0.1f);
    }
}

int GalaxyPreset::getParticleCount() const {
    return num_particles + 1;
}
