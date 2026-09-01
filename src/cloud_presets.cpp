#include "cloud_presets.h"
#include <algorithm>
#include <cmath>
#include <glm/gtc/constants.hpp>

namespace clouds {

void emit(std::vector<Particle>& out, std::mt19937& gen, int n,
          glm::vec2 center, glm::vec2 bulk, float mass,
          const CloudParams& shape, float G) {
    if (n <= 0 || mass <= 0.0f) return;

    const float r_max = std::max(shape.radius, 1e-3f);
    const float a = r_max / 3.0f;

    // 2D Plummer: M(<r)/M = r^2 / (r^2 + a^2), so the inverse CDF is
    // r = a * sqrt(u / (1 - u)). Sampling u only up to the value that maps to
    // r_max truncates the profile cleanly instead of rejection-sampling.
    const float u_max = (r_max * r_max) / (r_max * r_max + a * a);

    // Virial estimate: a bound self-gravitating system has <v^2> ~ GM/R, split
    // across two dimensions. Approximate, which is why `support` is a slider.
    const float sigma = shape.support * std::sqrt(G * mass / (2.0f * a));

    std::uniform_real_distribution<float> unit(0.0f, u_max);
    std::uniform_real_distribution<float> angle_dist(0.0f, 2.0f * glm::pi<float>());
    std::normal_distribution<float> gauss(0.0f, 1.0f);

    const float per_particle = mass / float(n);

    for (int i = 0; i < n; i++) {
        const float u = unit(gen);
        const float r = a * std::sqrt(u / std::max(1.0f - u, 1e-6f));
        const float theta = angle_dist(gen);

        const glm::vec2 pos = center + glm::vec2(r * std::cos(theta), r * std::sin(theta));
        const glm::vec2 vel = bulk + glm::vec2(gauss(gen), gauss(gen)) * sigma;

        out.emplace_back(pos, vel, per_particle);
    }
}

}  // namespace clouds

// ---------------------------------------------------------------------------

BinaryCloudsPreset::BinaryCloudsPreset(const BinaryCloudsParams& params) : params(params) {}

void BinaryCloudsPreset::apply(std::vector<Particle>& particles) {
    particles.clear();
    particles.reserve(params.count);

    const float total = std::max(params.total_mass, 1e-6f);
    const float ratio = std::max(params.mass_ratio, 1e-4f);
    const float m1 = total / (1.0f + ratio);
    const float m2 = total - m1;

    const float d = std::max(params.separation, 1e-3f);

    // Both clouds orbit the barycentre, so each sits at a distance weighted by
    // the *other* one's mass and moves at a speed weighted the same way.
    const glm::vec2 c1(-d * (m2 / total), 0.0f);
    const glm::vec2 c2( d * (m1 / total), 0.0f);

    const float v_rel = params.orbit_fraction * std::sqrt(params.G * total / d);
    const glm::vec2 v1(0.0f, -v_rel * (m2 / total));
    const glm::vec2 v2(0.0f,  v_rel * (m1 / total));

    CloudParams shape;
    shape.radius = params.cloud_radius;
    shape.support = params.support;

    // Split the particle budget by mass so both clouds have the same particle
    // mass, which keeps two-body relaxation uniform across the pair.
    const int n1 = std::clamp(int(std::lround(double(params.count) * double(m1) / double(total))),
                              1, params.count - 1);
    const int n2 = params.count - n1;

    std::mt19937 gen(params.seed);
    clouds::emit(particles, gen, n1, c1, v1, m1, shape, params.G);
    clouds::emit(particles, gen, n2, c2, v2, m2, shape, params.G);
}

// ---------------------------------------------------------------------------

CloudClusterPreset::CloudClusterPreset(const CloudClusterParams& params) : params(params) {}

void CloudClusterPreset::apply(std::vector<Particle>& particles) {
    particles.clear();
    particles.reserve(params.count);

    const int k = std::clamp(params.cloud_count, 1, 32);
    const float total = std::max(params.total_mass, 1e-6f);
    const float per_cloud = total / float(k);
    const float cluster_r = std::max(params.cluster_radius, 1e-3f);

    CloudParams shape;
    shape.radius = params.cloud_radius;
    shape.support = params.support;

    std::mt19937 gen(params.seed);
    std::uniform_real_distribution<float> unit(0.0f, 1.0f);
    std::uniform_real_distribution<float> angle_dist(0.0f, 2.0f * glm::pi<float>());

    for (int i = 0; i < k; i++) {
        // Even split of the budget, with the remainder spread over the first
        // few clouds so the totals match getParticleCount() exactly.
        const int n = params.count / k + (i < params.count % k ? 1 : 0);

        // sqrt keeps the cloud centres uniform over the cluster area.
        const float r = cluster_r * std::sqrt(unit(gen));
        const float theta = angle_dist(gen);
        const glm::vec2 center(r * std::cos(theta), r * std::sin(theta));

        // Circular speed against the mass interior to this cloud. Uniform
        // area coverage means the enclosed fraction is (r / cluster_r)^2.
        glm::vec2 bulk(0.0f);
        if (r > 1e-4f) {
            const float enclosed = total * (r * r) / (cluster_r * cluster_r);
            const float v = params.orbit_fraction * std::sqrt(params.G * enclosed / r);
            bulk = glm::vec2(-std::sin(theta), std::cos(theta)) * v;
        }

        clouds::emit(particles, gen, n, center, bulk, per_cloud, shape, params.G);
    }
}
