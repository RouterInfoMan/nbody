#include "brute_force_cpu.h"
#include <cmath>

BruteForceCPU::BruteForceCPU(std::unique_ptr<Preset> preset, float G, float softening)
    : preset(std::move(preset)), G(G), softening_sq(softening * softening) {
    reset();
}

void BruteForceCPU::reset() {
    current_time = 0.0f;
    preset->apply(particles);
    setParticleCount(static_cast<int>(particles.size()));

    if (!particles.empty()) computeForces();
}

void BruteForceCPU::step(float dt) {
    verletStep1(dt);
    computeForces();
    verletStep2(dt);

    if (boundary.active()) {
        const int n = static_cast<int>(particles.size());
        pool.parallel_for(0, n, [&](int a, int b) { applyBoundaryRange(a, b, dt); });
    }

    current_time += dt;
}

void BruteForceCPU::computeForces() {
    int n = static_cast<int>(particles.size());

    pool.parallel_for(0, n, [&](int start, int end) {
        for (int i = start; i < end; i++) {
            glm::vec2 acc(0.0f);
            float pot = 0.0f;
            glm::vec2 pi = particles[i].position;

            for (int j = 0; j < n; j++) {
                if (j == i) continue;
                glm::vec2 d = particles[j].position - pi;
                float dist_sq = glm::dot(d, d) + softening_sq;
                float inv_dist = 1.0f / std::sqrt(dist_sq);
                float inv_dist3 = inv_dist * inv_dist * inv_dist;
                acc += d * (particles[j].mass * inv_dist3);
                pot -= particles[j].mass * inv_dist;
            }

            particles[i].acceleration = acc * G;
            particles[i].potential = pot * G;
        }
    });
}

void BruteForceCPU::verletStep1(float dt) {
    int n = static_cast<int>(particles.size());
    pool.parallel_for(0, n, [&](int start, int end) {
        for (int i = start; i < end; i++) {
            particles[i].prev_acceleration = particles[i].acceleration;
            particles[i].position += particles[i].velocity * dt
                                   + 0.5f * particles[i].acceleration * dt * dt;
        }
    });
}

void BruteForceCPU::verletStep2(float dt) {
    int n = static_cast<int>(particles.size());
    pool.parallel_for(0, n, [&](int start, int end) {
        for (int i = start; i < end; i++) {
            particles[i].velocity += 0.5f * (particles[i].prev_acceleration
                                           + particles[i].acceleration) * dt;
        }
    });
}
