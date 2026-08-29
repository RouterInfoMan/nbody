#include "brute_force_gpu.h"
#include <iostream>
#include <vector>

BruteForceGPU::BruteForceGPU(std::unique_ptr<Preset> preset, float G, float softening)
    : preset(std::move(preset)), G(G), softening(softening) {
    force_shader = std::make_unique<Shader>("shaders/brute_force.comp");
    verlet1_shader = std::make_unique<Shader>("shaders/verlet_step1.comp");
    verlet2_shader = std::make_unique<Shader>("shaders/verlet_step2.comp");
    reset();
}

BruteForceGPU::~BruteForceGPU() {
    freeBuffers();
}

void BruteForceGPU::reset() {
    current_time = 0.0f;
    preset->apply(particles);

    freeBuffers();
    allocateBuffers();
    uploadToGPU();

    int n = static_cast<int>(particles.size());
    GLuint groups = (n + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;

    force_shader->use();
    force_shader->set_int("particleCount", n);
    force_shader->set_float("softening", softening);
    force_shader->set_float("G", G);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, pos_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, vel_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, acc_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, mass_ssbo);
    glDispatchCompute(groups, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

void BruteForceGPU::step(float dt) {
    int n = static_cast<int>(particles.size());
    GLuint groups = (n + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, pos_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, vel_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, acc_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, mass_ssbo);

    verlet1_shader->use();
    verlet1_shader->set_int("particleCount", n);
    verlet1_shader->set_float("dt", dt);
    glDispatchCompute(groups, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    force_shader->use();
    force_shader->set_int("particleCount", n);
    force_shader->set_float("softening", softening);
    force_shader->set_float("G", G);
    glDispatchCompute(groups, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    verlet2_shader->use();
    verlet2_shader->set_int("particleCount", n);
    verlet2_shader->set_float("dt", dt);
    glDispatchCompute(groups, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    glFinish();

    downloadFromGPU();
    current_time += dt;
}

void BruteForceGPU::allocateBuffers() {
    gpu_particle_count = static_cast<int>(particles.size());

    glGenBuffers(1, &pos_ssbo);
    glGenBuffers(1, &vel_ssbo);
    glGenBuffers(1, &acc_ssbo);
    glGenBuffers(1, &mass_ssbo);
}

void BruteForceGPU::freeBuffers() {
    if (pos_ssbo)  { glDeleteBuffers(1, &pos_ssbo);  pos_ssbo = 0; }
    if (vel_ssbo)  { glDeleteBuffers(1, &vel_ssbo);  vel_ssbo = 0; }
    if (acc_ssbo)  { glDeleteBuffers(1, &acc_ssbo);  acc_ssbo = 0; }
    if (mass_ssbo) { glDeleteBuffers(1, &mass_ssbo); mass_ssbo = 0; }
}

void BruteForceGPU::uploadToGPU() {
    int n = static_cast<int>(particles.size());

    std::vector<glm::vec2> positions(n), velocities(n);
    std::vector<float> masses(n);

    for (int i = 0; i < n; i++) {
        positions[i] = particles[i].position;
        velocities[i] = particles[i].velocity;
        masses[i] = particles[i].mass;
    }

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, pos_ssbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, n * sizeof(glm::vec2), positions.data(), GL_DYNAMIC_DRAW);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, vel_ssbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, n * sizeof(glm::vec2), velocities.data(), GL_DYNAMIC_DRAW);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, acc_ssbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, n * sizeof(glm::vec2), nullptr, GL_DYNAMIC_DRAW);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, mass_ssbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, n * sizeof(float), masses.data(), GL_DYNAMIC_DRAW);
}

void BruteForceGPU::downloadFromGPU() {
    int n = static_cast<int>(particles.size());

    std::vector<glm::vec2> positions(n), velocities(n);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, pos_ssbo);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, n * sizeof(glm::vec2), positions.data());

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, vel_ssbo);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, n * sizeof(glm::vec2), velocities.data());

    for (int i = 0; i < n; i++) {
        particles[i].position = positions[i];
        particles[i].velocity = velocities[i];
    }
}
