#include "barnes_hut_gpu.h"
#include <vector>

namespace {
constexpr GLuint KEY_MAX = 0xFFFFFFFFu;

void allocBuffer(GLuint& buf, GLsizeiptr bytes, const void* data = nullptr) {
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buf);
    glBufferData(GL_SHADER_STORAGE_BUFFER, bytes, data, GL_DYNAMIC_DRAW);
}

void clearUint(GLuint buf, GLuint value) {
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buf);
    glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_R32UI, GL_RED_INTEGER,
                      GL_UNSIGNED_INT, &value);
}
}  // namespace

BarnesHutGPU::BarnesHutGPU(std::unique_ptr<Preset> preset, float theta,
                           float G, float softening)
    : preset(std::move(preset)), theta(theta), G(G), softening(softening) {
    bounds_shader    = std::make_unique<Shader>("shaders/bh_bounds.comp");
    morton_shader    = std::make_unique<Shader>("shaders/bh_morton.comp");
    reorder_shader   = std::make_unique<Shader>("shaders/bh_reorder.comp");
    karras_shader    = std::make_unique<Shader>("shaders/bh_karras.comp");
    propagate_shader = std::make_unique<Shader>("shaders/bh_propagate.comp");
    force_shader     = std::make_unique<Shader>("shaders/bh_force.comp");
    verlet1_shader   = std::make_unique<Shader>("shaders/verlet_step1.comp");
    verlet2_shader   = std::make_unique<Shader>("shaders/verlet_step2.comp");

    reset();
}

BarnesHutGPU::~BarnesHutGPU() {
    freeBuffers();
}

void BarnesHutGPU::reset() {
    current_time = 0.0f;
    preset->apply(particles);
    setParticleCount(static_cast<int>(particles.size()));

    freeBuffers();
    allocateBuffers();
    uploadToGPU();

    // Seed the accelerations so the first drift half-step is correct.
    if (getParticleCount() >= 2)
        buildTreeAndComputeForces();
}

void BarnesHutGPU::step(float dt) {
    const int n = getParticleCount();
    if (n < 2) return;

    const GLuint groups = groupCount(n);

    // Drift + first half kick.
    verlet1_shader->use();
    verlet1_shader->set_int("particleCount", n);
    verlet1_shader->set_float("dt", dt);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, pos_ssbo[cur]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, vel_ssbo[cur]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, acc_ssbo);
    glDispatchCompute(groups, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    buildTreeAndComputeForces();

    // Second half kick. Note the state was permuted into Morton order in
    // between; velocities were permuted with it, so this stays consistent.
    verlet2_shader->use();
    verlet2_shader->set_int("particleCount", n);
    verlet2_shader->set_float("dt", dt);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, vel_ssbo[cur]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, acc_ssbo);
    glDispatchCompute(groups, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    current_time += dt;
}

void BarnesHutGPU::buildTreeAndComputeForces() {
    const int n = getParticleCount();
    const GLuint groups = groupCount(n);
    const GLuint internal_groups = groupCount(n - 1);

    // --- Bounding box -------------------------------------------------
    const GLuint bounds_init[4] = { KEY_MAX, KEY_MAX, 0u, 0u };
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, bounds_ssbo);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(bounds_init), bounds_init);

    bounds_shader->use();
    bounds_shader->set_int("particleCount", n);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, pos_ssbo[cur]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, bounds_ssbo);
    glDispatchCompute(groups, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    // --- Morton codes -------------------------------------------------
    morton_shader->use();
    morton_shader->set_int("particleCount", n);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, pos_ssbo[cur]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, bounds_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, key_ssbo[0]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, val_ssbo[0]);
    glDispatchCompute(groups, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    // --- Sort ---------------------------------------------------------
    sorter.sort(key_ssbo[0], val_ssbo[0], key_ssbo[1], val_ssbo[1], n);

    // --- Permute live state into Morton order -------------------------
    reorder_shader->use();
    reorder_shader->set_int("particleCount", n);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, pos_ssbo[cur]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, vel_ssbo[cur]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, mass_ssbo[cur]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, val_ssbo[0]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, pos_ssbo[1 - cur]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, vel_ssbo[1 - cur]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, mass_ssbo[1 - cur]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, id_ssbo[cur]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 8, id_ssbo[1 - cur]);
    glDispatchCompute(groups, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    cur = 1 - cur;

    // --- Radix tree ---------------------------------------------------
    // The root is the only node nothing links to, so its parent slot is the
    // sentinel that terminates the upward walk in bh_propagate.
    clearUint(node_flags_ssbo, 0u);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, node_parent_ssbo);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GLuint), &NO_PARENT);

    karras_shader->use();
    karras_shader->set_int("particleCount", n);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, key_ssbo[0]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, node_left_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, node_right_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, node_parent_ssbo);
    glDispatchCompute(internal_groups, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    // --- Bottom-up centre of mass -------------------------------------
    propagate_shader->use();
    propagate_shader->set_int("particleCount", n);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, pos_ssbo[cur]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, mass_ssbo[cur]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, node_left_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, node_right_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, node_parent_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, node_com_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, node_aabb_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, node_flags_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 8, node_quad_ssbo);
    glDispatchCompute(groups, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    // --- Traversal ----------------------------------------------------
    force_shader->use();
    force_shader->set_int("particleCount", n);
    force_shader->set_float("invTheta", 1.0f / theta);
    force_shader->set_float("softeningSq", softening * softening);
    force_shader->set_float("G", G);
    force_shader->set_int("useQuadrupole", use_quadrupole ? 1 : 0);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, pos_ssbo[cur]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, node_com_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, node_aabb_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, node_left_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, node_right_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, acc_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, node_quad_ssbo);
    glDispatchCompute(groups, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

void BarnesHutGPU::allocateBuffers() {
    const int n = getParticleCount();
    if (n <= 0) return;

    const GLsizeiptr vec2_bytes = GLsizeiptr(n) * sizeof(glm::vec2);
    const GLsizeiptr float_bytes = GLsizeiptr(n) * sizeof(float);
    const GLsizeiptr uint_bytes = GLsizeiptr(n) * sizeof(GLuint);

    const int internal_count = n - 1;
    const int total_nodes = 2 * n - 1;

    GLuint* handles[] = {
        &pos_ssbo[0], &pos_ssbo[1], &vel_ssbo[0], &vel_ssbo[1],
        &mass_ssbo[0], &mass_ssbo[1], &id_ssbo[0], &id_ssbo[1], &acc_ssbo,
        &key_ssbo[0], &key_ssbo[1], &val_ssbo[0], &val_ssbo[1],
        &bounds_ssbo,
        &node_left_ssbo, &node_right_ssbo, &node_parent_ssbo,
        &node_com_ssbo, &node_aabb_ssbo, &node_flags_ssbo, &node_quad_ssbo,
    };
    for (GLuint* h : handles) glGenBuffers(1, h);

    for (int i = 0; i < 2; i++) {
        allocBuffer(pos_ssbo[i], vec2_bytes);
        allocBuffer(vel_ssbo[i], vec2_bytes);
        allocBuffer(mass_ssbo[i], float_bytes);
        allocBuffer(id_ssbo[i], uint_bytes);
        allocBuffer(key_ssbo[i], uint_bytes);
        allocBuffer(val_ssbo[i], uint_bytes);
    }
    allocBuffer(acc_ssbo, vec2_bytes);
    allocBuffer(bounds_ssbo, 4 * sizeof(GLuint));

    if (internal_count > 0) {
        allocBuffer(node_left_ssbo, GLsizeiptr(internal_count) * sizeof(GLuint));
        allocBuffer(node_right_ssbo, GLsizeiptr(internal_count) * sizeof(GLuint));
        allocBuffer(node_flags_ssbo, GLsizeiptr(internal_count) * sizeof(GLuint));
    }
    allocBuffer(node_parent_ssbo, GLsizeiptr(total_nodes) * sizeof(GLuint));
    allocBuffer(node_com_ssbo, GLsizeiptr(total_nodes) * sizeof(glm::vec4));
    allocBuffer(node_aabb_ssbo, GLsizeiptr(total_nodes) * sizeof(glm::vec4));
    allocBuffer(node_quad_ssbo, GLsizeiptr(total_nodes) * sizeof(glm::vec4));

    cur = 0;
}

void BarnesHutGPU::freeBuffers() {
    GLuint* handles[] = {
        &pos_ssbo[0], &pos_ssbo[1], &vel_ssbo[0], &vel_ssbo[1],
        &mass_ssbo[0], &mass_ssbo[1], &id_ssbo[0], &id_ssbo[1], &acc_ssbo,
        &key_ssbo[0], &key_ssbo[1], &val_ssbo[0], &val_ssbo[1],
        &bounds_ssbo,
        &node_left_ssbo, &node_right_ssbo, &node_parent_ssbo,
        &node_com_ssbo, &node_aabb_ssbo, &node_flags_ssbo, &node_quad_ssbo,
    };
    for (GLuint* h : handles) {
        if (*h) { glDeleteBuffers(1, h); *h = 0; }
    }
}

void BarnesHutGPU::uploadToGPU() {
    const int n = getParticleCount();
    if (n <= 0) return;

    std::vector<glm::vec2> positions(n), velocities(n), zeros(n, glm::vec2(0.0f));
    std::vector<float> masses(n);
    std::vector<GLuint> ids(n);

    for (int i = 0; i < n; i++) {
        positions[i] = particles[i].position;
        velocities[i] = particles[i].velocity;
        masses[i] = particles[i].mass;
        ids[i] = GLuint(i);
    }

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, pos_ssbo[0]);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, GLsizeiptr(n) * sizeof(glm::vec2), positions.data());
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, vel_ssbo[0]);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, GLsizeiptr(n) * sizeof(glm::vec2), velocities.data());
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, mass_ssbo[0]);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, GLsizeiptr(n) * sizeof(float), masses.data());
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, id_ssbo[0]);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, GLsizeiptr(n) * sizeof(GLuint), ids.data());
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, acc_ssbo);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, GLsizeiptr(n) * sizeof(glm::vec2), zeros.data());

    cur = 0;
}

void BarnesHutGPU::syncToHost() {
    const int n = getParticleCount();
    if (n <= 0) return;

    std::vector<glm::vec2> positions(n), velocities(n), accelerations(n);
    std::vector<float> masses(n);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, pos_ssbo[cur]);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, GLsizeiptr(n) * sizeof(glm::vec2), positions.data());
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, vel_ssbo[cur]);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, GLsizeiptr(n) * sizeof(glm::vec2), velocities.data());
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, mass_ssbo[cur]);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, GLsizeiptr(n) * sizeof(float), masses.data());
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, acc_ssbo);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, GLsizeiptr(n) * sizeof(glm::vec2), accelerations.data());

    // Particles were permuted into Morton order on device, so index i no
    // longer refers to the particle it did at construction.
    for (int i = 0; i < n; i++) {
        particles[i].position = positions[i];
        particles[i].velocity = velocities[i];
        particles[i].acceleration = accelerations[i];
        particles[i].mass = masses[i];
    }
}
