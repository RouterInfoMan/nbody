#include "renderer.h"
#include <iostream>

Renderer::Renderer() = default;

Renderer::~Renderer() {
    if (particle_vao) glDeleteVertexArrays(1, &particle_vao);
    if (particle_vbo) glDeleteBuffers(1, &particle_vbo);
    if (position_ssbo) glDeleteBuffers(1, &position_ssbo);
    if (velocity_ssbo) glDeleteBuffers(1, &velocity_ssbo);
}

void Renderer::initialize() {
    particle_shader = std::make_unique<Shader>("shaders/particle.vert", "shaders/particle.frag");

    use_compute = checkComputeSupport();
    if (use_compute) {
        compute_shader = std::make_unique<Shader>("shaders/color_compute.comp");
        glGenBuffers(1, &position_ssbo);
        glGenBuffers(1, &velocity_ssbo);
    }

    setupBuffers();

    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
}

bool Renderer::checkComputeSupport() {
    GLint major, minor;
    glGetIntegerv(GL_MAJOR_VERSION, &major);
    glGetIntegerv(GL_MINOR_VERSION, &minor);
    return (major > 4 || (major == 4 && minor >= 3));
}

void Renderer::render(const std::vector<Particle>& particles, const Camera2D& camera, bool force_update) {
    if (particles.empty()) return;

    if (force_update || needs_update || particles.size() != last_particle_count) {
        if (use_compute && show_velocity)
            updateBufferCompute(particles);
        else
            updateBufferCPU(particles);
        last_particle_count = particles.size();
        needs_update = false;
    }

    particle_shader->use();
    particle_shader->set_mat4("projection", camera.getProjectionMatrix());
    particle_shader->set_float("pointSize", particle_size);

    glBindVertexArray(particle_vao);
    glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(particles.size()));
}

void Renderer::updateBufferCompute(const std::vector<Particle>& particles) {
    size_t n = particles.size();

    std::vector<glm::vec2> positions(n);
    std::vector<glm::vec2> velocities(n);
    for (size_t i = 0; i < n; i++) {
        positions[i] = particles[i].position;
        velocities[i] = particles[i].velocity;
    }

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, position_ssbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, n * sizeof(glm::vec2), positions.data(), GL_STREAM_DRAW);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, velocity_ssbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, n * sizeof(glm::vec2), velocities.data(), GL_STREAM_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, particle_vbo);
    glBufferData(GL_ARRAY_BUFFER, n * 6 * sizeof(float), nullptr, GL_STREAM_DRAW);

    compute_shader->use();
    compute_shader->set_int("particleCount", static_cast<int>(n));

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, position_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, velocity_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, particle_vbo);

    glDispatchCompute(static_cast<GLuint>((n + 255) / 256), 1, 1);
    glMemoryBarrier(GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT);
}

void Renderer::updateBufferCPU(const std::vector<Particle>& particles) {
    std::vector<float> data;
    data.reserve(particles.size() * 6);

    for (const auto& p : particles) {
        data.push_back(p.position.x);
        data.push_back(p.position.y);
        data.push_back(0.0f);

        glm::vec3 color(1.0f);
        if (show_velocity)
            color = velocityToColor(p.velocity);
        data.push_back(color.r);
        data.push_back(color.g);
        data.push_back(color.b);
    }

    glBindBuffer(GL_ARRAY_BUFFER, particle_vbo);
    glBufferData(GL_ARRAY_BUFFER, data.size() * sizeof(float), data.data(), GL_STREAM_DRAW);
}

void Renderer::setupBuffers() {
    glGenVertexArrays(1, &particle_vao);
    glGenBuffers(1, &particle_vbo);

    glBindVertexArray(particle_vao);
    glBindBuffer(GL_ARRAY_BUFFER, particle_vbo);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
}

glm::vec3 Renderer::velocityToColor(const glm::vec2& velocity) {
    float speed = glm::length(velocity);
    float hue = glm::clamp(std::log(speed + 1.0f) / 5.0f, 0.0f, 1.0f);

    float r = std::abs(hue * 6.0f - 3.0f) - 1.0f;
    float g = 2.0f - std::abs(hue * 6.0f - 2.0f);
    float b = 2.0f - std::abs(hue * 6.0f - 4.0f);

    return glm::vec3(glm::clamp(r, 0.0f, 1.0f),
                     glm::clamp(g, 0.0f, 1.0f),
                     glm::clamp(b, 0.0f, 1.0f));
}
