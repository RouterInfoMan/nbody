#pragma once
#include <GL/glew.h>
#include <memory>
#include <vector>
#include "shader.h"
#include "camera.h"
#include "particle.h"

class Renderer {
public:
    Renderer();
    ~Renderer();

    void initialize();
    void render(const std::vector<Particle>& particles, const Camera2D& camera, bool force_update);

    // Draws straight out of a solver's device buffers: no readback, no upload.
    // pos_ssbo/vel_ssbo must be std430 arrays of vec2 with `count` entries.
    void renderFromGPU(GLuint pos_ssbo, GLuint vel_ssbo, int count, const Camera2D& camera);

    void setParticleSize(float size) { particle_size = size; needs_update = true; }
    void setShowVelocity(bool show) { show_velocity = show; needs_update = true; }
    void setNeedsUpdate(bool update) { needs_update = update; }

private:
    std::unique_ptr<Shader> particle_shader;
    std::unique_ptr<Shader> compute_shader;
    std::unique_ptr<Shader> flat_shader;   // lazily built, GPU-resident path only

    GLuint particle_vao = 0, particle_vbo = 0;
    GLuint position_ssbo = 0, velocity_ssbo = 0;

    float particle_size = 2.0f;
    bool show_velocity = true;
    size_t last_particle_count = 0;
    size_t vertex_capacity = 0;
    bool needs_update = true;
    bool use_compute = false;

    void setupBuffers();
    void updateBufferCompute(const std::vector<Particle>& particles);
    void updateBufferCPU(const std::vector<Particle>& particles);
    void ensureVertexCapacity(size_t count);
    bool checkComputeSupport();

    glm::vec3 velocityToColor(const glm::vec2& velocity);
};
