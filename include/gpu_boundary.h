#pragma once
#include "boundary.h"
#include "shader.h"
#include <GL/glew.h>
#include <memory>

// Applies a Boundary to device-resident state.
class GpuBoundary {
public:
    GpuBoundary() = default;

    GpuBoundary(const GpuBoundary&) = delete;
    GpuBoundary& operator=(const GpuBoundary&) = delete;

    void apply(GLuint pos_ssbo, GLuint vel_ssbo, int count,
               const Boundary& boundary, float dt) {
        if (!boundary.active() || count <= 0 || !pos_ssbo || !vel_ssbo) return;

        if (!shader) shader = std::make_unique<Shader>("shaders/boundary.comp");

        shader->use();
        shader->set_int("particleCount", count);
        shader->set_int("mode", boundary.mode);
        shader->set_int("shape", boundary.shape);
        shader->set_float("radius", boundary.radius);
        shader->set_float("restitution", boundary.restitution);
        shader->set_float("drag", boundary.drag);
        shader->set_float("dt", dt);

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, pos_ssbo);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, vel_ssbo);
        glDispatchCompute(static_cast<GLuint>((count + 255) / 256), 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    }

private:
    std::unique_ptr<Shader> shader;
};
