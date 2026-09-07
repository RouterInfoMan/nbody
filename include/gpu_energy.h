#pragma once
#include "shader.h"
#include <GL/glew.h>
#include <glm/glm.hpp>
#include <memory>
#include <vector>

class GpuEnergy {
public:
    GpuEnergy() = default;
    GpuEnergy(const GpuEnergy&) = delete;
    GpuEnergy& operator=(const GpuEnergy&) = delete;

    ~GpuEnergy() { if (partial_ssbo) glDeleteBuffers(1, &partial_ssbo); }

    bool totals(GLuint vel, GLuint mass, GLuint pot, int count,
                double& kinetic, double& potential) {
        if (count <= 0 || !vel || !mass || !pot) return false;

        if (!shader) shader = std::make_unique<Shader>("shaders/energy_reduce.comp");

        const int groups = (count + 255) / 256;
        if (!partial_ssbo) glGenBuffers(1, &partial_ssbo);
        if (groups > capacity) {
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, partial_ssbo);
            glBufferData(GL_SHADER_STORAGE_BUFFER,
                         GLsizeiptr(groups) * sizeof(glm::vec2), nullptr, GL_DYNAMIC_READ);
            capacity = groups;
        }

        shader->use();
        shader->set_int("particleCount", count);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, vel);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, mass);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, pot);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, partial_ssbo);
        glDispatchCompute(static_cast<GLuint>(groups), 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);

        host.resize(static_cast<size_t>(groups));
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, partial_ssbo);
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
                           GLsizeiptr(groups) * sizeof(glm::vec2), host.data());

        double k = 0.0, u = 0.0;
        for (const glm::vec2& v : host) { k += double(v.x); u += double(v.y); }
        kinetic = k;
        potential = u;
        return true;
    }

private:
    std::unique_ptr<Shader> shader;
    GLuint partial_ssbo = 0;
    int capacity = 0;
    std::vector<glm::vec2> host;
};
