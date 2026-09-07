#pragma once
#include "simulation.h"
#include "preset.h"
#include "shader.h"
#include "gpu_energy.h"
#include "gpu_boundary.h"
#include <GL/glew.h>
#include <memory>

class BruteForceGPU : public Simulation {
public:
    BruteForceGPU(std::unique_ptr<Preset> preset, float G = 1.0f, float softening = 1.0f);
    ~BruteForceGPU();

    void step(float dt) override;
    void reset() override;
    const char* name() const override { return "Brute Force (GPU)"; }

    bool isGPUResident() const override { return true; }
    GLuint positionBuffer() const override { return pos_ssbo; }
    GLuint velocityBuffer() const override { return vel_ssbo; }
    GLuint massBuffer() const override { return mass_ssbo; }
    void syncToHost() override;

    bool energyTotals(double& kinetic, double& potential) override {
        return gpu_energy.totals(vel_ssbo, mass_ssbo, pot_ssbo,
                                 getParticleCount(), kinetic, potential);
    }

private:
    GpuBoundary gpu_boundary;
    std::unique_ptr<Preset> preset;
    float G;
    float softening;

    std::unique_ptr<Shader> force_shader;
    std::unique_ptr<Shader> verlet1_shader;
    std::unique_ptr<Shader> verlet2_shader;

    GLuint pos_ssbo = 0;
    GLuint vel_ssbo = 0;
    GLuint acc_ssbo = 0;
    GLuint mass_ssbo = 0;
    GLuint pot_ssbo = 0;
    GpuEnergy gpu_energy;
    int gpu_particle_count = 0;

    static constexpr int WORKGROUP_SIZE = 256;

    void allocateBuffers();
    void freeBuffers();
    void uploadToGPU();
};
