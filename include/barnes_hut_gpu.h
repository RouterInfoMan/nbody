#pragma once
#include "simulation.h"
#include "preset.h"
#include "shader.h"
#include "gpu_boundary.h"
#include "gpu_energy.h"
#include "gpu_radix_sort.h"
#include <GL/glew.h>
#include <memory>

// Barnes-Hut on the GPU: Morton sort -> LBVH -> node moments -> traversal,
// all on device with no readback.
class BarnesHutGPU : public Simulation {
public:
    BarnesHutGPU(std::unique_ptr<Preset> preset, float theta = 0.5f,
                 float G = 1.0f, float softening = 1.0f);
    ~BarnesHutGPU();

    void step(float dt) override;
    void reset() override;
    const char* name() const override { return "Barnes-Hut (GPU)"; }

    bool isGPUResident() const override { return true; }
    GLuint positionBuffer() const override { return pos_ssbo[cur]; }
    GLuint velocityBuffer() const override { return vel_ssbo[cur]; }
    GLuint massBuffer() const override { return mass_ssbo[cur]; }
    GLuint idBuffer() const override { return id_ssbo[cur]; }
    void syncToHost() override;

    bool energyTotals(double& kinetic, double& potential) override {
        return gpu_energy.totals(vel_ssbo[cur], mass_ssbo[cur], pot_ssbo,
                                 getParticleCount(), kinetic, potential);
    }

    bool treeGeometry(GLuint& buffer, int& count) override {
        const int n = getParticleCount();
        if (n < 2 || !node_aabb_ssbo) return false;
        buffer = node_aabb_ssbo;
        count = 2 * n - 1;
        return true;
    }
    const char* treeKind() const override { return "LBVH node AABBs"; }

    void setPhysics(float g, float soft) override { G = g; softening = soft; }
    void setTheta(float t) override { theta = t; }
    void setQuadrupole(bool on) { use_quadrupole = on; }
    bool quadrupole() const { return use_quadrupole; }

private:
    GpuBoundary gpu_boundary;
    std::unique_ptr<Preset> preset;

    float theta;
    float G;
    float softening;
    bool use_quadrupole = true;

    std::unique_ptr<Shader> bounds_shader;
    std::unique_ptr<Shader> morton_shader;
    std::unique_ptr<Shader> reorder_shader;
    std::unique_ptr<Shader> karras_shader;
    std::unique_ptr<Shader> merge_shader;
    GLint merge_level_loc = -1;
    std::unique_ptr<Shader> force_shader;
    std::unique_ptr<Shader> verlet1_shader;
    std::unique_ptr<Shader> verlet2_shader;

    GpuRadixSort sorter;

    GLuint pos_ssbo[2] = {0, 0};
    GLuint vel_ssbo[2] = {0, 0};
    GLuint mass_ssbo[2] = {0, 0};
    GLuint id_ssbo[2] = {0, 0};
    int cur = 0;

    GLuint acc_ssbo = 0;
    GLuint pot_ssbo = 0;
    GpuEnergy gpu_energy;
    GLuint key_ssbo[2] = {0, 0};
    GLuint val_ssbo[2] = {0, 0};
    GLuint bounds_ssbo = 0;

    // Node layout: [0, n-1) internal, [n-1, 2n-1) leaves.
    GLuint node_left_ssbo = 0;
    GLuint node_right_ssbo = 0;
    GLuint node_level_ssbo = 0;
    GLuint node_com_ssbo = 0;
    GLuint node_aabb_ssbo = 0;
    GLuint node_quad_ssbo = 0;

    static constexpr int WORKGROUP_SIZE = 256;
    static constexpr int MAX_MERGE_LEVEL = 63;

    void allocateBuffers();
    void freeBuffers();
    void uploadToGPU();

    void buildTreeAndComputeForces();

    GLuint groupCount(int n) const {
        return static_cast<GLuint>((n + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE);
    }
};
