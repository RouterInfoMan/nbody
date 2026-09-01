#pragma once
#include "simulation.h"
#include "preset.h"
#include "shader.h"
#include "gpu_boundary.h"
#include "gpu_radix_sort.h"
#include <GL/glew.h>
#include <memory>

// Barnes-Hut on the GPU via a linear BVH (Karras 2012).
//
// Per step, entirely on device:
//   verlet drift  ->  AABB reduce  ->  Morton codes  ->  radix sort
//   ->  permute state into Morton order  ->  parallel radix tree
//   ->  bottom-up COM/AABB  ->  traversal  ->  verlet kick
//
// Nothing is read back; the renderer draws out of these buffers directly.
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

    // The traversal's own AABBs, handed over as-is: the overlay needs no copy
    // and no readback. Leaves hold one particle and collapse to a point, which
    // the collector culls.
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
    std::unique_ptr<Shader> propagate_shader;
    std::unique_ptr<Shader> force_shader;
    std::unique_ptr<Shader> verlet1_shader;
    std::unique_ptr<Shader> verlet2_shader;

    GpuRadixSort sorter;

    // Particle state, double buffered because the Morton permutation is
    // applied to the live arrays every step.
    GLuint pos_ssbo[2] = {0, 0};
    GLuint vel_ssbo[2] = {0, 0};
    GLuint mass_ssbo[2] = {0, 0};
    GLuint id_ssbo[2] = {0, 0};
    int cur = 0;

    GLuint acc_ssbo = 0;
    GLuint key_ssbo[2] = {0, 0};
    GLuint val_ssbo[2] = {0, 0};
    GLuint bounds_ssbo = 0;

    // Node layout: [0, n-1) internal, [n-1, 2n-1) leaves.
    GLuint node_left_ssbo = 0;
    GLuint node_right_ssbo = 0;
    GLuint node_parent_ssbo = 0;
    GLuint node_com_ssbo = 0;
    GLuint node_aabb_ssbo = 0;
    GLuint node_flags_ssbo = 0;
    GLuint node_quad_ssbo = 0;

    static constexpr int WORKGROUP_SIZE = 256;
    static constexpr GLuint NO_PARENT = 0xFFFFFFFFu;

    void allocateBuffers();
    void freeBuffers();
    void uploadToGPU();

    // Everything between the two Verlet halves: rebuilds the tree from the
    // current positions and writes fresh accelerations.
    void buildTreeAndComputeForces();

    GLuint groupCount(int n) const {
        return static_cast<GLuint>((n + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE);
    }
};
