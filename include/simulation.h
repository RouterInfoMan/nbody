#pragma once
#include <vector>
#include <GL/glew.h>
#include "particle.h"
#include "boundary.h"

class Simulation {
public:
    virtual ~Simulation() = default;
    virtual void step(float dt) = 0;
    virtual void reset() = 0;
    virtual const char* name() const = 0;

    const std::vector<Particle>& getParticles() const { return particles; }
    int getParticleCount() const { return particle_count; }
    float getTime() const { return current_time; }

    // GPU-resident solvers keep state in device buffers and never round-trip
    // through `particles` during stepping. The renderer binds these directly.
    virtual bool isGPUResident() const { return false; }
    virtual GLuint positionBuffer() const { return 0; }
    virtual GLuint velocityBuffer() const { return 0; }
    virtual GLuint massBuffer() const { return 0; }
    // Stable per-particle identity, for solvers that permute their state.
    // Zero means array position is already stable and can be used directly.
    virtual GLuint idBuffer() const { return 0; }

    // Pull device state back into `particles`. On CPU solvers this is a no-op;
    // on GPU solvers it stalls the pipeline, so only call it when the host
    // actually needs the data (validation, stats, switching algorithms).
    virtual void syncToHost() {}

    // Fills `buffer` and `count` with the solver's acceleration-structure
    // boxes for the debug overlay: vec4(min.xy, max.xy), index 0 the root.
    // Not const because a host-side solver has to upload first. Returns false
    // if the solver has no structure to show.
    virtual bool treeGeometry(GLuint& buffer, int& count) {
        (void)buffer; (void)count; return false;
    }
    virtual const char* treeKind() const { return ""; }

    // Containment applied after each step. Lives on the base class because it
    // is orthogonal to the solver: every algorithm and preset can use it.
    void setBoundary(const Boundary& b) { boundary = b; }
    const Boundary& getBoundary() const { return boundary; }

    // Live parameter tweaks, so the UI sliders take effect without discarding
    // the running state. Solvers that do not use a given knob ignore it.
    virtual void setPhysics(float g, float softening) { (void)g; (void)softening; }
    virtual void setTheta(float theta) { (void)theta; }

protected:
    // Applies the boundary to particles [begin, end). CPU solvers call this
    // through their own thread pool at the end of step().
    void applyBoundaryRange(int begin, int end, float dt) {
        if (!boundary.active()) return;
        for (int i = begin; i < end; i++)
            applyBoundary(particles[i].position, particles[i].velocity, boundary, dt);
    }

    Boundary boundary;
    std::vector<Particle> particles;
    float current_time = 0.0f;
    int particle_count = 0;

    // Keeps particle_count valid for GPU solvers whose `particles` vector goes
    // stale between syncToHost() calls.
    void setParticleCount(int n) { particle_count = n; }
};
