#pragma once
#include <vector>
#include <GL/glew.h>
#include "particle.h"

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

    // Pull device state back into `particles`. On CPU solvers this is a no-op;
    // on GPU solvers it stalls the pipeline, so only call it when the host
    // actually needs the data (validation, stats, switching algorithms).
    virtual void syncToHost() {}

    // Live parameter tweaks, so the UI sliders take effect without discarding
    // the running state. Solvers that do not use a given knob ignore it.
    virtual void setPhysics(float g, float softening) { (void)g; (void)softening; }
    virtual void setTheta(float theta) { (void)theta; }

protected:
    std::vector<Particle> particles;
    float current_time = 0.0f;
    int particle_count = 0;

    // Keeps particle_count valid for GPU solvers whose `particles` vector goes
    // stale between syncToHost() calls.
    void setParticleCount(int n) { particle_count = n; }
};
