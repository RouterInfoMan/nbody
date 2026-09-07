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

    virtual bool isGPUResident() const { return false; }
    virtual GLuint positionBuffer() const { return 0; }
    virtual GLuint velocityBuffer() const { return 0; }
    virtual GLuint massBuffer() const { return 0; }
    // Stable id: array position is permuted every step.
    virtual GLuint idBuffer() const { return 0; }

    // Copies device state into `particles`. Stalls the pipeline on GPU solvers.
    virtual void syncToHost() {}

    virtual bool treeGeometry(GLuint& buffer, int& count) {
        (void)buffer; (void)count; return false;
    }
    virtual const char* treeKind() const { return ""; }

    void setBoundary(const Boundary& b) { boundary = b; }
    const Boundary& getBoundary() const { return boundary; }

    virtual bool energyTotals(double& kinetic, double& potential) {
        (void)kinetic; (void)potential; return false;
    }

    virtual void setPhysics(float g, float softening) { (void)g; (void)softening; }
    virtual void setTheta(float theta) { (void)theta; }

protected:
    bool hostEnergyTotals(double& kinetic, double& potential) const {
        if (particles.empty()) return false;
        double k = 0.0, u = 0.0;
        for (const Particle& p : particles) {
            k += 0.5 * double(p.mass) * double(glm::dot(p.velocity, p.velocity));
            u += 0.5 * double(p.mass) * double(p.potential);
        }
        kinetic = k;
        potential = u;
        return true;
    }

    void applyBoundaryRange(int begin, int end, float dt) {
        if (!boundary.active()) return;
        for (int i = begin; i < end; i++)
            applyBoundary(particles[i].position, particles[i].velocity, boundary, dt);
    }

    Boundary boundary;
    std::vector<Particle> particles;
    float current_time = 0.0f;
    int particle_count = 0;

    void setParticleCount(int n) { particle_count = n; }
};
