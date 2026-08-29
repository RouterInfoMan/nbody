#pragma once
#include <vector>
#include "particle.h"

class Simulation {
public:
    virtual ~Simulation() = default;
    virtual void step(float dt) = 0;
    virtual void reset() = 0;
    virtual const char* name() const = 0;

    const std::vector<Particle>& getParticles() const { return particles; }
    int getParticleCount() const { return static_cast<int>(particles.size()); }
    float getTime() const { return current_time; }

protected:
    std::vector<Particle> particles;
    float current_time = 0.0f;
};
