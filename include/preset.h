#pragma once
#include <vector>
#include "particle.h"

class Preset {
public:
    virtual ~Preset() = default;
    virtual void apply(std::vector<Particle>& particles) = 0;
    virtual int getParticleCount() const = 0;
    virtual const char* name() const = 0;
};
