#pragma once
#include "preset.h"

class GalaxyPreset : public Preset {
public:
    GalaxyPreset(int num_particles, float radius = 60.0f);
    void apply(std::vector<Particle>& particles) override;
    int getParticleCount() const override;
    const char* name() const override { return "Galaxy"; }

private:
    int num_particles;
    float radius;
};
