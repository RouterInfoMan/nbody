#pragma once
#include "preset.h"

class CollapsePreset : public Preset {
public:
    CollapsePreset(int num_particles, float radius = 100.0f);
    void apply(std::vector<Particle>& particles) override;
    int getParticleCount() const override;
    const char* name() const override { return "Collapse"; }

private:
    int num_particles;
    float radius;
};
