#pragma once
#include "preset.h"
#include <cstdint>

// Uniform disc of particles, optionally spinning, released to collapse.
struct CollapseParams {
    int count = 20000;

    float radius = 100.0f;
    float particle_mass = 1.0f;
    float central_mass = 0.0f;

    // Tangential speed as a fraction of what circular support would need.
    // 0 is a cold radial collapse; near 1 the cloud holds itself up and
    // settles into a rotating disc instead of a point.
    float rotation = 0.0f;

    // Random motion as a fraction of the local circular speed. Pressure
    // support, in effect: it resists collapse without adding net rotation.
    float velocity_dispersion = 0.0f;

    // Must match the simulation's G for `rotation` to mean what it says.
    float G = 1.0f;

    uint32_t seed = 42;
};

class CollapsePreset : public Preset {
public:
    explicit CollapsePreset(const CollapseParams& params);
    explicit CollapsePreset(int num_particles, float radius = 100.0f);

    void apply(std::vector<Particle>& particles) override;
    int getParticleCount() const override;
    const char* name() const override { return "Collapse"; }

private:
    CollapseParams params;
};
