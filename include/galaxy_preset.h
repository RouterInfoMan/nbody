#pragma once
#include "preset.h"
#include <cstdint>

// Rotating disc around a central point mass.
struct GalaxyParams {
    int count = 50000;

    float inner_radius = 10.0f;
    float outer_radius = 60.0f;

    float central_mass = 10000.0f;
    float particle_mass = 0.1f;

    float density_falloff = 1.0f;

    float velocity_dispersion = 0.02f;

    float spin = 1.0f;

    float G = 1.0f;

    uint32_t seed = 42;
};

class GalaxyPreset : public Preset {
public:
    explicit GalaxyPreset(const GalaxyParams& params);
    // Convenience for callers that only care about size.
    explicit GalaxyPreset(int num_particles, float radius = 60.0f);

    void apply(std::vector<Particle>& particles) override;
    int getParticleCount() const override;
    const char* name() const override { return "Galaxy"; }

private:
    GalaxyParams params;
};
