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

    // Surface density profile, sigma(r) ~ r^-falloff, sampled over
    // [inner, outer]. 0 spreads mass evenly over area, 1 concentrates it
    // toward the centre, 2 is scale-free.
    float density_falloff = 1.0f;

    // Scatter applied to the circular speed, as a fraction of it. A perfectly
    // cold disc fragments almost immediately; a few percent of dispersion is
    // what keeps it looking like a disc.
    float velocity_dispersion = 0.02f;

    // +1 prograde, -1 retrograde. Values below 1 leave the disc
    // under-supported so it falls inward.
    float spin = 1.0f;

    // Gravitational constant the orbits are balanced against. This must match
    // the simulation's G or the disc is born out of equilibrium.
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
