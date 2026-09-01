#pragma once
#include "preset.h"
#include <cstdint>
#include <random>

// Self-gravitating blobs held up by velocity dispersion rather than rotation.
// There is no pressure in an N-body code, so random motion is what stands in
// for it: `support` near 1 makes a cloud roughly virialised and long-lived,
// and 0 drops it into free-fall collapse.
struct CloudParams {
    // Plummer scale is radius/3, truncated at radius, which gives a smooth
    // centrally concentrated blob with a definite edge rather than the hard
    // rim of a uniform disc.
    float radius = 30.0f;
    float support = 1.0f;
};

// Two clouds on a mutual orbit.
struct BinaryCloudsParams {
    int count = 40000;

    float cloud_radius = 30.0f;
    float separation = 220.0f;
    float total_mass = 20000.0f;
    // Mass of the second cloud relative to the first.
    float mass_ratio = 1.0f;

    // Fraction of the circular-orbit speed. 0 is a head-on free fall, 1 is a
    // circular orbit, and values between put the pair on an eccentric one
    // that grazes and tidally shreds before merging.
    float orbit_fraction = 1.0f;
    float support = 1.0f;

    float G = 1.0f;
    uint32_t seed = 42;
};

// Several clouds orbiting a common centre, which merge hierarchically.
struct CloudClusterParams {
    int count = 60000;
    int cloud_count = 5;

    float cloud_radius = 25.0f;
    float cluster_radius = 320.0f;
    float total_mass = 30000.0f;

    float orbit_fraction = 0.85f;
    float support = 1.0f;

    float G = 1.0f;
    uint32_t seed = 7;
};

class BinaryCloudsPreset : public Preset {
public:
    explicit BinaryCloudsPreset(const BinaryCloudsParams& params);

    void apply(std::vector<Particle>& particles) override;
    int getParticleCount() const override { return params.count; }
    const char* name() const override { return "Binary Clouds"; }

private:
    BinaryCloudsParams params;
};

class CloudClusterPreset : public Preset {
public:
    explicit CloudClusterPreset(const CloudClusterParams& params);

    void apply(std::vector<Particle>& particles) override;
    int getParticleCount() const override { return params.count; }
    const char* name() const override { return "Cloud Cluster"; }

private:
    CloudClusterParams params;
};

namespace clouds {

// Appends one Plummer cloud of `n` particles totalling `mass`, centred on
// `center` and drifting at `bulk`.
void emit(std::vector<Particle>& out, std::mt19937& gen, int n,
          glm::vec2 center, glm::vec2 bulk, float mass,
          const CloudParams& shape, float G);

}  // namespace clouds
