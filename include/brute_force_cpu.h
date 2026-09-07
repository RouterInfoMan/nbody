#pragma once
#include "simulation.h"
#include "preset.h"
#include "thread_pool.h"
#include <memory>

class BruteForceCPU : public Simulation {
public:
    BruteForceCPU(std::unique_ptr<Preset> preset, float G = 1.0f, float softening = 1.0f);

    void step(float dt) override;
    void reset() override;

    bool energyTotals(double& kinetic, double& potential) override {
        return hostEnergyTotals(kinetic, potential);
    }
    const char* name() const override { return "Brute Force (CPU)"; }

private:
    std::unique_ptr<Preset> preset;
    ThreadPool pool;
    float G;
    float softening_sq;

    void computeForces();
    void verletStep1(float dt);
    void verletStep2(float dt);
};
