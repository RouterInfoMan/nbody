// Accuracy-versus-cost sweeps used to pick defaults for the tree solvers.
//
// Prints, for each configuration, the standard normalised force error next to
// the measured cost per step, so the two can be traded off directly rather
// than argued about. Run from the build directory.

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

#include "barnes_hut_cpu.h"
#include "barnes_hut_gpu.h"
#include "brute_force_cpu.h"
#include "collapse_preset.h"
#include "fmm_cpu.h"
#include "galaxy_preset.h"

namespace {

constexpr float G = 1.0f;
constexpr float SOFTENING = 1.0f;

std::unique_ptr<Preset> makePreset(int which, int n) {
    if (which == 0) return std::make_unique<GalaxyPreset>(n);
    return std::make_unique<CollapsePreset>(n);
}

double forceError(Simulation& sim, int max_targets, bool log_kernel = false) {
    sim.syncToHost();
    const std::vector<Particle>& ps = sim.getParticles();
    const int n = static_cast<int>(ps.size());
    const int stride = std::max(1, n / max_targets);
    const float eps_sq = SOFTENING * SOFTENING;

    double err_sq = 0.0, ref_sq = 0.0;
    for (int i = 0; i < n; i += stride) {
        glm::vec2 ref(0.0f);
        for (int j = 0; j < n; j++) {
            glm::vec2 d = ps[j].position - ps[i].position;
            if (log_kernel) {
                ref += d * (ps[j].mass / (glm::dot(d, d) + eps_sq));
            } else {
                float r2 = glm::dot(d, d) + eps_sq;
                float inv = 1.0f / std::sqrt(r2);
                ref += d * (ps[j].mass * inv * inv * inv);
            }
        }
        ref *= G;
        const double e = double(glm::length(ps[i].acceleration - ref));
        err_sq += e * e;
        ref_sq += double(glm::length(ref)) * double(glm::length(ref));
    }
    return ref_sq > 0.0 ? std::sqrt(err_sq / ref_sq) : 0.0;
}

double msPerStep(Simulation& sim, int steps) {
    sim.step(0.01f);
    if (sim.isGPUResident()) glFinish();

    std::vector<double> samples;
    for (int i = 0; i < steps; i++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        sim.step(0.01f);
        if (sim.isGPUResident()) glFinish();
        auto t1 = std::chrono::high_resolution_clock::now();
        samples.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

}  // namespace

int main(int argc, char** argv) {
    const int n_acc = 20000;
    const int n_time = (argc > 1) ? std::atoi(argv[1]) : 200000;

    if (!glfwInit()) return 1;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    GLFWwindow* window = glfwCreateWindow(64, 64, "sweep", nullptr, nullptr);
    if (!window) return 1;
    glfwMakeContextCurrent(window);
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) return 1;

    std::printf("accuracy at N=%d (Galaxy), cost at N=%d\n\n", n_acc, n_time);

    std::printf("Barnes-Hut GPU: monopole vs monopole+quadrupole\n");
    std::printf("  %-6s %-10s %12s %10s\n", "theta", "moments", "err/rms(a)", "ms/step");
    for (int quad = 0; quad < 2; quad++) {
        for (float theta : {0.3f, 0.5f, 0.7f, 0.9f, 1.2f}) {
            BarnesHutGPU acc(makePreset(0, n_acc), theta, G, SOFTENING);
            acc.setQuadrupole(quad != 0);
            acc.step(0.01f);
            const double err = forceError(acc, 512);

            BarnesHutGPU tim(makePreset(0, n_time), theta, G, SOFTENING);
            tim.setQuadrupole(quad != 0);
            std::printf("  %-6.1f %-10s %12.3e %10.2f\n", theta,
                        quad ? "mono+quad" : "mono", err, msPerStep(tim, 5));
        }
    }

    std::printf("\nBarnes-Hut CPU: monopole vs monopole+quadrupole\n");
    std::printf("  %-6s %-10s %12s %10s\n", "theta", "moments", "err/rms(a)", "ms/step");
    for (int quad = 0; quad < 2; quad++) {
        for (float theta : {0.3f, 0.5f, 0.7f, 0.9f, 1.2f}) {
            BarnesHutCPU acc(makePreset(0, n_acc), theta, G, SOFTENING);
            acc.setQuadrupole(quad != 0);
            acc.step(0.01f);
            const double err = forceError(acc, 512);

            BarnesHutCPU tim(makePreset(0, n_time), theta, G, SOFTENING);
            tim.setQuadrupole(quad != 0);
            std::printf("  %-6.1f %-10s %12.3e %10.2f\n", theta,
                        quad ? "mono+quad" : "mono", err, msPerStep(tim, 5));
        }
    }

    std::printf("\nFMM softened 1/r^2: order, leaf capacity, near-field radius\n");
    std::printf("  %-3s %-6s %-4s %12s %10s\n", "p", "leaf", "ws", "err/rms(a)", "ms/step");
    for (int ws : {1, 2}) {
        for (int p : {3, 4, 5, 6}) {
            for (int leaf : {8, 16, 32, 64, 128}) {
                FMMCPU acc(makePreset(0, n_acc), FMMKernel::Softened, p, leaf, G, SOFTENING);
                acc.setNearFieldRadius(ws);
                acc.step(0.01f);
                const double err = forceError(acc, 512);

                FMMCPU tim(makePreset(0, n_time), FMMKernel::Softened, p, leaf, G, SOFTENING);
                tim.setNearFieldRadius(ws);
                std::printf("  %-3d %-6d %-4d %12.3e %10.2f\n", p, leaf, ws,
                            err, msPerStep(tim, 3));
            }
        }
    }

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
