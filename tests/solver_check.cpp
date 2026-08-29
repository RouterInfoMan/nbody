// Correctness harness for the tree solvers.
//
// Every solver is stepped once from the same preset, then each is checked
// against a direct O(N^2) sum computed from *its own* particle array. That
// makes the check independent of particle ordering, which matters because the
// GPU Barnes-Hut and the FMM both permute their state into Morton order.
//
// Run from the build directory (shaders are loaded relative to the CWD).

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <chrono>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

#include "barnes_hut_cpu.h"
#include "barnes_hut_gpu.h"
#include "brute_force_cpu.h"
#include "brute_force_gpu.h"
#include "collapse_preset.h"
#include "fmm_cpu.h"
#include "galaxy_preset.h"
#include "gpu_radix_sort.h"
#include "camera.h"
#include "renderer.h"
#include "shader.h"

namespace {

constexpr float G = 1.0f;
constexpr float SOFTENING = 1.0f;

float active_softening = SOFTENING;

struct Error {
    // Standard N-body force-error metric: RMS of the absolute error divided by
    // the RMS acceleration of the sampled set. Per-particle relative error is
    // reported too, but it is a poor pass/fail signal -- in a collapse the
    // forces on central particles nearly cancel, so a tiny absolute error
    // there shows up as an enormous relative one.
    double rms_norm = 0.0;
    double max_rel = 0.0;
    int samples = 0;
};

// Direct sum for whichever force law the solver is supposed to reproduce.
glm::vec2 directAccel(const std::vector<Particle>& ps, int i, bool log_kernel) {
    const float eps_sq = active_softening * active_softening;
    glm::vec2 acc(0.0f);
    if (log_kernel) {
        for (size_t j = 0; j < ps.size(); j++) {
            glm::vec2 d = ps[j].position - ps[i].position;
            acc += d * (ps[j].mass / (glm::dot(d, d) + eps_sq));
        }
    } else {
        for (size_t j = 0; j < ps.size(); j++) {
            glm::vec2 d = ps[j].position - ps[i].position;
            float r2 = glm::dot(d, d) + eps_sq;
            float inv = 1.0f / std::sqrt(r2);
            acc += d * (ps[j].mass * inv * inv * inv);
        }
    }
    return acc * G;
}

Error measure(Simulation& sim, int max_targets, bool log_kernel = false) {
    sim.syncToHost();
    const std::vector<Particle>& ps = sim.getParticles();
    const int n = static_cast<int>(ps.size());
    const int stride = std::max(1, n / max_targets);

    Error e;
    double err_sq = 0.0, ref_sq = 0.0;
    for (int i = 0; i < n; i += stride) {
        const glm::vec2 ref = directAccel(ps, i, log_kernel);
        const double err = double(glm::length(ps[i].acceleration - ref));
        const double mag = double(glm::length(ref));

        err_sq += err * err;
        ref_sq += mag * mag;
        if (mag > 1e-12) e.max_rel = std::max(e.max_rel, err / mag);
        e.samples++;
    }
    e.rms_norm = ref_sq > 0.0 ? std::sqrt(err_sq / ref_sq) : 0.0;
    return e;
}

std::unique_ptr<Preset> makePreset(int which, int n) {
    if (which == 0) return std::make_unique<GalaxyPreset>(n);
    return std::make_unique<CollapsePreset>(n);
}

int failures = 0;

void report(const char* label, const Error& e, double tolerance) {
    const bool ok = e.rms_norm <= tolerance;
    if (!ok) failures++;
    std::printf("  %-32s  err/rms(a) %.3e   max rel %.2e   %s (tol %.0e)\n",
                label, e.rms_norm, e.max_rel, ok ? "PASS" : "FAIL", tolerance);
}

// ---------------------------------------------------------------------------

bool checkRadixSort(int n) {
    std::mt19937 gen(1234);
    std::uniform_int_distribution<uint32_t> dist(0, 0xFFFFFFFFu);

    std::vector<uint32_t> keys(n), values(n);
    for (int i = 0; i < n; i++) { keys[i] = dist(gen); values[i] = uint32_t(i); }

    GLuint bufs[4];
    glGenBuffers(4, bufs);
    const GLsizeiptr bytes = GLsizeiptr(n) * sizeof(uint32_t);
    for (int i = 0; i < 4; i++) {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, bufs[i]);
        glBufferData(GL_SHADER_STORAGE_BUFFER, bytes, nullptr, GL_DYNAMIC_COPY);
    }
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, bufs[0]);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, bytes, keys.data());
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, bufs[1]);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, bytes, values.data());

    {
        GpuRadixSort sorter;
        sorter.sort(bufs[0], bufs[1], bufs[2], bufs[3], n);
        glFinish();
    }

    std::vector<uint32_t> out_keys(n), out_values(n);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, bufs[0]);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, bytes, out_keys.data());
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, bufs[1]);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, bytes, out_values.data());
    glDeleteBuffers(4, bufs);

    bool ordered = std::is_sorted(out_keys.begin(), out_keys.end());

    // The payload must still name the element its key came from, and the
    // permutation must be a bijection.
    bool payload_ok = true;
    std::vector<char> seen(n, 0);
    for (int i = 0; i < n; i++) {
        uint32_t v = out_values[i];
        if (v >= uint32_t(n) || seen[v]) { payload_ok = false; break; }
        seen[v] = 1;
        if (keys[v] != out_keys[i]) { payload_ok = false; break; }
    }

    // Stability: equal keys must keep their original relative order.
    bool stable = true;
    for (int i = 1; i < n; i++)
        if (out_keys[i] == out_keys[i - 1] && out_values[i] < out_values[i - 1]) stable = false;

    const bool ok = ordered && payload_ok && stable;
    if (!ok) failures++;
    std::printf("  n = %-9d ordered %s  payload %s  stable %s   %s\n",
                n, ordered ? "y" : "N", payload_ok ? "y" : "N", stable ? "y" : "N",
                ok ? "PASS" : "FAIL");
    return ok;
}

}  // namespace

int main() {
    if (!glfwInit()) { std::fprintf(stderr, "glfwInit failed\n"); return 1; }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);

    GLFWwindow* window = glfwCreateWindow(64, 64, "check", nullptr, nullptr);
    if (!window) { std::fprintf(stderr, "window creation failed\n"); return 1; }
    glfwMakeContextCurrent(window);

    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) { std::fprintf(stderr, "glewInit failed\n"); return 1; }
    std::printf("GPU: %s\n\n", glGetString(GL_RENDERER));

    // Some shaders are only loaded on a UI path that a headless run never
    // takes, so compile every one of them explicitly.
    std::printf("Shader compilation\n");
    for (const char* path : {"shaders/brute_force.comp", "shaders/verlet_step1.comp",
                             "shaders/verlet_step2.comp", "shaders/color_compute.comp",
                             "shaders/particle_gather.comp", "shaders/bh_bounds.comp",
                             "shaders/bh_morton.comp", "shaders/bh_reorder.comp",
                             "shaders/bh_karras.comp", "shaders/bh_propagate.comp",
                             "shaders/bh_force.comp", "shaders/radix_histogram.comp",
                             "shaders/radix_scan.comp", "shaders/radix_scatter.comp"}) {
        try {
            Shader s(path);
            (void)s;
        } catch (const std::exception& e) {
            failures++;
            std::printf("  %-32s FAIL: %s\n", path, e.what());
            continue;
        }
        std::printf("  %-32s PASS\n", path);
    }
    try {
        Shader s("shaders/particle.vert", "shaders/particle.frag");
        (void)s;
        std::printf("  %-32s PASS\n", "shaders/particle.{vert,frag}");
    } catch (const std::exception& e) {
        failures++;
        std::printf("  %-32s FAIL: %s\n", "shaders/particle.{vert,frag}", e.what());
    }

    // The GPU-resident draw path binds a solver's own buffers into the
    // colouring pass; nothing else in this harness touches it.
    std::printf("\nGPU-resident render path\n");
    {
        while (glGetError() != GL_NO_ERROR) {}   // drain

        Renderer renderer;
        renderer.initialize();
        Camera2D camera(64, 64);
        BarnesHutGPU sim(makePreset(0, 20000), 0.5f, G, SOFTENING);
        sim.step(0.01f);

        for (bool colour : {true, false}) {
            renderer.setShowVelocity(colour);
            renderer.renderFromGPU(sim.positionBuffer(), sim.velocityBuffer(),
                                   sim.getParticleCount(), camera);
        }
        glFinish();

        const GLenum err = glGetError();
        if (err != GL_NO_ERROR) {
            failures++;
            std::printf("  draw from solver buffers        FAIL (GL error 0x%04x)\n", err);
        } else {
            std::printf("  draw from solver buffers        PASS\n");
        }
    }

    std::printf("\nGPU radix sort\n");
    for (int n : {1000, 2048, 5000, 100000, 400001}) checkRadixSort(n);

    for (int which = 0; which < 2; which++) {
        const char* preset_name = which == 0 ? "Galaxy" : "Collapse";
        const int n = 20000;

        std::printf("\n%s, N = %d, one step, softening = %.2f\n", preset_name, n, SOFTENING);

        {
            BruteForceCPU ref(makePreset(which, n), G, SOFTENING);
            ref.step(0.01f);
            // Sanity check on the harness: the reference must reproduce itself
            // to within float rounding.
            report("brute force CPU (self-check)", measure(ref, 512), 1e-5);
        }
        {
            BarnesHutCPU bh(makePreset(which, n), 0.5f, G, SOFTENING);
            bh.step(0.01f);
            report("Barnes-Hut CPU, theta 0.5", measure(bh, 512), 2e-2);
        }
        for (float theta : {0.5f, 0.2f}) {
            BarnesHutGPU bh(makePreset(which, n), theta, G, SOFTENING);
            bh.step(0.01f);
            char label[64];
            std::snprintf(label, sizeof(label), "Barnes-Hut GPU, theta %.1f", theta);
            report(label, measure(bh, 512), theta > 0.3f ? 2e-2 : 5e-3);
        }

        // A Cartesian Taylor FMM with one-cell separation converges at roughly
        // 0.4 per order (measured on an isolated M2L, and reproduced here).
        // The ceilings below allow for that; the more telling check is the
        // monotone decrease, since a missing or double-counted interaction
        // list would make the error plateau no matter how high p goes.
        double prev = 1e9;
        bool monotone = true;
        for (int p = 1; p <= 6; p++) {
            FMMCPU fmm(makePreset(which, n), FMMKernel::Softened, p, 32, G, SOFTENING);
            fmm.step(0.01f);
            char label[64];
            std::snprintf(label, sizeof(label), "FMM softened 1/r^2, p = %d", p);
            const Error e = measure(fmm, 512);
            report(label, e, 2.0 * std::pow(0.45, p));
            if (p > 1 && e.rms_norm > prev * 0.75) monotone = false;
            prev = e.rms_norm;
        }
        if (!monotone) {
            failures++;
            std::printf("  FMM error did not fall steadily with p -- FAIL\n");
        } else {
            std::printf("  FMM converges monotonically in p -- PASS\n");
        }

        // The logarithmic kernel is a different force law, so it is checked
        // against a direct sum of THAT law, not against 1/r^2.
        //
        // Its far field expands the bare log kernel: softening only
        // regularises the direct near field, as it does in every classic FMM.
        // Against a softened direct sum the error therefore floors out at
        // O(eps^2 / R^2) no matter how high p goes, so the translation
        // operators are exercised at a small softening where that floor is
        // far below the truncation error.
        for (float eps : {SOFTENING, 0.05f}) {
            active_softening = eps;
            std::printf("  -- log kernel, softening = %.2f\n", eps);
            for (int p : {4, 8, 12}) {
                FMMCPU fmm(makePreset(which, n), FMMKernel::Logarithmic, p, 32, G, eps);
                fmm.step(0.01f);
                char label[64];
                std::snprintf(label, sizeof(label), "FMM log kernel, p = %d", p);
                // Only the small-softening sweep is expected to keep
                // converging; the eps = 1 run is printed for the floor it hits.
                const double tol = (eps < 0.5f) ? std::pow(0.35, p) * 20.0 : 1e9;
                report(label, measure(fmm, 512, /*log_kernel=*/true), tol);
            }
        }
        active_softening = SOFTENING;
    }

    // ---- Throughput -----------------------------------------------------
    // GPU solvers are timed with an explicit finish, since nothing in the
    // step path reads back and would otherwise stall.
    std::printf("\nMilliseconds per step (Galaxy preset, median of %d after warmup)\n", 5);

    auto time_solver = [](Simulation& sim, int steps) {
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
    };

    std::printf("  %-30s %10s %10s %10s\n", "solver", "N=50k", "N=200k", "N=1M");
    for (int solver = 0; solver < 5; solver++) {
        const char* names[] = { "brute force GPU", "Barnes-Hut CPU (theta 0.5)",
                                "Barnes-Hut GPU (theta 0.5)", "FMM CPU 1/r^2 (p=4)",
                                "FMM CPU log (p=8)" };
        std::printf("  %-30s", names[solver]);

        for (int n : {50000, 200000, 1000000}) {
            // Brute force at a million particles is 10^12 interactions a step.
            if (solver == 0 && n > 200000) { std::printf(" %10s", "skipped"); continue; }

            std::unique_ptr<Simulation> sim;
            switch (solver) {
                case 0: sim = std::make_unique<BruteForceGPU>(makePreset(0, n), G, SOFTENING); break;
                case 1: sim = std::make_unique<BarnesHutCPU>(makePreset(0, n), 0.5f, G, SOFTENING); break;
                case 2: sim = std::make_unique<BarnesHutGPU>(makePreset(0, n), 0.5f, G, SOFTENING); break;
                case 3: sim = std::make_unique<FMMCPU>(makePreset(0, n), FMMKernel::Softened, 4, 32, G, SOFTENING); break;
                case 4: sim = std::make_unique<FMMCPU>(makePreset(0, n), FMMKernel::Logarithmic, 8, 32, G, SOFTENING); break;
            }
            std::printf(" %10.2f", time_solver(*sim, 5));
            std::fflush(stdout);
        }
        std::printf("\n");
    }

    std::printf("\n%s\n", failures ? "FAILURES PRESENT" : "all checks passed");
    glfwDestroyWindow(window);
    glfwTerminate();
    return failures ? 1 : 0;
}
