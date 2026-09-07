// Time per step, force error and fitted scaling exponent, swept over N.
// Run from the build directory.

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "barnes_hut_cpu.h"
#include "barnes_hut_gpu.h"
#include "brute_force_cpu.h"
#include "brute_force_gpu.h"
#include "collapse_preset.h"
#include "fmm_cpu.h"
#include "galaxy_preset.h"

namespace {

constexpr float G = 1.0f;
constexpr float SOFTENING = 1.0f;

struct Options {
    std::vector<int> sizes = {1000, 4000, 16000, 64000, 256000, 1000000};
    int reps = 7;
    int warmup = 3;
    double budget_ms = 4000.0;   // stop growing a variant past this per step
    float theta = 0.9f;
    bool quadrupole = true;
    int preset = 0;              // 0 galaxy, 1 collapse
    int error_samples = 128;
    bool csv = false;
};

std::unique_ptr<Preset> makePreset(int which, int n) {
    if (which == 1) return std::make_unique<CollapsePreset>(n);
    return std::make_unique<GalaxyPreset>(n);
}

double forceError(Simulation& sim, int samples) {
    sim.syncToHost();
    const std::vector<Particle>& ps = sim.getParticles();
    const int n = static_cast<int>(ps.size());
    if (n < 2) return 0.0;

    const int stride = std::max(1, n / std::max(1, samples));
    const double eps_sq = double(SOFTENING) * double(SOFTENING);

    double err_sq = 0.0, ref_sq = 0.0;
    for (int i = 0; i < n; i += stride) {
        const double xi = ps[i].position.x, yi = ps[i].position.y;
        double ax = 0.0, ay = 0.0;
        for (int j = 0; j < n; j++) {
            const double dx = double(ps[j].position.x) - xi;
            const double dy = double(ps[j].position.y) - yi;
            const double r2 = dx * dx + dy * dy + eps_sq;
            const double w = double(ps[j].mass) / (r2 * std::sqrt(r2));
            ax += dx * w;
            ay += dy * w;
        }
        ax *= G; ay *= G;

        const double ex = double(ps[i].acceleration.x) - ax;
        const double ey = double(ps[i].acceleration.y) - ay;
        err_sq += ex * ex + ey * ey;
        ref_sq += ax * ax + ay * ay;
    }
    return ref_sq > 0.0 ? std::sqrt(err_sq / ref_sq) : 0.0;
}

double msPerStep(Simulation& sim, const Options& o) {
    for (int i = 0; i < o.warmup; i++) {
        sim.step(0.01f);
        if (sim.isGPUResident()) glFinish();
    }

    std::vector<double> t;
    t.reserve(static_cast<size_t>(o.reps));
    for (int i = 0; i < o.reps; i++) {
        const auto a = std::chrono::high_resolution_clock::now();
        sim.step(0.01f);
        if (sim.isGPUResident()) glFinish();
        const auto b = std::chrono::high_resolution_clock::now();
        t.push_back(std::chrono::duration<double, std::milli>(b - a).count());
    }
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}

struct Variant {
    std::string label;
    std::string family;    // brute | barnes-hut | fmm
    std::string backend;   // CPU | GPU
    std::function<std::unique_ptr<Simulation>(int)> make;

    std::vector<double> ms;    // per size, -1 = not run
    std::vector<double> err;
};

// Empirical complexity exponent, fitted over the largest sizes only.
double scalingExponent(const std::vector<int>& sizes, const std::vector<double>& ms,
                       int& points_used) {
    std::vector<double> x, y;
    for (size_t i = 0; i < ms.size(); i++) {
        if (ms[i] > 0.0) { x.push_back(std::log(double(sizes[i]))); y.push_back(std::log(ms[i])); }
    }
    // Keep the top three sizes where there are enough to choose from.
    if (x.size() > 3) {
        x.erase(x.begin(), x.end() - 3);
        y.erase(y.begin(), y.end() - 3);
    }
    points_used = static_cast<int>(x.size());
    if (x.size() < 2) return 0.0;

    double mx = 0.0, my = 0.0;
    for (size_t i = 0; i < x.size(); i++) { mx += x[i]; my += y[i]; }
    mx /= double(x.size());
    my /= double(y.size());

    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < x.size(); i++) {
        num += (x[i] - mx) * (y[i] - my);
        den += (x[i] - mx) * (x[i] - mx);
    }
    return den > 0.0 ? num / den : 0.0;
}

void printRow(const char* label, const std::vector<double>& v, const char* fmt) {
    std::printf("  %-30s", label);
    for (double d : v) {
        if (d < 0.0) std::printf("%12s", "-");
        else std::printf(fmt, d);
    }
    std::printf("\n");
}

}  // namespace

int main(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        auto next = [&]() { return (i + 1 < argc) ? argv[++i] : nullptr; };
        if (a == "--max-n") {
            if (const char* v = next()) {
                const int cap = std::atoi(v);
                std::vector<int> keep;
                for (int n : o.sizes) if (n <= cap) keep.push_back(n);
                if (!keep.empty()) o.sizes = keep;
            }
        } else if (a == "--reps") { if (const char* v = next()) o.reps = std::max(1, std::atoi(v)); }
        else if (a == "--budget") { if (const char* v = next()) o.budget_ms = std::atof(v); }
        else if (a == "--theta") { if (const char* v = next()) o.theta = float(std::atof(v)); }
        else if (a == "--monopole") { o.quadrupole = false; }
        else if (a == "--collapse") { o.preset = 1; }
        else if (a == "--csv") { o.csv = true; }
        else if (a == "--help") {
            std::printf("usage: benchmark [--max-n N] [--reps R] [--budget MS] [--theta T]\n"
                        "                 [--monopole] [--collapse] [--csv]\n");
            return 0;
        }
    }

    if (!glfwInit()) { std::fprintf(stderr, "glfwInit failed\n"); return 1; }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    GLFWwindow* window = glfwCreateWindow(64, 64, "benchmark", nullptr, nullptr);
    if (!window) { std::fprintf(stderr, "window creation failed\n"); return 1; }
    glfwMakeContextCurrent(window);
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) { std::fprintf(stderr, "glewInit failed\n"); return 1; }

    const float theta = o.theta;
    const bool quad = o.quadrupole;
    const int preset = o.preset;

    std::vector<Variant> variants;
    variants.push_back({"brute force CPU", "brute", "CPU", [=](int n) {
        return std::unique_ptr<Simulation>(
            new BruteForceCPU(makePreset(preset, n), G, SOFTENING)); }, {}, {}});
    variants.push_back({"brute force GPU", "brute", "GPU", [=](int n) {
        return std::unique_ptr<Simulation>(
            new BruteForceGPU(makePreset(preset, n), G, SOFTENING)); }, {}, {}});
    variants.push_back({"Barnes-Hut CPU", "barnes-hut", "CPU", [=](int n) {
        auto s = std::make_unique<BarnesHutCPU>(makePreset(preset, n), theta, G, SOFTENING);
        s->setQuadrupole(quad);
        return std::unique_ptr<Simulation>(std::move(s)); }, {}, {}});
    variants.push_back({"Barnes-Hut GPU", "barnes-hut", "GPU", [=](int n) {
        auto s = std::make_unique<BarnesHutGPU>(makePreset(preset, n), theta, G, SOFTENING);
        s->setQuadrupole(quad);
        return std::unique_ptr<Simulation>(std::move(s)); }, {}, {}});
    variants.push_back({"FMM CPU", "fmm", "CPU", [=](int n) {
        auto s = std::make_unique<FMMCPU>(makePreset(preset, n), FMMKernel::Softened,
                                          4, 32, G, SOFTENING);
        s->setNearFieldRadius(2);
        return std::unique_ptr<Simulation>(std::move(s)); }, {}, {}});

    std::printf("N-body solver benchmark\n");
    std::printf("  GPU          %s\n", glGetString(GL_RENDERER));
    std::printf("  CPU threads  %u\n", std::thread::hardware_concurrency());
    std::printf("  preset       %s\n", preset == 1 ? "Collapse" : "Galaxy");
    std::printf("  Barnes-Hut   theta %.2f, %s\n", theta, quad ? "quadrupole" : "monopole only");
    std::printf("  timing       median of %d steps after %d warmup, GPU synced\n", o.reps, o.warmup);
    std::printf("  cutoff       a variant stops once a step exceeds %.0f ms\n\n", o.budget_ms);

    for (Variant& v : variants) {
        v.ms.assign(o.sizes.size(), -1.0);
        v.err.assign(o.sizes.size(), -1.0);
        bool over_budget = false;

        for (size_t i = 0; i < o.sizes.size(); i++) {
            if (over_budget) continue;
            const int n = o.sizes[i];

            if (i >= 1 && v.ms[i - 1] > 0.0) {
                double k = 2.0;   // assume quadratic until there are two points
                if (i >= 2 && v.ms[i - 2] > 0.0) {
                    k = std::log(v.ms[i - 1] / v.ms[i - 2])
                      / std::log(double(o.sizes[i - 1]) / double(o.sizes[i - 2]));
                    k = std::max(k, 0.5);
                }
                const double predicted =
                    v.ms[i - 1] * std::pow(double(n) / double(o.sizes[i - 1]), k);
                if (predicted > o.budget_ms) { over_budget = true; continue; }
            }

            std::fprintf(stderr, "  ... %s at N=%d\r", v.label.c_str(), n);
            std::unique_ptr<Simulation> sim = v.make(n);
            v.ms[i] = msPerStep(*sim, o);
            v.err[i] = forceError(*sim, o.error_samples);

            if (v.ms[i] > o.budget_ms) over_budget = true;
        }
    }
    std::fprintf(stderr, "%-60s\r", "");

    std::printf("Milliseconds per step\n  %-30s", "");
    for (int n : o.sizes) std::printf("%12d", n);
    std::printf("\n");
    for (const Variant& v : variants) printRow(v.label.c_str(), v.ms, "%12.2f");

    std::printf("\nForce error (rms, normalised by rms |a|)\n  %-30s", "");
    for (int n : o.sizes) std::printf("%12d", n);
    std::printf("\n");
    for (const Variant& v : variants) printRow(v.label.c_str(), v.err, "%12.2e");

    // --- speedups -------------------------------------------------------
    auto ratio = [&](const Variant& num, const Variant& den) {
        std::vector<double> r(o.sizes.size(), -1.0);
        for (size_t i = 0; i < r.size(); i++)
            if (num.ms[i] > 0.0 && den.ms[i] > 0.0) r[i] = num.ms[i] / den.ms[i];
        return r;
    };

    const Variant& bf_cpu = variants[0];
    const Variant& bf_gpu = variants[1];
    const Variant& bh_cpu = variants[2];
    const Variant& bh_gpu = variants[3];

    std::printf("\nSpeedup (x faster; blank where the slower side was skipped)\n  %-30s", "");
    for (int n : o.sizes) std::printf("%12d", n);
    std::printf("\n");
    printRow("algo: BH vs brute CPU", ratio(bf_cpu, bh_cpu), "%12.1f");
    printRow("algo: BH vs brute GPU", ratio(bf_gpu, bh_gpu), "%12.1f");
    printRow("backend: GPU vs CPU brute", ratio(bf_cpu, bf_gpu), "%12.1f");
    printRow("backend: GPU vs CPU BH", ratio(bh_cpu, bh_gpu), "%12.1f");
    printRow("combined: BH GPU vs brute CPU", ratio(bf_cpu, bh_gpu), "%12.1f");

    std::printf("\nMeasured scaling, time ~ N^k   (fitted over the largest sizes)\n");
    for (const Variant& v : variants) {
        int pts = 0;
        const double k = scalingExponent(o.sizes, v.ms, pts);
        if (pts < 2) { std::printf("  %-30s  too few sizes to fit\n", v.label.c_str()); continue; }

        int a = -1, b = -1;
        for (size_t i = 0; i < v.ms.size(); i++) if (v.ms[i] > 0.0) { a = b; b = int(i); }
        double top = 0.0;
        if (a >= 0 && b > a)
            top = std::log(v.ms[b] / v.ms[a])
                / std::log(double(o.sizes[b]) / double(o.sizes[a]));

        std::printf("  %-30s k = %.2f over %d sizes,  %.2f between the top two\n",
                    v.label.c_str(), k, pts, top);
    }
    std::printf("\n  Brute force is O(N^2) and a tree solver O(N log N), so the speedup\n"
                "  above is not a constant -- it grows with N. Below roughly 10k the GPU\n"
                "  variants are latency-bound rather than work-bound, which is why their\n"
                "  times barely move and the tree can even lose to brute force there.\n");

    if (o.csv) {
        std::printf("\nCSV\nvariant,family,backend,n,ms,err\n");
        for (const Variant& v : variants)
            for (size_t i = 0; i < o.sizes.size(); i++)
                if (v.ms[i] > 0.0)
                    std::printf("%s,%s,%s,%d,%.4f,%.6e\n", v.label.c_str(), v.family.c_str(),
                                v.backend.c_str(), o.sizes[i], v.ms[i], v.err[i]);
    }

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
