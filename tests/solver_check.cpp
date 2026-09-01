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
#include "boundary.h"
#include "cloud_presets.h"
#include "gpu_boundary.h"
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
//
// Accumulated in double even though the solvers work in float. A galaxy's
// disc contributions largely cancel, so a float reference carries its own
// summation error of the same order as the tree error being measured, and the
// two are indistinguishable. Double costs nothing here and makes the reference
// exact for this purpose.
glm::vec2 directAccel(const std::vector<Particle>& ps, int i, bool log_kernel) {
    const double eps_sq = double(active_softening) * double(active_softening);
    const double xi = ps[i].position.x, yi = ps[i].position.y;

    double ax = 0.0, ay = 0.0;
    for (size_t j = 0; j < ps.size(); j++) {
        const double dx = double(ps[j].position.x) - xi;
        const double dy = double(ps[j].position.y) - yi;
        const double r2 = dx * dx + dy * dy + eps_sq;
        const double w = log_kernel ? (double(ps[j].mass) / r2)
                                    : (double(ps[j].mass) / (r2 * std::sqrt(r2)));
        ax += dx * w;
        ay += dy * w;
    }
    return glm::vec2(float(ax * G), float(ay * G));
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
                             "shaders/radix_scan.comp", "shaders/radix_scatter.comp",
                             "shaders/tree_collect.comp", "shaders/boundary.comp"}) {
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
    const char* pairs[][2] = {
        {"shaders/particle.vert",   "shaders/particle.frag"},
        {"shaders/fullscreen.vert", "shaders/bloom_down.frag"},
        {"shaders/fullscreen.vert", "shaders/bloom_up.frag"},
        {"shaders/fullscreen.vert", "shaders/composite.frag"},
        {"shaders/tree_lines.vert", "shaders/tree_lines.frag"},
        {"shaders/boundary_line.vert", "shaders/boundary_line.frag"},
    };
    for (auto& pr : pairs) {
        try {
            Shader s(pr[0], pr[1]);
            (void)s;
            std::printf("  %-32s PASS\n", pr[1]);
        } catch (const std::exception& e) {
            failures++;
            std::printf("  %-32s FAIL: %s\n", pr[1], e.what());
        }
    }

    // The GPU-resident draw path binds a solver's own buffers into the
    // colouring pass; nothing else in this harness touches it.
    std::printf("\nGPU-resident render path\n");
    {
        while (glGetError() != GL_NO_ERROR) {}   // drain

        Renderer renderer;
        renderer.initialize(64, 64);
        Camera2D camera(64, 64);
        BarnesHutGPU sim(makePreset(0, 20000), 0.5f, G, SOFTENING);
        sim.step(0.01f);

        // Exercise every colouring path and both bloom states.
        for (int mode = 0; mode < COLOR_MODE_COUNT; mode++) {
            renderer.setColorMode(mode);
            for (bool bloom : {true, false}) {
                renderer.setBloomEnabled(bloom);
                renderer.renderFromGPU(sim.positionBuffer(), sim.velocityBuffer(),
                                       sim.massBuffer(), sim.idBuffer(),
                                       sim.getParticleCount(), camera);
            }
        }
        // And the host-side path, which uploads rather than binding solver buffers.
        BarnesHutCPU cpu(makePreset(0, 5000), 0.9f, G, SOFTENING);
        cpu.step(0.01f);
        renderer.render(cpu.getParticles(), camera, true);
        glFinish();

        const GLenum err = glGetError();
        if (err != GL_NO_ERROR) {
            failures++;
            std::printf("  HDR draw paths                  FAIL (GL error 0x%04x)\n", err);
        } else {
            std::printf("  HDR draw paths                  PASS\n");
        }
    }

    // ---- Boundary -------------------------------------------------------
    // The containment rule exists twice: once in C++ for the CPU solvers and
    // once in GLSL for the GPU ones. Comparing them on identical input catches
    // any drift between the two directly, without the chaotic divergence that
    // makes a solver-level comparison unusable after a few steps.
    std::printf("\nBoundary CPU/GPU parity\n");
    {
        const int n = 20000;
        std::mt19937 gen(99);
        std::uniform_real_distribution<float> place(-900.0f, 900.0f);
        std::uniform_real_distribution<float> speed(-60.0f, 60.0f);

        std::vector<glm::vec2> pos0(n), vel0(n);
        for (int i = 0; i < n; i++) {
            pos0[i] = glm::vec2(place(gen), place(gen));   // spans both sides of the wall
            vel0[i] = glm::vec2(speed(gen), speed(gen));
        }

        GLuint bufs[2];
        glGenBuffers(2, bufs);

        for (int shape = 0; shape < BOUNDARY_SHAPE_COUNT; shape++) {
            for (int mode = BOUNDARY_BOUNCE; mode < BOUNDARY_MODE_COUNT; mode++) {
                Boundary b;
                b.mode = mode;
                b.shape = shape;
                b.radius = 500.0f;
                b.restitution = 0.4f;
                b.drag = 6.0f;
                const float dt = 0.01f;

                std::vector<Particle> host(n);
                for (int i = 0; i < n; i++) { host[i].position = pos0[i]; host[i].velocity = vel0[i]; }
                for (int i = 0; i < n; i++)
                    applyBoundary(host[i].position, host[i].velocity, b, dt);

                std::vector<glm::vec2> gp = pos0, gv = vel0;
                const GLsizeiptr bytes = GLsizeiptr(n) * sizeof(glm::vec2);
                for (int k = 0; k < 2; k++) {
                    glBindBuffer(GL_SHADER_STORAGE_BUFFER, bufs[k]);
                    glBufferData(GL_SHADER_STORAGE_BUFFER, bytes,
                                 k == 0 ? gp.data() : gv.data(), GL_DYNAMIC_COPY);
                }
                { GpuBoundary gb; gb.apply(bufs[0], bufs[1], n, b, dt); glFinish(); }

                glBindBuffer(GL_SHADER_STORAGE_BUFFER, bufs[0]);
                glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, bytes, gp.data());
                glBindBuffer(GL_SHADER_STORAGE_BUFFER, bufs[1]);
                glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, bytes, gv.data());

                double worst = 0.0;
                int contained = 0;
                for (int i = 0; i < n; i++) {
                    worst = std::max(worst, double(glm::length(host[i].position - gp[i])));
                    worst = std::max(worst, double(glm::length(host[i].velocity - gv[i])));

                    const float reach = (shape == BOUNDARY_CIRCLE)
                        ? glm::length(gp[i])
                        : std::max(std::abs(gp[i].x), std::abs(gp[i].y));
                    if (reach <= b.radius * 1.0001f) contained++;
                }

                const bool parity = worst < 1e-3;
                // Bounce clamps to the wall, so nothing may remain outside it.
                // Drag only damps, so containment is not expected.
                const bool ok = parity && (mode != BOUNDARY_BOUNCE || contained == n);
                if (!ok) failures++;

                std::printf("  %-6s %-6s  max |cpu - gpu| %.2e  inside %d/%d   %s\n",
                            shape == BOUNDARY_CIRCLE ? "circle" : "box",
                            mode == BOUNDARY_BOUNCE ? "bounce" : "drag",
                            worst, contained, n, ok ? "PASS" : "FAIL");
            }
        }
        glDeleteBuffers(2, bufs);
    }

    // ---- Presets --------------------------------------------------------
    std::printf("\nPreset parameters\n");
    {
        auto check = [](const char* what, bool ok) {
            if (!ok) failures++;
            std::printf("  %-46s %s\n", what, ok ? "PASS" : "FAIL");
        };

        GalaxyParams g;
        g.count = 20000;
        g.inner_radius = 25.0f;
        g.outer_radius = 90.0f;
        g.central_mass = 5000.0f;
        g.particle_mass = 0.25f;
        g.G = 2.0f;

        std::vector<Particle> p;
        GalaxyPreset(g).apply(p);

        check("galaxy: count matches getParticleCount()",
              int(p.size()) == GalaxyPreset(g).getParticleCount());
        check("galaxy: central mass present", p[0].mass == g.central_mass);

        bool radii_ok = true, mass_ok = true;
        double lz = 0.0;
        for (size_t i = 1; i < p.size(); i++) {
            const float r = glm::length(p[i].position);
            // A hair of slack for float rounding at the interval ends.
            if (r < g.inner_radius * 0.999f || r > g.outer_radius * 1.001f) radii_ok = false;
            if (p[i].mass != g.particle_mass) mass_ok = false;
            lz += double(p[i].position.x * p[i].velocity.y - p[i].position.y * p[i].velocity.x);
        }
        check("galaxy: every particle within [inner, outer]", radii_ok);
        check("galaxy: particle mass applied", mass_ok);
        check("galaxy: prograde spin gives positive angular momentum", lz > 0.0);

        // Orbits must be balanced against the G they were built with, and
        // against the enclosed disc mass, not the central mass alone.
        {
            size_t outermost = 1;
            for (size_t i = 1; i < p.size(); i++)
                if (glm::length(p[i].position) > glm::length(p[outermost].position)) outermost = i;
            const float r = glm::length(p[outermost].position);
            const float v = glm::length(p[outermost].velocity);
            const float enclosed = g.central_mass + g.particle_mass * float(g.count);
            const float expect = std::sqrt(g.G * enclosed / r);
            check("galaxy: outer speed balances G and enclosed disc mass",
                  std::abs(v - expect) < expect * 0.15f);
        }

        GalaxyParams g2 = g;
        g2.spin = -1.0f;
        std::vector<Particle> p2;
        GalaxyPreset(g2).apply(p2);
        double lz2 = 0.0;
        for (size_t i = 1; i < p2.size(); i++)
            lz2 += double(p2[i].position.x * p2[i].velocity.y - p2[i].position.y * p2[i].velocity.x);
        check("galaxy: retrograde spin flips angular momentum", lz2 < 0.0);

        GalaxyParams g3 = g;
        g3.seed = 777;
        std::vector<Particle> p3;
        GalaxyPreset(g3).apply(p3);
        bool differs = false;
        for (size_t i = 1; i < p.size() && !differs; i++)
            if (p[i].position != p3[i].position) differs = true;
        check("galaxy: a different seed gives a different realisation", differs);

        std::vector<Particle> p4;
        GalaxyPreset(g).apply(p4);
        bool identical = p4.size() == p.size();
        for (size_t i = 0; i < p.size() && identical; i++)
            if (p4[i].position != p[i].position) identical = false;
        check("galaxy: the same seed reproduces exactly", identical);

        // Falloff must actually redistribute mass inward.
        auto medianRadius = [](float falloff) {
            GalaxyParams gp;
            gp.count = 20000;
            gp.inner_radius = 25.0f;
            gp.outer_radius = 90.0f;
            gp.density_falloff = falloff;
            std::vector<Particle> v;
            GalaxyPreset(gp).apply(v);
            std::vector<float> radii;
            for (size_t i = 1; i < v.size(); i++) radii.push_back(glm::length(v[i].position));
            std::sort(radii.begin(), radii.end());
            return radii[radii.size() / 2];
        };
        check("galaxy: higher falloff concentrates mass inward",
              medianRadius(2.0f) < medianRadius(0.0f));

        CollapseParams c;
        c.count = 20000;
        c.radius = 40.0f;
        c.particle_mass = 2.0f;
        c.rotation = 0.0f;

        std::vector<Particle> q;
        CollapsePreset(c).apply(q);
        bool cold = true, inside = true;
        for (const Particle& part : q) {
            if (glm::length(part.velocity) != 0.0f) cold = false;
            if (glm::length(part.position) > c.radius * 1.001f) inside = false;
        }
        check("collapse: rotation 0 starts at rest", cold);
        check("collapse: every particle within radius", inside);

        c.rotation = 1.0f;
        std::vector<Particle> q2;
        CollapsePreset(c).apply(q2);
        double lz3 = 0.0;
        for (const Particle& part : q2)
            lz3 += double(part.position.x * part.velocity.y - part.position.y * part.velocity.x);
        check("collapse: rotation adds net angular momentum", lz3 > 0.0);

        // --- clouds ---
        BinaryCloudsParams b;
        b.count = 20000;
        b.cloud_radius = 20.0f;
        b.separation = 300.0f;
        b.total_mass = 8000.0f;
        b.mass_ratio = 3.0f;
        b.support = 0.0f;          // isolates the orbital motion exactly

        std::vector<Particle> bc;
        BinaryCloudsPreset(b).apply(bc);

        check("binary: count matches getParticleCount()",
              int(bc.size()) == BinaryCloudsPreset(b).getParticleCount());

        double mass_sum = 0.0, px = 0.0, py = 0.0, cx = 0.0, cy = 0.0;
        for (const Particle& part : bc) {
            mass_sum += double(part.mass);
            px += double(part.mass) * double(part.velocity.x);
            py += double(part.mass) * double(part.velocity.y);
            cx += double(part.mass) * double(part.position.x);
            cy += double(part.mass) * double(part.position.y);
        }
        check("binary: total mass matches", std::abs(mass_sum - b.total_mass) < b.total_mass * 1e-3);
        // Both are exact by construction: the clouds are placed and boosted
        // with barycentric weights, so any drift means the weighting is wrong.
        check("binary: barycentre at the origin",
              std::hypot(cx, cy) < mass_sum * b.separation * 1e-3);
        check("binary: zero net momentum",
              std::hypot(px, py) < mass_sum * 1e-3);

        // Uniform particle mass across both clouds keeps relaxation even.
        bool uniform_mass = true;
        for (const Particle& part : bc)
            if (std::abs(part.mass - bc[0].mass) > bc[0].mass * 1e-4f) uniform_mass = false;
        check("binary: particle mass uniform across both clouds", uniform_mass);

        // Every particle should sit within its cloud's truncation radius.
        const float m1 = b.total_mass / (1.0f + b.mass_ratio);
        const float m2 = b.total_mass - m1;
        const glm::vec2 k1(-b.separation * (m2 / b.total_mass), 0.0f);
        const glm::vec2 k2( b.separation * (m1 / b.total_mass), 0.0f);
        bool contained = true;
        for (const Particle& part : bc) {
            const float d = std::min(glm::length(part.position - k1),
                                     glm::length(part.position - k2));
            if (d > b.cloud_radius * 1.001f) contained = false;
        }
        check("binary: particles inside their cloud's truncation radius", contained);

        b.orbit_fraction = 0.0f;
        std::vector<Particle> bc0;
        BinaryCloudsPreset(b).apply(bc0);
        bool at_rest = true;
        for (const Particle& part : bc0)
            if (glm::length(part.velocity) > 1e-6f) at_rest = false;
        check("binary: orbit 0 with no support is a head-on free fall", at_rest);

        CloudClusterParams cl;
        cl.count = 30000;
        cl.cloud_count = 6;
        cl.total_mass = 12000.0f;
        cl.support = 0.0f;

        std::vector<Particle> cc;
        CloudClusterPreset(cl).apply(cc);
        check("cluster: count matches getParticleCount()",
              int(cc.size()) == CloudClusterPreset(cl).getParticleCount());

        double cluster_mass = 0.0;
        for (const Particle& part : cc) cluster_mass += double(part.mass);
        check("cluster: total mass matches",
              std::abs(cluster_mass - cl.total_mass) < cl.total_mass * 1e-3);

        // That cloud_count distinct, separated clouds were actually emitted.
        // Comparing individual particles would not show this: the first cloud
        // is drawn from the same RNG state whatever the count, so particle 0
        // is legitimately identical. This walks the per-cloud blocks instead
        // and checks their centroids are all further apart than a cloud is
        // wide -- which does mean knowing the block layout.
        {
            std::vector<glm::vec2> centroids;
            int offset = 0;
            for (int i = 0; i < cl.cloud_count; i++) {
                const int n = cl.count / cl.cloud_count
                            + (i < cl.count % cl.cloud_count ? 1 : 0);
                glm::vec2 sum(0.0f);
                for (int j = offset; j < offset + n; j++) sum += cc[j].position;
                centroids.push_back(sum / float(n));
                offset += n;
            }

            bool covered = offset == int(cc.size());
            bool separated = true;
            for (size_t i = 0; i < centroids.size(); i++)
                for (size_t j = i + 1; j < centroids.size(); j++)
                    if (glm::length(centroids[i] - centroids[j]) < cl.cloud_radius)
                        separated = false;

            check("cluster: blocks cover every particle", covered);
            check("cluster: emits that many separated clouds",
                  separated && int(centroids.size()) == cl.cloud_count);
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
        // Galaxy carries a 10000-mass black hole that dominates every
        // particle's acceleration and, sitting alone in its own leaf, is
        // summed exactly. That flatters relative error by roughly 20x.
        // Collapse -- uniform, equal masses, no dominant term -- is the
        // honest test of tree accuracy, so its ceilings are scaled to match.
        const double tol_scale = (which == 0) ? 1.0 : 20.0;

        double bh_err[2][2][2] = {};   // [gpu][quad][theta index]
        const float thetas[2] = { 0.5f, 0.9f };

        for (int gpu = 0; gpu < 2; gpu++) {
            for (int quad = 0; quad < 2; quad++) {
                for (int ti = 0; ti < 2; ti++) {
                    const float theta = thetas[ti];
                    std::unique_ptr<Simulation> sim;
                    if (gpu) {
                        auto b = std::make_unique<BarnesHutGPU>(makePreset(which, n), theta, G, SOFTENING);
                        b->setQuadrupole(quad != 0);
                        sim = std::move(b);
                    } else {
                        auto b = std::make_unique<BarnesHutCPU>(makePreset(which, n), theta, G, SOFTENING);
                        b->setQuadrupole(quad != 0);
                        sim = std::move(b);
                    }
                    sim->step(0.01f);

                    char label[80];
                    std::snprintf(label, sizeof(label), "Barnes-Hut %s theta %.1f %s",
                                  gpu ? "GPU" : "CPU", theta, quad ? "mono+quad" : "mono");
                    const double base = quad ? (theta > 0.7f ? 5e-3 : 5e-4)
                                             : (theta > 0.7f ? 3e-2 : 5e-3);
                    const Error e = measure(*sim, 512);
                    report(label, e, base * tol_scale);
                    bh_err[gpu][quad][ti] = e.rms_norm;
                }
            }
        }

        // The invariant that matters and does not depend on the preset:
        // carrying second moments must buy real accuracy at equal theta.
        for (int gpu = 0; gpu < 2; gpu++) {
            for (int ti = 0; ti < 2; ti++) {
                const double ratio = bh_err[gpu][1][ti] / std::max(bh_err[gpu][0][ti], 1e-30);
                const bool ok = ratio < 0.5;
                if (!ok) failures++;
                std::printf("  quadrupole gain, %s theta %.1f: %.1fx   %s\n",
                            gpu ? "GPU" : "CPU", thetas[ti], 1.0 / ratio,
                            ok ? "PASS" : "FAIL");
            }
        }

        // A Cartesian Taylor FMM with one-cell separation converges at roughly
        // 0.4 per order (measured on an isolated M2L, and reproduced here).
        // The ceilings below allow for that; the more telling check is the
        // monotone decrease, since a missing or double-counted interaction
        // list would make the error plateau no matter how high p goes.
        for (int ws = 1; ws <= 2; ws++) {
            std::printf("  -- near-field radius %d\n", ws);
            double prev = 1e9;
            bool monotone = true;
            for (int p = 1; p <= 6; p++) {
                FMMCPU fmm(makePreset(which, n), FMMKernel::Softened, p, 32, G, SOFTENING);
                fmm.setNearFieldRadius(ws);
                fmm.step(0.01f);
                char label[64];
                std::snprintf(label, sizeof(label), "FMM softened 1/r^2, p = %d", p);
                const Error e = measure(fmm, 512);
                // Convergence ratio per order is ~0.42 at ws=1 and ~0.28 at
                // ws=2, measured on an isolated M2L; these are loose ceilings.
                report(label, e, 2.0 * std::pow(ws == 1 ? 0.45 : 0.32, p));
                (void)tol_scale;
                if (p > 1 && e.rms_norm > prev * 0.75) monotone = false;
                prev = e.rms_norm;
            }
            if (!monotone) {
                failures++;
                std::printf("  FMM error did not fall steadily with p -- FAIL\n");
            } else {
                std::printf("  FMM converges monotonically in p -- PASS\n");
            }
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

    const char* names[] = {
        "brute force GPU (exact)",
        "Barnes-Hut CPU mono th 0.5",
        "Barnes-Hut CPU quad th 0.9",
        "Barnes-Hut GPU mono th 0.5",
        "Barnes-Hut GPU quad th 0.9",
        "FMM CPU 1/r^2 p=4 ws=2",
        "FMM CPU log p=8 ws=2",
    };
    auto build = [](int solver, int n) -> std::unique_ptr<Simulation> {
        switch (solver) {
            case 0: return std::make_unique<BruteForceGPU>(makePreset(0, n), G, SOFTENING);
            case 1: { auto b = std::make_unique<BarnesHutCPU>(makePreset(0, n), 0.5f, G, SOFTENING);
                      b->setQuadrupole(false); return b; }
            case 2: { auto b = std::make_unique<BarnesHutCPU>(makePreset(0, n), 0.9f, G, SOFTENING);
                      b->setQuadrupole(true); return b; }
            case 3: { auto b = std::make_unique<BarnesHutGPU>(makePreset(0, n), 0.5f, G, SOFTENING);
                      b->setQuadrupole(false); return b; }
            case 4: { auto b = std::make_unique<BarnesHutGPU>(makePreset(0, n), 0.9f, G, SOFTENING);
                      b->setQuadrupole(true); return b; }
            case 5: { auto f = std::make_unique<FMMCPU>(makePreset(0, n), FMMKernel::Softened, 4, 32, G, SOFTENING);
                      f->setNearFieldRadius(2); return f; }
            default: { auto f = std::make_unique<FMMCPU>(makePreset(0, n), FMMKernel::Logarithmic, 8, 32, G, SOFTENING);
                       f->setNearFieldRadius(2); return f; }
        }
    };

    std::printf("  %-30s %11s %9s %9s %9s\n", "solver", "err@20k", "N=50k", "N=200k", "N=1M");
    for (int solver = 0; solver < 7; solver++) {
        std::printf("  %-30s", names[solver]);

        {
            std::unique_ptr<Simulation> acc = build(solver, 20000);
            acc->step(0.01f);
            const double e = measure(*acc, 512, solver == 6).rms_norm;
            // Brute force is exact by construction; anything above the float
            // summation floor means a defect, not an approximation.
            if (solver == 0 && e > 1e-4) {
                failures++;
                std::printf("  <- brute force GPU exceeds the float floor, FAIL");
            }
            std::printf(" %11.2e", e);
            std::fflush(stdout);
        }

        for (int n : {50000, 200000, 1000000}) {
            // Brute force at a million particles is 10^12 interactions a step.
            if (solver == 0 && n > 200000) { std::printf(" %9s", "skipped"); continue; }
            std::unique_ptr<Simulation> sim = build(solver, n);
            std::printf(" %9.2f", time_solver(*sim, 5));
            std::fflush(stdout);
        }
        std::printf("\n");
    }

    std::printf("\n%s\n", failures ? "FAILURES PRESENT" : "all checks passed");
    glfwDestroyWindow(window);
    glfwTerminate();
    return failures ? 1 : 0;
}
