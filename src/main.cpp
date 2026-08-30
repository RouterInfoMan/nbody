#include <iostream>
#include <memory>
#include <chrono>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include "renderer.h"
#include "camera.h"
#include "simulation.h"
#include "brute_force_cpu.h"
#include "brute_force_gpu.h"
#include "barnes_hut_cpu.h"
#include "barnes_hut_gpu.h"
#include "fmm_cpu.h"
#include "galaxy_preset.h"
#include "collapse_preset.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

static const int INIT_WIDTH = 1280;
static const int INIT_HEIGHT = 720;

enum Algorithm {
    ALG_BRUTE_CPU,
    ALG_BRUTE_GPU,
    ALG_BARNES_HUT_CPU,
    ALG_BARNES_HUT_GPU,
    ALG_FMM_CPU,
    ALG_COUNT
};

static const char* ALG_NAMES[] = {
    "Brute Force (CPU)",
    "Brute Force (GPU - Compute Shader)",
    "Barnes-Hut (CPU)",
    "Barnes-Hut (GPU - LBVH)",
    "FMM (CPU)",
};

static const char* COLOR_MODE_NAMES[] = {
    "Speed (blackbody)",
    "Mass (blackbody)",
    "Monochrome",
};

static const char* FMM_KERNEL_NAMES[] = {
    "Softened 1/r^2 (matches other solvers)",
    "Logarithmic (true 2D gravity)",
};

enum PresetType {
    PRESET_GALAXY,
    PRESET_COLLAPSE,
    PRESET_COUNT
};

static const char* PRESET_NAMES[] = {
    "Galaxy",
    "Collapse",
};

class Application {
public:
    ~Application() { cleanup(); }

    bool initialize() {
        if (!glfwInit()) {
            std::cerr << "Failed to init GLFW" << std::endl;
            return false;
        }

        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
        glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

        window = glfwCreateWindow(INIT_WIDTH, INIT_HEIGHT, "N-Body Simulation", nullptr, nullptr);
        if (!window) {
            std::cerr << "Failed to create window" << std::endl;
            glfwTerminate();
            return false;
        }

        glfwMakeContextCurrent(window);
        glfwSwapInterval(1);

        glewExperimental = GL_TRUE;
        if (glewInit() != GLEW_OK) {
            std::cerr << "Failed to init GLEW" << std::endl;
            return false;
        }

        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
        ImGui::StyleColorsDark();
        ImGui_ImplGlfw_InitForOpenGL(window, true);
        ImGui_ImplOpenGL3_Init("#version 430");

        renderer = std::make_unique<Renderer>();
        {
            int fbw, fbh;
            glfwGetFramebufferSize(window, &fbw, &fbh);
            renderer->initialize(fbw, fbh);
        }

        camera = std::make_unique<Camera2D>(INIT_WIDTH, INIT_HEIGHT);

        createSimulation();

        glfwSetWindowUserPointer(window, this);
        glfwSetFramebufferSizeCallback(window, [](GLFWwindow* w, int width, int height) {
            glViewport(0, 0, width, height);
            auto* app = static_cast<Application*>(glfwGetWindowUserPointer(w));
            app->camera->setScreenSize(float(width), float(height));
            app->renderer->setViewport(width, height);
        });
        glfwSetScrollCallback(window, [](GLFWwindow* w, double, double y) {
            auto* app = static_cast<Application*>(glfwGetWindowUserPointer(w));
            app->camera->zoom(1.0f + float(y) * 0.1f);
        });

        std::cout << "OpenGL " << glGetString(GL_VERSION) << std::endl;
        return true;
    }

    void run() {
        double last_time = glfwGetTime();
        int frame_count = 0;
        double fps_timer = 0.0;

        while (!glfwWindowShouldClose(window)) {
            double now = glfwGetTime();
            double delta = now - last_time;
            last_time = now;

            frame_count++;
            fps_timer += delta;
            if (fps_timer >= 1.0) {
                fps = frame_count;
                frame_count = 0;
                fps_timer = 0.0;
            }

            processInput(delta);

            // The speed distribution drifts as the system heats, so the colour
            // ramp is periodically refitted to it -- auto-exposure, in effect.
            calibrate_timer += delta;
            if (auto_calibrate && !paused && simulation && calibrate_timer > 1.0) {
                calibrate_timer = 0.0;
                if (simulation->isGPUResident())
                    renderer->calibrateRangeFromGPU(simulation->velocityBuffer(),
                                                    simulation->massBuffer(),
                                                    simulation->getParticleCount());
                else
                    renderer->calibrateRange(simulation->getParticles());
            }

            auto t0 = std::chrono::high_resolution_clock::now();
            if (!paused && simulation) {
                simulation->step(dt * time_scale);
                if (sync_gpu_timing && simulation->isGPUResident())
                    glFinish();
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            sim_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

            auto t2 = std::chrono::high_resolution_clock::now();
            renderScene();
            renderUI();
            auto t3 = std::chrono::high_resolution_clock::now();
            render_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

            glfwSwapBuffers(window);
            glfwPollEvents();
        }
    }

private:
    GLFWwindow* window = nullptr;
    std::unique_ptr<Renderer> renderer;
    std::unique_ptr<Camera2D> camera;
    std::unique_ptr<Simulation> simulation;

    bool paused = true;
    float time_scale = 1.0f;
    float particle_size = 2.0f;
    float dt = 0.01f;

    int color_mode = COLOR_SPEED;
    float temp_range[2] = { 1700.0f, 11000.0f };
    float intensity = 1.0f;
    float exposure = 1.0f;
    float framed_radius = 0.0f;
    bool rebuild_pending = false;
    bool auto_calibrate = true;
    double calibrate_timer = 0.0;
    bool bloom_enabled = true;
    float bloom_strength = 0.55f;
    float bloom_threshold = 0.7f;
    float bloom_radius = 1.0f;
    float vignette = 0.35f;
    float saturation = 1.65f;
    float hue_preserve = 0.75f;
    float variety = 0.6f;
    int fps = 0;
    double sim_ms = 0.0;
    double render_ms = 0.0;

    int current_alg = ALG_BARNES_HUT_CPU;
    int current_preset = PRESET_GALAXY;
    int particle_count = 50000;

    GalaxyParams galaxy_params;
    CollapseParams collapse_params;

    float sim_G = 1.0f;
    float sim_softening = 1.0f;
    float sim_theta = 0.9f;
    bool bh_quadrupole = true;

    int fmm_kernel = 0;
    int fmm_order = 4;
    int fmm_leaf_capacity = 32;
    int fmm_near_radius = 2;
    int fmm_depth = 0;          // 0 = auto

    // GPU solvers no longer stall on a readback, so timing them needs an
    // explicit sync; without it sim_ms would only measure command submission.
    bool sync_gpu_timing = true;

    float err_max = -1.0f;
    float err_rms = -1.0f;
    int err_samples = 0;

    void createSimulation() {
        auto preset = makePreset();

        switch (current_alg) {
            case ALG_BRUTE_CPU:
                simulation = std::make_unique<BruteForceCPU>(std::move(preset), sim_G, sim_softening);
                break;
            case ALG_BRUTE_GPU:
                simulation = std::make_unique<BruteForceGPU>(std::move(preset), sim_G, sim_softening);
                break;
            case ALG_BARNES_HUT_CPU: {
                auto bh = std::make_unique<BarnesHutCPU>(std::move(preset), sim_theta, sim_G, sim_softening);
                bh->setQuadrupole(bh_quadrupole);
                simulation = std::move(bh);
                break;
            }
            case ALG_BARNES_HUT_GPU: {
                auto bh = std::make_unique<BarnesHutGPU>(std::move(preset), sim_theta, sim_G, sim_softening);
                bh->setQuadrupole(bh_quadrupole);
                simulation = std::move(bh);
                break;
            }
            case ALG_FMM_CPU: {
                auto fmm = std::make_unique<FMMCPU>(
                    std::move(preset),
                    fmm_kernel == 0 ? FMMKernel::Softened : FMMKernel::Logarithmic,
                    fmm_order, fmm_leaf_capacity, sim_G, sim_softening);
                fmm->setNearFieldRadius(fmm_near_radius);
                if (fmm_depth > 0) fmm->setDepth(fmm_depth);
                simulation = std::move(fmm);
                break;
            }
        }

        err_max = err_rms = -1.0f;
        err_samples = 0;
        renderer->setNeedsUpdate(true);
        // The preset's initial state is representative, and at t = 0 the host
        // copy is valid even for GPU solvers, so this costs no readback.
        renderer->calibrateRange(simulation->getParticles());
        calibrate_timer = 0.0;

        // Emission is additive, so the same per-particle brightness that looks
        // right at ten thousand particles saturates the whole core at a
        // million. Scale it with density so every count starts out legible;
        // the slider still overrides.
        const float n = float(std::max(1, simulation->getParticleCount()));
        intensity = std::clamp(1.9f * std::sqrt(20000.0f / n), 0.15f, 2.5f);
        renderer->setIntensity(intensity);

        // Frame the new configuration, and make R return there. Measured from
        // the particles rather than read off a preset field, so it stays
        // correct for any preset.
        //
        // Only jump the view when the scale actually changed. A rebuild fires
        // for every preset tweak, and yanking the camera back each time you
        // nudge dispersion or reroll the seed -- while zoomed into the core --
        // would make those controls unusable.
        float extent = 0.0f;
        for (const Particle& part : simulation->getParticles())
            extent = std::max(extent, glm::length(part.position));

        if (extent > 0.0f) {
            const bool rescaled = framed_radius <= 0.0f
                || std::abs(extent - framed_radius) > framed_radius * 0.2f;
            camera->setHomeRadius(extent, rescaled);
            framed_radius = extent;
        }
    }

    // Compares the solver's current accelerations against a direct sum over
    // every particle, for an evenly spaced sample of targets. The state is
    // already consistent after a step (positions were advanced, then forces
    // recomputed at those positions), so no extra stepping is needed.
    void validateAgainstBruteForce() {
        if (!simulation) return;

        simulation->syncToHost();
        const std::vector<Particle>& ps = simulation->getParticles();
        const int n = static_cast<int>(ps.size());
        if (n == 0) return;

        const int samples = std::min(n, 1024);
        const int step_size = std::max(1, n / samples);
        const float eps_sq = sim_softening * sim_softening;

        double sum_sq = 0.0;
        double worst = 0.0;
        int counted = 0;

        for (int i = 0; i < n; i += step_size) {
            glm::vec2 ref(0.0f);
            for (int j = 0; j < n; j++) {
                glm::vec2 d = ps[j].position - ps[i].position;
                float r2 = glm::dot(d, d) + eps_sq;
                float inv = 1.0f / std::sqrt(r2);
                ref += d * (ps[j].mass * inv * inv * inv);
            }
            ref *= sim_G;

            const float ref_mag = glm::length(ref);
            if (ref_mag < 1e-12f) continue;

            const float rel = glm::length(ps[i].acceleration - ref) / ref_mag;
            worst = std::max(worst, double(rel));
            sum_sq += double(rel) * double(rel);
            counted++;
        }

        err_samples = counted;
        err_max = counted ? float(worst) : -1.0f;
        err_rms = counted ? float(std::sqrt(sum_sq / counted)) : -1.0f;
    }

    std::unique_ptr<Preset> makePreset() {
        // Particle count and G are shared controls, so they are pushed into
        // whichever parameter set is active rather than duplicated in the UI.
        // G matters: the presets balance orbits against it, and a preset built
        // for a different G starts out of equilibrium.
        if (current_preset == PRESET_COLLAPSE) {
            collapse_params.count = particle_count;
            collapse_params.G = sim_G;
            return std::make_unique<CollapsePreset>(collapse_params);
        }
        galaxy_params.count = particle_count;
        galaxy_params.G = sim_G;
        return std::make_unique<GalaxyPreset>(galaxy_params);
    }

    // Preset parameters. Returns true if anything was edited.
    bool presetControls() {
        bool edited = false;

        // The two parameter sets reuse labels ("Central Mass", "Particle
        // Mass", "Dispersion"), so scope them per preset. Only one branch is
        // ever submitted, but without this they share an ID and any in-flight
        // widget state would carry across a preset switch.
        ImGui::PushID(current_preset);

        if (current_preset == PRESET_GALAXY) {
            GalaxyParams& g = galaxy_params;
            // Logarithmic: the range spans five decades, and on a linear slider
            // everything below a few hundred would collapse into a few pixels.
            edited |= ImGui::SliderFloat("Inner Radius", &g.inner_radius, 0.1f, 5000.0f, "%.1f",
                                         ImGuiSliderFlags_Logarithmic);
            edited |= ImGui::SliderFloat("Outer Radius", &g.outer_radius, 1.0f, 5000.0f, "%.1f",
                                         ImGuiSliderFlags_Logarithmic);
            g.outer_radius = std::max(g.outer_radius, g.inner_radius * 1.01f);

            edited |= ImGui::SliderFloat("Central Mass", &g.central_mass, 0.0f, 200000.0f, "%.0f",
                                         ImGuiSliderFlags_Logarithmic);
            edited |= ImGui::SliderFloat("Particle Mass", &g.particle_mass, 0.0f, 100.0f, "%.4f",
                                         ImGuiSliderFlags_Logarithmic);
            edited |= ImGui::SliderFloat("Falloff", &g.density_falloff, 0.0f, 2.0f, "%.2f");
            edited |= ImGui::SliderFloat("Dispersion", &g.velocity_dispersion, 0.0f, 0.5f, "%.3f");
            edited |= ImGui::SliderFloat("Spin", &g.spin, -1.5f, 1.5f, "%.2f");

            const float disc = g.particle_mass * float(particle_count);
            ImGui::TextDisabled("Disc mass %.4g  (disc:central %.3g)", disc,
                                g.central_mass > 0.0f ? disc / g.central_mass : INFINITY);
            if (disc > g.central_mass * 0.5f && g.velocity_dispersion < 0.05f)
                ImGui::TextDisabled("Self-gravity dominates: expect fragmentation.");
        } else {
            CollapseParams& c = collapse_params;
            edited |= ImGui::SliderFloat("Radius", &c.radius, 1.0f, 5000.0f, "%.1f",
                                         ImGuiSliderFlags_Logarithmic);
            edited |= ImGui::SliderFloat("Central Mass", &c.central_mass, 0.0f, 200000.0f, "%.0f",
                                         ImGuiSliderFlags_Logarithmic);
            edited |= ImGui::SliderFloat("Particle Mass", &c.particle_mass, 0.0f, 100.0f, "%.4f",
                                         ImGuiSliderFlags_Logarithmic);
            edited |= ImGui::SliderFloat("Rotation", &c.rotation, 0.0f, 1.5f, "%.2f");
            edited |= ImGui::SliderFloat("Dispersion", &c.velocity_dispersion, 0.0f, 1.5f, "%.2f");

            ImGui::TextDisabled("Total mass %.4g", c.particle_mass * float(particle_count));
            ImGui::TextDisabled("Rotation 0 = cold radial collapse, 1 = supported.");
        }

        uint32_t& seed = (current_preset == PRESET_GALAXY) ? galaxy_params.seed
                                                           : collapse_params.seed;
        int seed_i = int(seed);
        if (ImGui::InputInt("Seed", &seed_i)) {
            seed = uint32_t(std::max(0, seed_i));
            edited = true;
        }
        ImGui::SameLine();
        if (ImGui::Button("Randomize")) {
            seed = uint32_t(std::random_device{}());
            edited = true;
        }

        ImGui::PopID();
        return edited;
    }

    void processInput(double delta) {
        ImGuiIO& io = ImGui::GetIO();

        static double last_x = 0.0, last_y = 0.0;
        static bool was_dragging = false;

        bool right_down = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;
        if (right_down && !io.WantCaptureMouse) {
            double xpos, ypos;
            glfwGetCursorPos(window, &xpos, &ypos);

            if (was_dragging) {
                int w, h;
                glfwGetWindowSize(window, &w, &h);
                float aspect = float(w) / float(h);
                float vh = 100.0f;
                float vw = vh * aspect;

                float world_dx = float(xpos - last_x) * vw / w;
                float world_dy = float(ypos - last_y) * vh / h;
                camera->move(-world_dx, world_dy);
            }

            last_x = xpos;
            last_y = ypos;
            was_dragging = true;
        } else {
            was_dragging = false;
        }

        if (!io.WantCaptureKeyboard) {
            if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
                camera->zoom(1.0f + float(delta));
            if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
                camera->zoom(1.0f - float(delta));

            static bool space_held = false;
            if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS) {
                if (!space_held) { paused = !paused; space_held = true; }
            } else {
                space_held = false;
            }

            if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS)
                camera->reset();
            if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
                glfwSetWindowShouldClose(window, true);
        }
    }

    void renderScene() {
        // The renderer owns its HDR target and clears it itself; clearing the
        // default framebuffer here would only be overwritten by the composite.
        if (!simulation) return;

        if (simulation->isGPUResident()) {
            renderer->renderFromGPU(simulation->positionBuffer(),
                                    simulation->velocityBuffer(),
                                    simulation->massBuffer(),
                                    simulation->idBuffer(),
                                    simulation->getParticleCount(),
                                    *camera);
        } else {
            renderer->render(simulation->getParticles(), *camera, !paused);
        }
    }

    void renderUI() {
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(410, 680), ImGuiCond_FirstUseEver);
        ImGui::Begin("Simulation Controls");

        // Declared up front because the sections that set them are
        // collapsible: a folded section submits no widgets, and the rebuild
        // decision at the bottom still has to be made.
        bool alg_changed = false;
        bool preset_changed = false;
        bool rebuild_needed = false;
        bool count_changed = false;
        bool preset_params_changed = false;

        // Transport stays outside the collapsible sections so it is always
        // reachable. The button is submitted unconditionally -- a widget
        // short-circuited out of a boolean chain is not drawn that frame.
        const bool reset_clicked = ImGui::Button("Reset");
        ImGui::SameLine();
        if (ImGui::Button(paused ? "Play (Space)" : "Pause (Space)"))
            paused = !paused;
        ImGui::SameLine();
        ImGui::TextDisabled("%d fps", fps);

        if (ImGui::CollapsingHeader("Algorithm", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::PushID("algorithm");
            alg_changed = ImGui::Combo("Algorithm", &current_alg, ALG_NAMES, ALG_COUNT);

            const bool is_bh = current_alg == ALG_BARNES_HUT_CPU
                            || current_alg == ALG_BARNES_HUT_GPU;

            if (is_bh) {
                if (ImGui::SliderFloat("Theta", &sim_theta, 0.1f, 2.0f, "%.2f") && simulation)
                    simulation->setTheta(sim_theta);

                if (ImGui::Checkbox("Quadrupole moments", &bh_quadrupole) && simulation) {
                    if (auto* g = dynamic_cast<BarnesHutGPU*>(simulation.get())) g->setQuadrupole(bh_quadrupole);
                    if (auto* c = dynamic_cast<BarnesHutCPU*>(simulation.get())) c->setQuadrupole(bh_quadrupole);
                }
                ImGui::TextDisabled("Second moments cost ~5%% and buy ~10x accuracy,\n"
                                    "so theta 0.9-1.2 beats monopole at theta 0.5.");
            }

            if (current_alg == ALG_FMM_CPU) {
                rebuild_needed |= ImGui::Combo("Kernel", &fmm_kernel, FMM_KERNEL_NAMES, 2);

                const int max_order = (fmm_kernel == 0) ? 6 : 15;
                fmm_order = std::min(fmm_order, max_order);
                rebuild_needed |= ImGui::SliderInt("Order (p)", &fmm_order, 1, max_order);
                rebuild_needed |= ImGui::SliderInt("Leaf Capacity", &fmm_leaf_capacity, 4, 256);
                rebuild_needed |= ImGui::SliderInt("Near-field radius", &fmm_near_radius, 1, 2);
                ImGui::TextDisabled("radius 2 costs ~2x and buys ~17x accuracy.");

                const int auto_depth = FMMCPU::autoDepth(particle_count, fmm_leaf_capacity);
                rebuild_needed |= ImGui::SliderInt("Depth (0 = auto)", &fmm_depth, 0, 12);
                ImGui::TextDisabled("Auto depth for these settings: %d", auto_depth);

                if (fmm_kernel == 1)
                    ImGui::TextDisabled("Log kernel is a different force law -- it will not\n"
                                        "agree with brute force, and validation is meaningless.");
            }
            ImGui::PopID();
        }

        if (ImGui::CollapsingHeader("Preset", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::PushID("preset");
            preset_changed = ImGui::Combo("Preset", &current_preset, PRESET_NAMES, PRESET_COUNT);

            ImGui::SetNextItemWidth(120);
            count_changed = ImGui::InputInt("Particles", &particle_count, 1000, 10000);
            particle_count = std::max(100, particle_count);

            preset_params_changed = presetControls();
            ImGui::PopID();
        }

        if (ImGui::CollapsingHeader("Physics", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::PushID("physics");
            bool physics_changed = false;
            physics_changed |= ImGui::SliderFloat("G", &sim_G, 0.01f, 10.0f, "%.2f");
            physics_changed |= ImGui::SliderFloat("Softening", &sim_softening, 0.01f, 5.0f, "%.2f");
            if (physics_changed && simulation)
                simulation->setPhysics(sim_G, sim_softening);
            if (physics_changed)
                ImGui::TextDisabled("Presets balance orbits against G; Reset to rebuild.");

            ImGui::SliderFloat("Time Step", &dt, 0.001f, 0.1f, "%.4f");
            ImGui::SliderFloat("Time Scale", &time_scale, 0.1f, 10.0f, "%.1f");
            ImGui::PopID();
        }

        if (ImGui::CollapsingHeader("Appearance")) {
            ImGui::PushID("appearance");
            if (ImGui::SliderFloat("Point Size", &particle_size, 1.0f, 20.0f))
                renderer->setParticleSize(particle_size);

            if (ImGui::Combo("Color By", &color_mode, COLOR_MODE_NAMES, COLOR_MODE_COUNT)) {
                renderer->setColorMode(color_mode);
                if (simulation) {
                    simulation->syncToHost();
                    renderer->calibrateRange(simulation->getParticles());
                }
            }

            if (color_mode != COLOR_MONO) {
                ImGui::Text("Ramp spans %.3g to %.3g", renderer->rangeLow(), renderer->rangeHigh());
                ImGui::Checkbox("Auto", &auto_calibrate);
                ImGui::SameLine();
                if (ImGui::Button("Recalibrate colors") && simulation) {
                    simulation->syncToHost();
                    renderer->calibrateRange(simulation->getParticles());
                }
                ImGui::SameLine();
                ImGui::TextDisabled("(5th-95th pct)");

                if (ImGui::SliderFloat2("Temp K (cool/hot)", temp_range, 1700.0f, 20000.0f, "%.0f")) {
                    temp_range[1] = std::max(temp_range[1], temp_range[0] + 100.0f);
                    renderer->setTemperatureRange(temp_range[0], temp_range[1]);
                }
            }

            if (ImGui::SliderFloat("Brightness", &intensity, 0.05f, 5.0f, "%.2f"))
                renderer->setIntensity(intensity);
            if (ImGui::SliderFloat("Exposure", &exposure, 0.05f, 4.0f, "%.2f"))
                renderer->setExposure(exposure);

            if (ImGui::Checkbox("Bloom", &bloom_enabled))
                renderer->setBloomEnabled(bloom_enabled);
            if (bloom_enabled) {
                if (ImGui::SliderFloat("Bloom Strength", &bloom_strength, 0.0f, 2.0f, "%.2f"))
                    renderer->setBloomStrength(bloom_strength);
                if (ImGui::SliderFloat("Bloom Threshold", &bloom_threshold, 0.0f, 4.0f, "%.2f"))
                    renderer->setBloomThreshold(bloom_threshold);
                if (ImGui::SliderFloat("Bloom Radius", &bloom_radius, 0.5f, 3.0f, "%.2f"))
                    renderer->setBloomRadius(bloom_radius);
            }

            if (ImGui::SliderFloat("Saturation", &saturation, 0.0f, 2.5f, "%.2f"))
                renderer->setSaturation(saturation);
            if (ImGui::SliderFloat("Hue Preserve", &hue_preserve, 0.0f, 1.0f, "%.2f"))
                renderer->setHuePreserve(hue_preserve);
            ImGui::SameLine();
            ImGui::TextDisabled("(?)");
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("0 = per-channel ACES, which desaturates bright areas.\n"
                                  "1 = tone map luminance only, keeping chroma.");

            if (color_mode != COLOR_MONO) {
                if (ImGui::SliderFloat("Star Variety", &variety, 0.0f, 1.0f, "%.2f"))
                    renderer->setVariety(variety);
                ImGui::SameLine();
                ImGui::TextDisabled("(?)");
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("Per-particle temperature spread, skewed like a\n"
                                      "stellar mass function: mostly cool, few hot and bright.");
            }

            if (ImGui::SliderFloat("Vignette", &vignette, 0.0f, 1.5f, "%.2f"))
                renderer->setVignette(vignette);
            ImGui::PopID();
        }

        if (ImGui::CollapsingHeader("Performance", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::PushID("performance");
            if (simulation) {
                ImGui::Text("Particles: %d", simulation->getParticleCount());
                ImGui::Text("Sim Time: %.2f", simulation->getTime());
                if (current_alg == ALG_FMM_CPU) {
                    if (auto* fmm = dynamic_cast<FMMCPU*>(simulation.get()))
                        ImGui::Text("Tree: depth %d, %d cells, p = %d, ws = %d",
                                    fmm->depth(), fmm->cellCount(), fmm->order(),
                                    fmm->nearFieldRadius());
                }
            }
            ImGui::Text("Sim: %.2f ms  Render: %.2f ms", sim_ms, render_ms);
            ImGui::Text("Total: %.2f ms", sim_ms + render_ms);

            if (simulation && simulation->isGPUResident())
                ImGui::Checkbox("Sync GPU for timing", &sync_gpu_timing);
            ImGui::PopID();
        }

        if (ImGui::CollapsingHeader("Accuracy")) {
            ImGui::PushID("accuracy");
            if (ImGui::Button("Validate vs Brute Force"))
                validateAgainstBruteForce();
            ImGui::SameLine();
            ImGui::TextDisabled("(O(N) direct sums, may stall)");

            if (err_samples > 0) {
                ImGui::Text("Rel. accel error over %d samples:", err_samples);
                ImGui::Text("  max %.3e   rms %.3e", err_max, err_rms);
            } else {
                ImGui::TextDisabled("Not measured yet.");
            }
            ImGui::PopID();
        }

        if (ImGui::CollapsingHeader("Camera")) {
            ImGui::PushID("camera");
            const glm::vec2 cam = camera->getPosition();
            ImGui::Text("Position: (%.1f, %.1f)  Zoom: %.3fx", cam.x, cam.y, camera->getZoom());
            ImGui::TextDisabled("RMB drag | Q/E zoom | R reset");
            ImGui::PopID();
        }

        // Sliders report an edit on every frame they are dragged, and a
        // rebuild reallocates every buffer and re-seeds the whole system.
        // Defer until the control is released so dragging stays interactive.
        if (rebuild_needed || preset_params_changed || count_changed)
            rebuild_pending = true;

        const bool commit = rebuild_pending && !ImGui::IsAnyItemActive();
        if (alg_changed || preset_changed || commit || reset_clicked) {
            rebuild_pending = false;
            paused = true;
            createSimulation();
        }

        ImGui::End();
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    }

    void cleanup() {
        simulation.reset();
        renderer.reset();

        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();

        if (window) {
            glfwDestroyWindow(window);
            window = nullptr;
        }
        glfwTerminate();
    }
};

int main() {
    Application app;
    if (!app.initialize()) return 1;
    app.run();
    return 0;
}
