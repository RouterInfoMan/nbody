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
#include "cloud_presets.h"
#include "boundary.h"
#include "thread_pool.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>
#include <cfloat>

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

static const char* BOUNDARY_MODE_NAMES[] = {
    "Off",
    "Bounce (wall)",
    "Drag (absorbing)",
};

static const char* BOUNDARY_SHAPE_NAMES[] = {
    "Circle",
    "Box",
};

static const char* FMM_KERNEL_NAMES[] = {
    "Softened 1/r^2 (matches other solvers)",
    "Logarithmic (true 2D gravity)",
};

enum PresetType {
    PRESET_GALAXY,
    PRESET_COLLAPSE,
    PRESET_BINARY_CLOUDS,
    PRESET_CLOUD_CLUSTER,
    PRESET_COUNT
};

static const char* PRESET_NAMES[] = {
    "Galaxy",
    "Collapse",
    "Binary Clouds",
    "Cloud Cluster",
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

            energy_timer += delta;
            if (energy_auto && !paused && simulation && energy_timer > double(energy_interval)) {
                energy_timer = 0.0;
                measureEnergy();
            }

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
    int tree_boxes_drawn = 0;
    bool rebuild_pending = false;

    bool energy_auto = true;
    float energy_interval = 0.25f;
    double energy_timer = 0.0;
    bool e_available = false;

    double e_kinetic = 0.0;
    double e_potential = 0.0;
    double e_total = 0.0;
    double e_reference = 0.0;      // first reading after a rebuild
    bool e_have_reference = false;
    double energy_ms = 0.0;

    static constexpr int ENERGY_HISTORY = 512;
    std::vector<float> e_hist_total, e_hist_kinetic;
    int e_hist_count = 0;

    void resetEnergy() {
        e_kinetic = e_potential = e_total = e_reference = 0.0;
        e_have_reference = false;
        e_hist_total.assign(ENERGY_HISTORY, 0.0f);
        e_hist_kinetic.assign(ENERGY_HISTORY, 0.0f);
        e_hist_count = 0;
        energy_timer = 0.0;
    }

    void measureEnergy() {
        if (!simulation) return;

        const auto t0 = std::chrono::high_resolution_clock::now();

        double k = 0.0, u = 0.0;
        e_available = simulation->energyTotals(k, u);
        if (!e_available) return;

        e_kinetic = k;
        e_potential = u;
        e_total = k + u;

        if (!e_have_reference) {
            e_reference = e_total;
            e_have_reference = true;
        }

        if (e_hist_total.empty()) resetEnergy();
        e_hist_total[e_hist_count % ENERGY_HISTORY] = float(e_total);
        e_hist_kinetic[e_hist_count % ENERGY_HISTORY] = float(e_kinetic);
        e_hist_count++;

        const auto t1 = std::chrono::high_resolution_clock::now();
        energy_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

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

    bool sync_gpu_timing = true;

    int current_alg = ALG_BARNES_HUT_CPU;
    int current_preset = PRESET_GALAXY;
    int particle_count = 50000;

    GalaxyParams galaxy_params;
    CollapseParams collapse_params;
    BinaryCloudsParams binary_params;
    CloudClusterParams cluster_params;

    Boundary boundary;
    bool show_boundary = true;

    bool show_tree = false;
    int tree_max_depth = 7;
    float tree_alpha = 0.5f;

    float sim_G = 1.0f;
    float sim_softening = 1.0f;
    float sim_theta = 0.9f;
    bool bh_quadrupole = true;

    int fmm_kernel = 0;
    int fmm_order = 4;
    int fmm_leaf_capacity = 32;
    int fmm_near_radius = 2;
    int fmm_depth = 0;          // 0 = auto

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

        simulation->setBoundary(boundary);
        resetEnergy();

        err_max = err_rms = -1.0f;
        err_samples = 0;
        renderer->setNeedsUpdate(true);
        renderer->calibrateRange(simulation->getParticles());
        calibrate_timer = 0.0;

        const float n = float(std::max(1, simulation->getParticleCount()));
        intensity = std::clamp(1.9f * std::sqrt(20000.0f / n), 0.15f, 2.5f);
        renderer->setIntensity(intensity);

        float extent = 0.0f;
        for (const Particle& part : simulation->getParticles())
            extent = std::max(extent, glm::length(part.position));
        // A wall the particles have not reached yet is still worth framing.
        if (boundary.active())
            extent = std::max(extent, boundary.radius * (boundary.shape == BOUNDARY_BOX ? 1.42f : 1.0f));

        if (extent > 0.0f) {
            const bool rescaled = framed_radius <= 0.0f
                || std::abs(extent - framed_radius) > framed_radius * 0.2f;
            camera->setHomeRadius(extent, rescaled);
            framed_radius = extent;
        }
    }

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
        switch (current_preset) {
            case PRESET_COLLAPSE:
                collapse_params.count = particle_count;
                collapse_params.G = sim_G;
                return std::make_unique<CollapsePreset>(collapse_params);
            case PRESET_BINARY_CLOUDS:
                binary_params.count = particle_count;
                binary_params.G = sim_G;
                return std::make_unique<BinaryCloudsPreset>(binary_params);
            case PRESET_CLOUD_CLUSTER:
                cluster_params.count = particle_count;
                cluster_params.G = sim_G;
                return std::make_unique<CloudClusterPreset>(cluster_params);
            default:
                galaxy_params.count = particle_count;
                galaxy_params.G = sim_G;
                return std::make_unique<GalaxyPreset>(galaxy_params);
        }
    }

    // Preset parameters. Returns true if anything was edited.
    bool presetControls() {
        bool edited = false;

        // Scope per preset: the two sets reuse labels and would share ImGui IDs.
        ImGui::PushID(current_preset);

        if (current_preset == PRESET_GALAXY) {
            GalaxyParams& g = galaxy_params;
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
        } else if (current_preset == PRESET_BINARY_CLOUDS) {
            BinaryCloudsParams& b = binary_params;
            edited |= ImGui::SliderFloat("Cloud Radius", &b.cloud_radius, 1.0f, 2000.0f, "%.1f",
                                         ImGuiSliderFlags_Logarithmic);
            edited |= ImGui::SliderFloat("Separation", &b.separation, 1.0f, 5000.0f, "%.1f",
                                         ImGuiSliderFlags_Logarithmic);
            edited |= ImGui::SliderFloat("Total Mass", &b.total_mass, 1.0f, 500000.0f, "%.0f",
                                         ImGuiSliderFlags_Logarithmic);
            edited |= ImGui::SliderFloat("Mass Ratio", &b.mass_ratio, 0.05f, 20.0f, "%.2f",
                                         ImGuiSliderFlags_Logarithmic);
            edited |= ImGui::SliderFloat("Orbit", &b.orbit_fraction, 0.0f, 1.5f, "%.2f");
            edited |= ImGui::SliderFloat("Support", &b.support, 0.0f, 2.0f, "%.2f");

            ImGui::TextDisabled("Orbit 0 = head-on free fall, 1 = circular.\n"
                                "Support 1 ~ virialised; 0 collapses at once.");
        } else if (current_preset == PRESET_CLOUD_CLUSTER) {
            CloudClusterParams& cc = cluster_params;
            edited |= ImGui::SliderInt("Clouds", &cc.cloud_count, 1, 32);
            edited |= ImGui::SliderFloat("Cloud Radius", &cc.cloud_radius, 1.0f, 2000.0f, "%.1f",
                                         ImGuiSliderFlags_Logarithmic);
            edited |= ImGui::SliderFloat("Cluster Radius", &cc.cluster_radius, 1.0f, 5000.0f, "%.1f",
                                         ImGuiSliderFlags_Logarithmic);
            edited |= ImGui::SliderFloat("Total Mass", &cc.total_mass, 1.0f, 500000.0f, "%.0f",
                                         ImGuiSliderFlags_Logarithmic);
            edited |= ImGui::SliderFloat("Orbit", &cc.orbit_fraction, 0.0f, 1.5f, "%.2f");
            edited |= ImGui::SliderFloat("Support", &cc.support, 0.0f, 2.0f, "%.2f");

            ImGui::TextDisabled("Clouds merge hierarchically; each holds %.4g mass.",
                                cc.total_mass / float(std::max(1, cc.cloud_count)));
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

        uint32_t* seed_ptr = &galaxy_params.seed;
        if (current_preset == PRESET_COLLAPSE)            seed_ptr = &collapse_params.seed;
        else if (current_preset == PRESET_BINARY_CLOUDS)  seed_ptr = &binary_params.seed;
        else if (current_preset == PRESET_CLOUD_CLUSTER)  seed_ptr = &cluster_params.seed;
        uint32_t& seed = *seed_ptr;

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

        if (show_boundary && boundary.active())
            renderer->drawBoundary(*camera, boundary.shape, boundary.radius, 0.55f);

        if (show_tree) {
            GLuint boxes = 0;
            int count = 0;
            if (simulation->treeGeometry(boxes, count)) {
                renderer->drawTreeOverlay(boxes, count, *camera, tree_max_depth, tree_alpha);
                tree_boxes_drawn = count;
            } else {
                tree_boxes_drawn = 0;
            }
        }
    }

    void drawEnergyWindow(float x, float w) {
        ImGui::SetNextWindowPos(ImVec2(x, 560.0f), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(w, 260.0f), ImGuiCond_FirstUseEver);
        ImGui::Begin("Energy");
        ImGui::PushID("energy");

        if (e_have_reference) {
            const double drift = (e_reference != 0.0)
                ? (e_total - e_reference) / std::abs(e_reference) * 100.0 : 0.0;

            ImGui::Text("Kinetic    %+.6g", e_kinetic);
            ImGui::Text("Potential  %+.6g", e_potential);
            ImGui::Separator();
            ImGui::Text("Total      %+.6g", e_total);

            ImGui::TextColored(std::abs(drift) < 1.0 ? ImVec4(0.6f, 0.9f, 0.6f, 1.0f)
                                                     : ImVec4(1.0f, 0.7f, 0.4f, 1.0f),
                               "Drift      %+.3f %% since reset", drift);

            const int count = std::min(e_hist_count, ENERGY_HISTORY);
            const int offset = (e_hist_count > ENERGY_HISTORY) ? (e_hist_count % ENERGY_HISTORY) : 0;
            ImGui::PlotLines("##etot", e_hist_total.data(), count, offset,
                             "total energy", FLT_MAX, FLT_MAX, ImVec2(-1, 60));
            ImGui::PlotLines("##ekin", e_hist_kinetic.data(), count, offset,
                             "kinetic", FLT_MAX, FLT_MAX, ImVec2(-1, 45));
        } else if (simulation && !e_available && e_hist_count == 0 && !energy_auto) {
            ImGui::TextDisabled("Not available for this solver.");
        } else {
            ImGui::TextDisabled("No measurement yet.");
        }

        ImGui::Separator();
        ImGui::Checkbox("Auto", &energy_auto);
        ImGui::SameLine();
        if (ImGui::Button("Measure now")) measureEnergy();
        ImGui::SameLine();
        if (ImGui::Button("Reset baseline")) resetEnergy();

        ImGui::SliderFloat("Interval (s)", &energy_interval, 0.05f, 5.0f, "%.2f");
        ImGui::TextDisabled("%.2f ms per measurement", energy_ms);
        ImGui::TextDisabled("Exact to the solver's own accuracy.");

        ImGui::PopID();
        ImGui::End();
    }

    void renderUI() {
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        int fbw = INIT_WIDTH, fbh = INIT_HEIGHT;
        glfwGetFramebufferSize(window, &fbw, &fbh);
        const float right_w = 400.0f;
        const float right_x = float(fbw) - right_w - 10.0f;

        ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(410, float(fbh) - 20.0f), ImGuiCond_FirstUseEver);
        ImGui::Begin("Simulation");

        bool alg_changed = false;
        bool preset_changed = false;
        bool rebuild_needed = false;
        bool count_changed = false;
        bool preset_params_changed = false;

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

                ImGui::Checkbox("Show tree", &show_tree);
                if (show_tree) {
                    ImGui::SliderInt("Tree Depth", &tree_max_depth, 1, 16);
                    ImGui::SliderFloat("Tree Opacity", &tree_alpha, 0.05f, 1.0f, "%.2f");
                    if (simulation)
                        ImGui::TextDisabled("%d %s (depth-limited)", tree_boxes_drawn,
                                            simulation->treeKind());
                }
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

            ImGui::SeparatorText("Boundary");
            bool bc = false;
            bc |= ImGui::Combo("Containment", &boundary.mode, BOUNDARY_MODE_NAMES,
                               BOUNDARY_MODE_COUNT);
            if (boundary.active()) {
                bc |= ImGui::Combo("Wall Shape", &boundary.shape, BOUNDARY_SHAPE_NAMES,
                                   BOUNDARY_SHAPE_COUNT);
                bc |= ImGui::SliderFloat("Wall Radius", &boundary.radius, 10.0f, 20000.0f, "%.0f",
                                         ImGuiSliderFlags_Logarithmic);
                if (boundary.mode == BOUNDARY_BOUNCE)
                    bc |= ImGui::SliderFloat("Restitution", &boundary.restitution, 0.0f, 1.0f, "%.2f");
                else
                    bc |= ImGui::SliderFloat("Drag", &boundary.drag, 0.0f, 40.0f, "%.1f");

                ImGui::Checkbox("Show wall", &show_boundary);
                ImGui::TextDisabled(boundary.mode == BOUNDARY_BOUNCE
                    ? "Reflects the outward velocity component;\n"
                      "restitution 1 is perfectly elastic."
                    : "Damps velocity beyond the wall, ramping with\n"
                      "distance, so escapers slow and fall back.");
            }
            if (bc && simulation) simulation->setBoundary(boundary);

            ImGui::PopID();
        }

        ImGui::End();   // Simulation

        ImGui::SetNextWindowPos(ImVec2(right_x, 10.0f), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(right_w, 300.0f), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowCollapsed(true, ImGuiCond_FirstUseEver);
        ImGui::Begin("Appearance");
        {
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

        ImGui::End();   // Appearance

        ImGui::SetNextWindowPos(ImVec2(right_x, 320.0f), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(right_w, 230.0f), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowCollapsed(true, ImGuiCond_FirstUseEver);
        ImGui::Begin("Performance");
        {
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

        ImGui::End();   // Performance

        drawEnergyWindow(right_x, right_w);

        // Defer the rebuild until the slider is released.
        if (rebuild_needed || preset_params_changed || count_changed)
            rebuild_pending = true;

        const bool commit = rebuild_pending && !ImGui::IsAnyItemActive();
        if (alg_changed || preset_changed || commit || reset_clicked) {
            rebuild_pending = false;
            paused = true;
            createSimulation();
        }

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
