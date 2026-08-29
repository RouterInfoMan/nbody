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
        renderer->initialize();

        camera = std::make_unique<Camera2D>(INIT_WIDTH, INIT_HEIGHT);

        createSimulation();

        glfwSetWindowUserPointer(window, this);
        glfwSetFramebufferSizeCallback(window, [](GLFWwindow* w, int width, int height) {
            glViewport(0, 0, width, height);
            auto* app = static_cast<Application*>(glfwGetWindowUserPointer(w));
            app->camera->setScreenSize(float(width), float(height));
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
    bool show_velocity = true;
    int fps = 0;
    double sim_ms = 0.0;
    double render_ms = 0.0;

    int current_alg = ALG_BARNES_HUT_CPU;
    int current_preset = PRESET_GALAXY;
    int particle_count = 10000;

    float sim_G = 1.0f;
    float sim_softening = 1.0f;
    float sim_theta = 0.5f;

    int fmm_kernel = 0;
    int fmm_order = 4;
    int fmm_leaf_capacity = 32;
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
            case ALG_BARNES_HUT_CPU:
                simulation = std::make_unique<BarnesHutCPU>(std::move(preset), sim_theta, sim_G, sim_softening);
                break;
            case ALG_BARNES_HUT_GPU:
                simulation = std::make_unique<BarnesHutGPU>(std::move(preset), sim_theta, sim_G, sim_softening);
                break;
            case ALG_FMM_CPU: {
                auto fmm = std::make_unique<FMMCPU>(
                    std::move(preset),
                    fmm_kernel == 0 ? FMMKernel::Softened : FMMKernel::Logarithmic,
                    fmm_order, fmm_leaf_capacity, sim_G, sim_softening);
                if (fmm_depth > 0) fmm->setDepth(fmm_depth);
                simulation = std::move(fmm);
                break;
            }
        }

        err_max = err_rms = -1.0f;
        err_samples = 0;
        renderer->setNeedsUpdate(true);
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
        switch (current_preset) {
            case PRESET_GALAXY:
                return std::make_unique<GalaxyPreset>(particle_count);
            case PRESET_COLLAPSE:
                return std::make_unique<CollapsePreset>(particle_count);
            default:
                return std::make_unique<GalaxyPreset>(particle_count);
        }
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
        glClearColor(0.05f, 0.05f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        if (!simulation) return;

        if (simulation->isGPUResident()) {
            renderer->renderFromGPU(simulation->positionBuffer(),
                                    simulation->velocityBuffer(),
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
        ImGui::SetNextWindowSize(ImVec2(320, 0), ImGuiCond_FirstUseEver);
        ImGui::Begin("Simulation Controls");

        // Algorithm selection
        ImGui::SeparatorText("Algorithm");
        bool alg_changed = ImGui::Combo("Algorithm", &current_alg, ALG_NAMES, ALG_COUNT);

        const bool is_bh = current_alg == ALG_BARNES_HUT_CPU || current_alg == ALG_BARNES_HUT_GPU;
        bool rebuild_needed = false;

        if (is_bh) {
            if (ImGui::SliderFloat("Theta", &sim_theta, 0.1f, 2.0f, "%.2f") && simulation)
                simulation->setTheta(sim_theta);
        }

        if (current_alg == ALG_FMM_CPU) {
            rebuild_needed |= ImGui::Combo("Kernel", &fmm_kernel, FMM_KERNEL_NAMES, 2);

            const int max_order = (fmm_kernel == 0) ? 6 : 15;
            fmm_order = std::min(fmm_order, max_order);
            rebuild_needed |= ImGui::SliderInt("Order (p)", &fmm_order, 1, max_order);
            rebuild_needed |= ImGui::SliderInt("Leaf Capacity", &fmm_leaf_capacity, 4, 256);

            int auto_depth = FMMCPU::autoDepth(particle_count, fmm_leaf_capacity);
            rebuild_needed |= ImGui::SliderInt("Depth (0 = auto)", &fmm_depth, 0, 12);
            ImGui::TextDisabled("Auto depth for these settings: %d", auto_depth);

            if (fmm_kernel == 1)
                ImGui::TextDisabled("Log kernel is a different force law -- it will not\n"
                                    "agree with brute force, and validation is meaningless.");
        }

        // Preset selection
        ImGui::SeparatorText("Preset");
        bool preset_changed = ImGui::Combo("Preset", &current_preset, PRESET_NAMES, PRESET_COUNT);

        ImGui::SetNextItemWidth(120);
        ImGui::InputInt("Particles", &particle_count, 1000, 10000);
        particle_count = std::max(100, particle_count);

        // Physics params
        ImGui::SeparatorText("Physics");
        bool physics_changed = false;
        physics_changed |= ImGui::SliderFloat("G", &sim_G, 0.01f, 10.0f, "%.2f");
        physics_changed |= ImGui::SliderFloat("Softening", &sim_softening, 0.01f, 5.0f, "%.2f");
        if (physics_changed && simulation)
            simulation->setPhysics(sim_G, sim_softening);

        ImGui::SliderFloat("Time Step", &dt, 0.001f, 0.1f, "%.4f");
        ImGui::SliderFloat("Time Scale", &time_scale, 0.1f, 10.0f, "%.1f");

        if (alg_changed || preset_changed || rebuild_needed || ImGui::Button("Reset")) {
            paused = true;
            createSimulation();
        }

        ImGui::SameLine();
        if (ImGui::Button(paused ? "Play (Space)" : "Pause (Space)"))
            paused = !paused;

        // Rendering
        ImGui::SeparatorText("Rendering");
        if (ImGui::SliderFloat("Point Size", &particle_size, 1.0f, 20.0f))
            renderer->setParticleSize(particle_size);
        if (ImGui::Checkbox("Color by Velocity", &show_velocity))
            renderer->setShowVelocity(show_velocity);

        // Camera
        ImGui::SeparatorText("Camera");
        glm::vec2 cam = camera->getPosition();
        ImGui::Text("Position: (%.1f, %.1f)  Zoom: %.3fx", cam.x, cam.y, camera->getZoom());
        ImGui::TextDisabled("RMB drag | Q/E zoom | R reset");

        // Performance
        ImGui::SeparatorText("Performance");
        ImGui::Text("FPS: %d", fps);
        if (simulation) {
            ImGui::Text("Particles: %d", simulation->getParticleCount());
            ImGui::Text("Sim Time: %.2f", simulation->getTime());
            if (current_alg == ALG_FMM_CPU) {
                if (auto* fmm = dynamic_cast<FMMCPU*>(simulation.get()))
                    ImGui::Text("Tree: depth %d, %d cells, p = %d",
                                fmm->depth(), fmm->cellCount(), fmm->order());
            }
        }
        ImGui::Text("Sim: %.2f ms  Render: %.2f ms", sim_ms, render_ms);
        ImGui::Text("Total: %.2f ms", sim_ms + render_ms);

        if (simulation && simulation->isGPUResident())
            ImGui::Checkbox("Sync GPU for timing", &sync_gpu_timing);

        // Accuracy
        ImGui::SeparatorText("Accuracy");
        if (ImGui::Button("Validate vs Brute Force"))
            validateAgainstBruteForce();
        ImGui::SameLine();
        ImGui::TextDisabled("(O(N) direct sums, may stall)");

        if (err_samples > 0) {
            ImGui::Text("Rel. accel error over %d samples:", err_samples);
            ImGui::Text("  max %.3e   rms %.3e", err_max, err_rms);
        } else if (err_samples == 0 && err_max < 0.0f) {
            ImGui::TextDisabled("Not measured yet.");
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
