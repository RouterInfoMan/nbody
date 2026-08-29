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
#include "galaxy_preset.h"
#include "collapse_preset.h"

static const int INIT_WIDTH = 1280;
static const int INIT_HEIGHT = 720;

enum Algorithm {
    ALG_BRUTE_CPU,
    ALG_BRUTE_GPU,
    ALG_BARNES_HUT_CPU,
    ALG_COUNT
};

static const char* ALG_NAMES[] = {
    "Brute Force (CPU)",
    "Brute Force (GPU - Compute Shader)",
    "Barnes-Hut (CPU)",
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
            if (!paused && simulation)
                simulation->step(dt * time_scale);
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
        }

        renderer->setNeedsUpdate(true);
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

        if (simulation) {
            const auto& p = simulation->getParticles();
            renderer->render(p, *camera, !paused);
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

        if (current_alg == ALG_BARNES_HUT_CPU)
            ImGui::SliderFloat("Theta", &sim_theta, 0.1f, 2.0f, "%.2f");

        // Preset selection
        ImGui::SeparatorText("Preset");
        bool preset_changed = ImGui::Combo("Preset", &current_preset, PRESET_NAMES, PRESET_COUNT);

        ImGui::SetNextItemWidth(120);
        ImGui::InputInt("Particles", &particle_count, 1000, 10000);
        particle_count = std::max(100, particle_count);

        // Physics params
        ImGui::SeparatorText("Physics");
        ImGui::SliderFloat("G", &sim_G, 0.01f, 10.0f, "%.2f");
        ImGui::SliderFloat("Softening", &sim_softening, 0.01f, 5.0f, "%.2f");
        ImGui::SliderFloat("Time Step", &dt, 0.001f, 0.1f, "%.4f");
        ImGui::SliderFloat("Time Scale", &time_scale, 0.1f, 10.0f, "%.1f");

        if (alg_changed || preset_changed || ImGui::Button("Reset")) {
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
        }
        ImGui::Text("Sim: %.2f ms  Render: %.2f ms", sim_ms, render_ms);
        ImGui::Text("Total: %.2f ms", sim_ms + render_ms);

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
