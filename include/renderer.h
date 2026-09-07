#pragma once
#include <GL/glew.h>
#include <memory>
#include <vector>
#include "shader.h"
#include "camera.h"
#include "particle.h"

// Particle colouring source.
enum ColorMode {
    COLOR_SPEED = 0,      // blackbody temperature from orbital speed
    COLOR_MASS = 1,       // blackbody temperature from particle mass
    COLOR_MONO = 2,       // flat warm white, brightness from overlap alone
    COLOR_MODE_COUNT
};

// HDR renderer: additive sprites -> bloom chain -> filmic tone map.
class Renderer {
public:
    Renderer();
    ~Renderer();

    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;

    void initialize(int width, int height);
    void setViewport(int width, int height);

    void render(const std::vector<Particle>& particles, const Camera2D& camera, bool force_update);

    void renderFromGPU(GLuint pos_ssbo, GLuint vel_ssbo, GLuint mass_ssbo,
                       GLuint id_ssbo, int count, const Camera2D& camera);

    void drawTreeOverlay(GLuint box_ssbo, int box_count, const Camera2D& camera,
                         int max_depth, float alpha);

    void drawBoundary(const Camera2D& camera, int shape, float radius, float alpha);

    void setParticleSize(float size) { particle_size = size; needs_update = true; }
    void setNeedsUpdate(bool update) { needs_update = update; }

    void setColorMode(int mode) { color_mode = mode; needs_update = true; }
    void setTemperatureRange(float low, float high) { temp_low = low; temp_high = high; needs_update = true; }
    void setRange(float low, float high) { range_lo = low; range_hi = high; needs_update = true; }
    void calibrateRange(const std::vector<Particle>& particles);
    void calibrateRangeFromGPU(GLuint vel_ssbo, GLuint mass_buf, int count);
    void setIntensity(float i) { intensity = i; needs_update = true; }
    void setExposure(float e) { exposure = e; }
    void setBloomStrength(float s) { bloom_strength = s; }
    void setBloomThreshold(float t) { bloom_threshold = t; }
    void setBloomRadius(float r) { bloom_radius = r; }
    void setBloomEnabled(bool on) { bloom_enabled = on; }
    void setVignette(float v) { vignette = v; }
    void setSaturation(float s) { saturation = s; }
    void setHuePreserve(float h) { hue_preserve = h; }
    void setVariety(float v) { variety = v; needs_update = true; }

    int colorMode() const { return color_mode; }
    float temperatureLow() const { return temp_low; }
    float temperatureHigh() const { return temp_high; }
    float rangeLow() const { return range_lo; }
    float rangeHigh() const { return range_hi; }
    float intensityValue() const { return intensity; }
    float exposureValue() const { return exposure; }
    float bloomStrengthValue() const { return bloom_strength; }
    float bloomThresholdValue() const { return bloom_threshold; }
    float bloomRadiusValue() const { return bloom_radius; }
    bool bloomEnabled() const { return bloom_enabled; }
    float vignetteValue() const { return vignette; }
    float saturationValue() const { return saturation; }
    float huePreserveValue() const { return hue_preserve; }
    float varietyValue() const { return variety; }

private:
    std::unique_ptr<Shader> particle_shader;
    std::unique_ptr<Shader> color_shader;
    std::unique_ptr<Shader> gather_shader;
    std::unique_ptr<Shader> down_shader;
    std::unique_ptr<Shader> up_shader;
    std::unique_ptr<Shader> composite_shader;
    std::unique_ptr<Shader> tree_collect_shader;
    std::unique_ptr<Shader> tree_line_shader;
    std::unique_ptr<Shader> boundary_line_shader;

    GLuint particle_vao = 0, particle_vbo = 0;
    GLuint fullscreen_vao = 0;
    GLuint position_ssbo = 0, velocity_ssbo = 0, mass_ssbo = 0;

    GLuint hdr_fbo = 0, hdr_tex = 0;
    GLuint tree_box_ssbo = 0, tree_cmd_ssbo = 0;

    struct BloomMip {
        GLuint fbo = 0;
        GLuint tex = 0;
        int width = 0;
        int height = 0;
    };
    std::vector<BloomMip> bloom_mips;

    int fb_width = 0, fb_height = 0;

    float particle_size = 2.0f;
    int color_mode = COLOR_SPEED;
    float temp_low = 1700.0f;
    float temp_high = 11000.0f;
    float range_lo = 0.0f;
    float range_hi = 40.0f;
    float intensity = 1.0f;
    float exposure = 1.0f;
    float bloom_strength = 0.55f;
    float bloom_threshold = 0.7f;
    float bloom_radius = 1.0f;
    bool bloom_enabled = true;
    float vignette = 0.35f;
    float saturation = 1.65f;
    float hue_preserve = 0.75f;
    float variety = 0.6f;

    size_t last_particle_count = 0;
    size_t vertex_capacity = 0;
    bool needs_update = true;

    static constexpr int MAX_BLOOM_MIPS = 6;
    static constexpr int TREE_BOX_CAPACITY = 1 << 18;

    void setupBuffers();
    void createTargets(int width, int height);
    void destroyTargets();
    void ensureVertexCapacity(size_t count);

    // Uploads host particle state into the compute path's SSBOs.
    void uploadParticles(const std::vector<Particle>& particles);
    // Runs the colouring compute pass into the vertex buffer.
    void shadeIntoVertexBuffer(GLuint pos, GLuint vel, GLuint mass, GLuint ids, int count);

    void beginScene();
    void drawPoints(int count, const Camera2D& camera);
    void endScene();
    void runBloom();
    void drawFullscreen();
    void setRangeFromSample(std::vector<float>& sample);
};
