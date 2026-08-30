#include "renderer.h"
#include <algorithm>
#include <cmath>
#include <iostream>

Renderer::Renderer() = default;

Renderer::~Renderer() {
    destroyTargets();
    if (particle_vao) glDeleteVertexArrays(1, &particle_vao);
    if (fullscreen_vao) glDeleteVertexArrays(1, &fullscreen_vao);
    if (particle_vbo) glDeleteBuffers(1, &particle_vbo);
    if (position_ssbo) glDeleteBuffers(1, &position_ssbo);
    if (velocity_ssbo) glDeleteBuffers(1, &velocity_ssbo);
    if (mass_ssbo) glDeleteBuffers(1, &mass_ssbo);
}

void Renderer::initialize(int width, int height) {
    range_hi = 0.0f;   // marks the ramp as uncalibrated
    particle_shader  = std::make_unique<Shader>("shaders/particle.vert", "shaders/particle.frag");
    color_shader     = std::make_unique<Shader>("shaders/color_compute.comp");
    gather_shader    = std::make_unique<Shader>("shaders/particle_gather.comp");
    down_shader      = std::make_unique<Shader>("shaders/fullscreen.vert", "shaders/bloom_down.frag");
    up_shader        = std::make_unique<Shader>("shaders/fullscreen.vert", "shaders/bloom_up.frag");
    composite_shader = std::make_unique<Shader>("shaders/fullscreen.vert", "shaders/composite.frag");

    glGenBuffers(1, &position_ssbo);
    glGenBuffers(1, &velocity_ssbo);
    glGenBuffers(1, &mass_ssbo);
    glGenVertexArrays(1, &fullscreen_vao);

    setupBuffers();
    createTargets(width, height);

    glEnable(GL_PROGRAM_POINT_SIZE);
    glDisable(GL_DEPTH_TEST);
}

void Renderer::setupBuffers() {
    glGenVertexArrays(1, &particle_vao);
    glGenBuffers(1, &particle_vbo);

    glBindVertexArray(particle_vao);
    glBindBuffer(GL_ARRAY_BUFFER, particle_vbo);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    glBindVertexArray(0);
}

void Renderer::createTargets(int width, int height) {
    destroyTargets();

    fb_width = std::max(1, width);
    fb_height = std::max(1, height);

    // RGBA16F: particle emission routinely sums well past 1.0 and the bloom
    // pass needs that headroom to have anything to work with.
    glGenTextures(1, &hdr_tex);
    glBindTexture(GL_TEXTURE_2D, hdr_tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, fb_width, fb_height, 0, GL_RGBA, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glGenFramebuffers(1, &hdr_fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, hdr_fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, hdr_tex, 0);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        std::cerr << "HDR framebuffer incomplete" << std::endl;

    int w = fb_width, h = fb_height;
    for (int i = 0; i < MAX_BLOOM_MIPS; i++) {
        w = std::max(1, w / 2);
        h = std::max(1, h / 2);

        BloomMip mip;
        mip.width = w;
        mip.height = h;

        glGenTextures(1, &mip.tex);
        glBindTexture(GL_TEXTURE_2D, mip.tex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R11F_G11F_B10F, w, h, 0, GL_RGB, GL_FLOAT, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        // Clamping matters: the tent filter samples outside the mip at the
        // edges, and wrapping would smear the opposite side of the screen in.
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

        glGenFramebuffers(1, &mip.fbo);
        glBindFramebuffer(GL_FRAMEBUFFER, mip.fbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, mip.tex, 0);

        bloom_mips.push_back(mip);

        if (w == 1 || h == 1) break;
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void Renderer::destroyTargets() {
    for (BloomMip& m : bloom_mips) {
        if (m.fbo) glDeleteFramebuffers(1, &m.fbo);
        if (m.tex) glDeleteTextures(1, &m.tex);
    }
    bloom_mips.clear();

    if (hdr_fbo) { glDeleteFramebuffers(1, &hdr_fbo); hdr_fbo = 0; }
    if (hdr_tex) { glDeleteTextures(1, &hdr_tex); hdr_tex = 0; }
}

void Renderer::setViewport(int width, int height) {
    if (width == fb_width && height == fb_height) return;
    createTargets(width, height);
}

void Renderer::ensureVertexCapacity(size_t count) {
    if (count <= vertex_capacity) return;
    glBindBuffer(GL_ARRAY_BUFFER, particle_vbo);
    glBufferData(GL_ARRAY_BUFFER, count * 6 * sizeof(float), nullptr, GL_STREAM_DRAW);
    vertex_capacity = count;
}

void Renderer::setRangeFromSample(std::vector<float>& sample) {
    if (sample.empty()) return;

    // Percentiles rather than min/max: a single escaping particle at a huge
    // speed would otherwise squash the entire visible range into one colour.
    std::sort(sample.begin(), sample.end());
    const size_t lo_i = size_t(double(sample.size() - 1) * 0.05);
    const size_t hi_i = size_t(double(sample.size() - 1) * 0.95);

    const float lo = sample[lo_i];
    const float hi = std::max(sample[hi_i], lo * 1.05f + 1e-3f);

    // Ease toward the new window. A self-gravitating system heats in bursts,
    // and snapping the ramp makes the whole field change colour at once.
    if (range_hi > 0.0f) {
        const float k = 0.25f;
        range_lo += (lo - range_lo) * k;
        range_hi += (hi - range_hi) * k;
    } else {
        range_lo = lo;
        range_hi = hi;
    }
    needs_update = true;
}

void Renderer::calibrateRange(const std::vector<Particle>& particles) {
    if (particles.empty()) return;

    const size_t stride = std::max<size_t>(1, particles.size() / 8192);
    std::vector<float> sample;
    sample.reserve(particles.size() / stride + 1);

    for (size_t i = 0; i < particles.size(); i += stride) {
        sample.push_back(color_mode == COLOR_MASS ? particles[i].mass
                                                  : glm::length(particles[i].velocity));
    }
    setRangeFromSample(sample);
}

void Renderer::calibrateRangeFromGPU(GLuint vel_ssbo, GLuint mass_buf, int count) {
    if (count <= 0) return;

    std::vector<float> sample;
    const size_t stride = std::max<size_t>(1, size_t(count) / 8192);

    if (color_mode == COLOR_MASS) {
        if (!mass_buf) return;
        std::vector<float> masses(count);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, mass_buf);
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
                           GLsizeiptr(count) * sizeof(float), masses.data());
        for (size_t i = 0; i < masses.size(); i += stride) sample.push_back(masses[i]);
    } else {
        if (!vel_ssbo) return;
        std::vector<glm::vec2> vel(count);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, vel_ssbo);
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
                           GLsizeiptr(count) * sizeof(glm::vec2), vel.data());
        for (size_t i = 0; i < vel.size(); i += stride) sample.push_back(glm::length(vel[i]));
    }

    setRangeFromSample(sample);
}

void Renderer::uploadParticles(const std::vector<Particle>& particles) {
    const size_t n = particles.size();

    std::vector<glm::vec2> positions(n), velocities(n);
    std::vector<float> masses(n);
    for (size_t i = 0; i < n; i++) {
        positions[i] = particles[i].position;
        velocities[i] = particles[i].velocity;
        masses[i] = particles[i].mass;
    }

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, position_ssbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, n * sizeof(glm::vec2), positions.data(), GL_STREAM_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, velocity_ssbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, n * sizeof(glm::vec2), velocities.data(), GL_STREAM_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, mass_ssbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, n * sizeof(float), masses.data(), GL_STREAM_DRAW);
}

void Renderer::shadeIntoVertexBuffer(GLuint pos, GLuint vel, GLuint mass, GLuint ids, int count) {
    ensureVertexCapacity(static_cast<size_t>(count));

    const GLuint groups = static_cast<GLuint>((count + 255) / 256);

    if (vel) {
        color_shader->use();
        color_shader->set_int("particleCount", count);
        color_shader->set_int("colorMode", color_mode);
        color_shader->set_int("hasMass", mass ? 1 : 0);
        color_shader->set_float("logLo", std::log(1.0f + std::max(range_lo, 0.0f)));
        color_shader->set_float("logHi", std::log(1.0f + std::max(range_hi, range_lo + 1e-3f)));
        color_shader->set_float("tempLow", temp_low);
        color_shader->set_float("tempHigh", temp_high);
        color_shader->set_float("intensity", intensity);
        color_shader->set_int("hasIds", ids ? 1 : 0);
        color_shader->set_float("variety", variety);

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, pos);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, vel);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, particle_vbo);
        // The shader only reads this when hasMass is set, but a binding must
        // still be present for the buffer block to be valid.
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, mass ? mass : pos);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, ids ? ids : pos);
    } else {
        gather_shader->use();
        gather_shader->set_int("particleCount", count);
        gather_shader->set_float("intensity", intensity);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, pos);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, particle_vbo);
    }

    glDispatchCompute(groups, 1, 1);
    glMemoryBarrier(GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT);
}

void Renderer::beginScene() {
    glBindFramebuffer(GL_FRAMEBUFFER, hdr_fbo);
    glViewport(0, 0, fb_width, fb_height);
    glClearColor(0.004f, 0.006f, 0.014f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    // Additive: overlapping sprites sum instead of occluding, which is what
    // turns particle density into brightness.
    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE);
}

void Renderer::drawPoints(int count, const Camera2D& camera) {
    particle_shader->use();
    particle_shader->set_mat4("projection", camera.getProjectionMatrix());
    particle_shader->set_float("pointSize", particle_size);

    glBindVertexArray(particle_vao);
    glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(count));
    glBindVertexArray(0);
}

void Renderer::drawFullscreen() {
    glBindVertexArray(fullscreen_vao);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glBindVertexArray(0);
}

void Renderer::runBloom() {
    if (bloom_mips.empty()) return;

    glDisable(GL_BLEND);

    // Downsample chain, bright-passing only on the first read of the scene.
    down_shader->use();
    down_shader->set_float("threshold", bloom_threshold);
    down_shader->set_float("knee", std::max(bloom_threshold * 0.5f, 1e-3f));

    GLuint src_tex = hdr_tex;
    int src_w = fb_width, src_h = fb_height;

    for (size_t i = 0; i < bloom_mips.size(); i++) {
        const BloomMip& mip = bloom_mips[i];
        glBindFramebuffer(GL_FRAMEBUFFER, mip.fbo);
        glViewport(0, 0, mip.width, mip.height);

        down_shader->set_int("firstPass", i == 0 ? 1 : 0);
        down_shader->set_int("source", 0);
        down_shader->set_vec2("texelSize", 1.0f / float(src_w), 1.0f / float(src_h));

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, src_tex);
        drawFullscreen();

        src_tex = mip.tex;
        src_w = mip.width;
        src_h = mip.height;
    }

    // Upsample back up the chain, accumulating additively.
    up_shader->use();
    up_shader->set_int("source", 0);
    up_shader->set_float("radius", bloom_radius);

    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE);

    for (size_t i = bloom_mips.size() - 1; i > 0; i--) {
        const BloomMip& src = bloom_mips[i];
        const BloomMip& dst = bloom_mips[i - 1];

        glBindFramebuffer(GL_FRAMEBUFFER, dst.fbo);
        glViewport(0, 0, dst.width, dst.height);

        up_shader->set_vec2("texelSize", 1.0f / float(src.width), 1.0f / float(src.height));
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, src.tex);
        drawFullscreen();
    }

    glDisable(GL_BLEND);
}

void Renderer::endScene() {
    if (bloom_enabled) runBloom();

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(0, 0, fb_width, fb_height);
    glDisable(GL_BLEND);

    composite_shader->use();
    composite_shader->set_int("scene", 0);
    composite_shader->set_int("bloom", 1);
    composite_shader->set_float("exposure", exposure);
    composite_shader->set_float("bloomStrength", bloom_enabled ? bloom_strength : 0.0f);
    composite_shader->set_float("vignette", vignette);
    composite_shader->set_float("saturation", saturation);
    composite_shader->set_float("huePreserve", hue_preserve);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, hdr_tex);
    glActiveTexture(GL_TEXTURE1);
    // With bloom off the shader still samples the slot, so point it at
    // something valid and scale its contribution to zero above.
    glBindTexture(GL_TEXTURE_2D, bloom_mips.empty() ? hdr_tex : bloom_mips[0].tex);

    drawFullscreen();

    glActiveTexture(GL_TEXTURE0);

    // ImGui draws next and expects ordinary alpha blending.
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
}

void Renderer::render(const std::vector<Particle>& particles, const Camera2D& camera, bool force_update) {
    const int count = static_cast<int>(particles.size());

    if (force_update || needs_update || particles.size() != last_particle_count) {
        uploadParticles(particles);
        // Host-side solvers keep `particles` in a stable order, so array
        // position is already a valid identity and needs no id buffer.
        shadeIntoVertexBuffer(position_ssbo, velocity_ssbo, mass_ssbo, 0, count);
        last_particle_count = particles.size();
        needs_update = false;
    }

    beginScene();
    if (count > 0) drawPoints(count, camera);
    endScene();
}

void Renderer::renderFromGPU(GLuint pos_ssbo, GLuint vel_ssbo, GLuint mass_buf,
                             GLuint id_buf, int count, const Camera2D& camera) {
    if (pos_ssbo && count > 0)
        shadeIntoVertexBuffer(pos_ssbo, vel_ssbo, mass_buf, id_buf, count);

    last_particle_count = static_cast<size_t>(count);
    needs_update = false;

    beginScene();
    if (pos_ssbo && count > 0) drawPoints(count, camera);
    endScene();
}
