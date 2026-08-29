#include "gpu_radix_sort.h"
#include <utility>

GpuRadixSort::GpuRadixSort() {
    histogram_shader = std::make_unique<Shader>("shaders/radix_histogram.comp");
    scan_shader = std::make_unique<Shader>("shaders/radix_scan.comp");
    scatter_shader = std::make_unique<Shader>("shaders/radix_scatter.comp");
    glGenBuffers(1, &hist_ssbo);
}

GpuRadixSort::~GpuRadixSort() {
    if (hist_ssbo) glDeleteBuffers(1, &hist_ssbo);
}

void GpuRadixSort::ensureHistogram(int num_blocks) {
    int needed = RADIX * num_blocks;
    if (needed <= hist_capacity) return;

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, hist_ssbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, needed * sizeof(GLuint), nullptr, GL_DYNAMIC_COPY);
    hist_capacity = needed;
}

void GpuRadixSort::sort(GLuint keys, GLuint values, GLuint keys_tmp, GLuint values_tmp, int count) {
    if (count <= 1) return;

    const int num_blocks = blockCount(count);
    ensureHistogram(num_blocks);

    GLuint key_src = keys, val_src = values;
    GLuint key_dst = keys_tmp, val_dst = values_tmp;

    for (int pass = 0; pass < PASSES; pass++) {
        const int shift = pass * RADIX_BITS;

        histogram_shader->use();
        histogram_shader->set_int("particleCount", count);
        histogram_shader->set_int("numBlocks", num_blocks);
        histogram_shader->set_int("shiftBits", shift);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, key_src);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, hist_ssbo);
        glDispatchCompute(static_cast<GLuint>(num_blocks), 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        scan_shader->use();
        scan_shader->set_int("histSize", RADIX * num_blocks);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, hist_ssbo);
        glDispatchCompute(1, 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        scatter_shader->use();
        scatter_shader->set_int("particleCount", count);
        scatter_shader->set_int("numBlocks", num_blocks);
        scatter_shader->set_int("shiftBits", shift);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, key_src);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, hist_ssbo);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, val_src);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, key_dst);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, val_dst);
        glDispatchCompute(static_cast<GLuint>(num_blocks), 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        std::swap(key_src, key_dst);
        std::swap(val_src, val_dst);
    }

    // PASSES is even, so the final swap returns the data to the input buffers.
}
