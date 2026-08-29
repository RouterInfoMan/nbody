#pragma once
#include "shader.h"
#include <GL/glew.h>
#include <memory>

// 4-bit LSD radix sort over 32-bit keys with a uint payload, entirely on the
// GPU. Eight passes of histogram -> global scan -> stable scatter; because the
// pass count is even the sorted result lands back in the caller's original
// buffers and the temporaries are left holding scratch.
class GpuRadixSort {
public:
    static constexpr int RADIX_BITS = 4;
    static constexpr int RADIX = 1 << RADIX_BITS;
    static constexpr int PASSES = 32 / RADIX_BITS;
    static constexpr int WORKGROUP_SIZE = 256;
    static constexpr int ITEMS_PER_THREAD = 8;
    static constexpr int BLOCK_SIZE = WORKGROUP_SIZE * ITEMS_PER_THREAD;

    GpuRadixSort();
    ~GpuRadixSort();

    GpuRadixSort(const GpuRadixSort&) = delete;
    GpuRadixSort& operator=(const GpuRadixSort&) = delete;

    // Sorts `count` key/value pairs. On return `keys` and `values` hold the
    // sorted sequence; `keys_tmp` and `values_tmp` must be buffers of the same
    // size and are clobbered.
    void sort(GLuint keys, GLuint values, GLuint keys_tmp, GLuint values_tmp, int count);

    static int blockCount(int count) { return (count + BLOCK_SIZE - 1) / BLOCK_SIZE; }

private:
    std::unique_ptr<Shader> histogram_shader;
    std::unique_ptr<Shader> scan_shader;
    std::unique_ptr<Shader> scatter_shader;

    GLuint hist_ssbo = 0;
    int hist_capacity = 0;

    void ensureHistogram(int num_blocks);
};
