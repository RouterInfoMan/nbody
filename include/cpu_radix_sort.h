#pragma once
#include "thread_pool.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

// Parallel LSD radix sort over the high 32 bits of packed
// (key << 32 | payload) values.
//
// Only the key half is sorted -- four byte passes instead of eight -- and
// because each pass is stable the payload half stays in ascending order
// within a run of equal keys, which is exactly the tie-break a Morton sort
// wants. Packing key and payload into one 64-bit word also keeps the scatter
// to a single stream instead of two.
class CpuRadixSort {
public:
    static constexpr int RADIX = 256;
    static constexpr int PASSES = 4;
    // Below this the histogram and dispatch overhead costs more than it saves.
    static constexpr int PARALLEL_THRESHOLD = 1 << 14;

    void sort(std::vector<uint64_t>& data, int n, ThreadPool& pool) {
        if (n < 2) return;
        if (n < PARALLEL_THRESHOLD) {
            std::sort(data.begin(), data.begin() + n);
            return;
        }

        const int chunks = std::max(1, pool.size());

        scratch.resize(size_t(n));
        histograms.assign(size_t(chunks) * RADIX, 0u);
        bounds.resize(size_t(chunks) + 1);
        for (int i = 0; i <= chunks; i++)
            bounds[i] = int(int64_t(n) * i / chunks);

        uint64_t* src = data.data();
        uint64_t* dst = scratch.data();

        for (int pass = 0; pass < PASSES; pass++) {
            const int shift = 32 + pass * 8;

            std::fill(histograms.begin(), histograms.end(), 0u);
            pool.parallel_for(0, chunks, [&](int a, int b) {
                for (int c = a; c < b; c++) {
                    uint32_t* h = &histograms[size_t(c) * RADIX];
                    for (int i = bounds[c]; i < bounds[c + 1]; i++)
                        h[(src[i] >> shift) & 0xFFu]++;
                }
            });

            // Exclusive scan in bin-major order, so each chunk's run of a
            // given digit lands contiguously and the scatter stays stable.
            uint32_t running = 0;
            for (int bin = 0; bin < RADIX; bin++) {
                for (int c = 0; c < chunks; c++) {
                    uint32_t& slot = histograms[size_t(c) * RADIX + bin];
                    const uint32_t count = slot;
                    slot = running;
                    running += count;
                }
            }

            pool.parallel_for(0, chunks, [&](int a, int b) {
                uint32_t offsets[RADIX];
                for (int c = a; c < b; c++) {
                    std::memcpy(offsets, &histograms[size_t(c) * RADIX], sizeof(offsets));
                    for (int i = bounds[c]; i < bounds[c + 1]; i++) {
                        const uint64_t v = src[i];
                        dst[offsets[(v >> shift) & 0xFFu]++] = v;
                    }
                }
            });

            std::swap(src, dst);
        }

        // PASSES is even, so the sorted data is back in `data`.
    }

    static uint64_t pack(uint32_t key, uint32_t payload) {
        return (uint64_t(key) << 32) | uint64_t(payload);
    }
    static uint32_t key(uint64_t v) { return uint32_t(v >> 32); }
    static uint32_t payload(uint64_t v) { return uint32_t(v & 0xFFFFFFFFu); }

private:
    std::vector<uint64_t> scratch;
    std::vector<uint32_t> histograms;
    std::vector<int> bounds;
};
