#pragma once
#include "simulation.h"
#include "preset.h"
#include "thread_pool.h"
#include <memory>
#include <atomic>
#include <cstdint>

namespace Morton {
    inline uint64_t expandBits(uint32_t v) {
        uint64_t x = v & 0x00000000FFFFFFFF;
        x = (x | (x << 32)) & 0x00000000FFFFFFFF;
        x = (x | (x << 16)) & 0x0000FFFF0000FFFF;
        x = (x | (x <<  8)) & 0x00FF00FF00FF00FF;
        x = (x | (x <<  4)) & 0x0F0F0F0F0F0F0F0F;
        x = (x | (x <<  2)) & 0x3333333333333333;
        x = (x | (x <<  1)) & 0x5555555555555555;
        return x;
    }

    inline uint64_t encode(uint32_t x, uint32_t y) {
        return expandBits(x) | (expandBits(y) << 1);
    }

    inline uint64_t fromPosition(glm::vec2 pos, glm::vec2 min_b, glm::vec2 max_b) {
        glm::vec2 n = glm::clamp((pos - min_b) / (max_b - min_b), 0.0f, 0.999999f);
        return encode(uint32_t(n.x * 0xFFFFFFFF), uint32_t(n.y * 0xFFFFFFFF));
    }
}

struct Quad {
    glm::vec2 center{0.0f};
    float size = 0.0f;

    Quad() = default;
    Quad(glm::vec2 c, float s) : center(c), size(s) {}

    Quad child(int quadrant) const {
        Quad q;
        q.size = size * 0.5f;
        q.center.x = center.x + ((quadrant & 1) - 0.5f) * q.size;
        q.center.y = center.y + ((quadrant >> 1) - 0.5f) * q.size;
        return q;
    }
};

struct QuadNode {
    int children = 0;
    int next = 0;
    glm::vec2 com{0.0f};
    float mass = 0.0f;
    Quad quad;
    int body_start = 0;
    int body_end = 0;

    bool isLeaf() const { return children == 0; }
    int bodyCount() const { return body_end - body_start; }
};

class BarnesHutCPU : public Simulation {
public:
    BarnesHutCPU(std::unique_ptr<Preset> preset, float theta = 0.5f,
                 float G = 1.0f, float softening = 1.0f,
                 int leaf_cap = 16, int thread_cap = 512);

    void step(float dt) override;
    void reset() override;
    const char* name() const override { return "Barnes-Hut (CPU)"; }

private:
    std::unique_ptr<Preset> preset;
    ThreadPool pool;

    std::vector<QuadNode> nodes;
    std::vector<int> parent_map;
    std::atomic<int> node_count{0};

    std::vector<int> indices;
    std::vector<Particle> sorted;
    std::vector<uint64_t> morton_codes;

    float theta, theta_sq;
    float G;
    float softening_sq;
    int leaf_cap;
    int thread_cap;

    glm::vec2 bounds_min, bounds_max;

    void sortByMorton();
    void buildTree();
    void clearTree();
    int subdivide(int node_idx, int range_start, int range_end);
    void propagate();

    void computeForces();
    glm::vec2 computeAcceleration(glm::vec2 pos);

    void verletStep1(float dt);
    void verletStep2(float dt);

    template<typename Pred>
    int partitionRange(int start, int end, Pred pred);

    static constexpr int ROOT = 0;
};
