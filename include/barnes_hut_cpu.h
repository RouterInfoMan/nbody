#pragma once
#include "simulation.h"
#include "preset.h"
#include "thread_pool.h"
#include "cpu_radix_sort.h"
#include <memory>
#include <atomic>
#include <cstdint>

namespace Morton {
    // 16 bits per axis. The sort only exists to give the top-down build good
    // locality -- the tree structure itself comes from geometric partitioning
    // against cell centres -- so 65536 cells per axis is ample, and a 32-bit
    // code halves the radix sort's work versus a 64-bit one.
    inline uint32_t expandBits(uint32_t v) {
        v &= 0x0000FFFFu;
        v = (v | (v << 8)) & 0x00FF00FFu;
        v = (v | (v << 4)) & 0x0F0F0F0Fu;
        v = (v | (v << 2)) & 0x33333333u;
        v = (v | (v << 1)) & 0x55555555u;
        return v;
    }

    inline uint32_t encode(uint32_t x, uint32_t y) {
        return expandBits(x) | (expandBits(y) << 1);
    }

    inline uint32_t fromPosition(glm::vec2 pos, glm::vec2 min_b, glm::vec2 max_b) {
        glm::vec2 n = glm::clamp((pos - min_b) / (max_b - min_b), 0.0f, 0.999999f);
        return encode(uint32_t(n.x * 65536.0f), uint32_t(n.y * 65536.0f));
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
    // Second mass moment about the centre of mass, Q_ij = sum_k m_k t_i t_j.
    // Symmetric in 2D, so three components. Carrying it turns the multipole
    // error from O(theta^2) into O(theta^3), which buys back far more in
    // opening angle than the extra 12 bytes and handful of flops cost.
    float qxx = 0.0f, qxy = 0.0f, qyy = 0.0f;
    // Distance from the cell centre to the centre of mass. The opening test
    // measures distance to the COM but node extent from the cell centre; with
    // the COM near a cell boundary those disagree badly enough that a source
    // particle can be nearly as far from the COM as the target is. Requiring
    // d > size/theta + delta closes that gap (Salmon & Warren).
    float com_offset = 0.0f;
    Quad quad;
    int body_start = 0;
    int body_end = 0;

    bool isLeaf() const { return children == 0; }
    int bodyCount() const { return body_end - body_start; }
};

class BarnesHutCPU : public Simulation {
public:
    BarnesHutCPU(std::unique_ptr<Preset> preset, float theta = 0.7f,
                 float G = 1.0f, float softening = 1.0f,
                 int leaf_cap = 16, int split_threshold = 512);
    ~BarnesHutCPU() override;

    void step(float dt) override;
    void reset() override;
    const char* name() const override { return "Barnes-Hut (CPU)"; }

    void setPhysics(float g, float soft) override {
        G = g;
        softening_sq = soft * soft;
    }
    void setTheta(float t) override { theta = t; inv_theta = 1.0f / t; }
    bool treeGeometry(GLuint& buffer, int& count) override;
    const char* treeKind() const override { return "quadtree cells"; }

    void setQuadrupole(bool on) { use_quadrupole = on; }
    bool quadrupole() const { return use_quadrupole; }

private:
    std::unique_ptr<Preset> preset;
    ThreadPool pool;
    CpuRadixSort sorter;

    std::vector<QuadNode> nodes;
    std::vector<int> parent_map;
    std::atomic<int> node_count{0};
    std::atomic<bool> node_overflow{false};

    std::vector<uint64_t> sort_keys;   // (morton << 32) | index
    std::vector<Particle> sorted;
    // Original index of each entry in `sorted`. The tree build partitions
    // `sorted` in place, so this is permuted alongside it to keep the route
    // back to `particles` valid.
    std::vector<int> sorted_ids;

    float theta, inv_theta;
    float G;
    float softening_sq;
    int leaf_cap;
    // A node holding at least this many bodies is split breadth-first so the
    // resulting subtrees can be expanded in parallel.
    int split_threshold;
    bool use_quadrupole = true;

    glm::vec2 bounds_min, bounds_max;

    // Debug overlay geometry, built and uploaded only when asked for.
    GLuint tree_box_ssbo = 0;
    int tree_box_count = 0;
    std::vector<glm::vec4> tree_boxes;

    void sortByMorton();
    void buildTree();
    void clearTree();
    int subdivide(int node_idx, int range_start, int range_end);
    void expandSubtree(int root_node);
    void makeLeaf(int node_idx);
    bool shouldBeLeaf(int node_idx) const;
    void propagate();

    void computeForces();
    glm::vec2 computeAcceleration(glm::vec2 pos) const;

    void verletStep1(float dt);
    void verletStep2(float dt);

    template<typename Pred>
    int partitionRange(int start, int end, Pred pred);

    static constexpr int ROOT = 0;
};
