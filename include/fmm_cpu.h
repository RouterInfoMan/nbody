#pragma once
#include "simulation.h"
#include "preset.h"
#include "thread_pool.h"
#include "cpu_radix_sort.h"
#include <complex>
#include <functional>
#include <cstdint>
#include <memory>
#include <vector>

// Which analytic kernel the expansions are built for.
//
// Softened: the same softened inverse-square law the other solvers use
//   (K(r) = 1/sqrt(r^2 + eps^2), a = G*grad(K)), expanded with Cartesian
//   Taylor tensors. Kernel-agnostic and therefore directly comparable
//   against brute force.
//
// Logarithmic: true 2D gravity (phi = G*m*log r, a ~ 1/r), expanded with the
//   classic Greengard-Rokhlin complex multipoles. Far more accurate per term,
//   but it is a DIFFERENT force law -- results will not match the other
//   solvers, and the presets' orbital velocities are tuned for 1/r^2.
enum class FMMKernel {
    Softened,
    Logarithmic,
};

class FMMCPU : public Simulation {
public:
    FMMCPU(std::unique_ptr<Preset> preset,
           FMMKernel kernel = FMMKernel::Softened,
           int order = 4,
           int leaf_capacity = 32,
           float G = 1.0f,
           float softening = 1.0f);

    void step(float dt) override;
    void reset() override;
    const char* name() const override {
        return kernel == FMMKernel::Softened ? "FMM (CPU, 1/r^2)"
                                             : "FMM (CPU, log)";
    }

    void setPhysics(float g, float soft) override { G = g; softening = soft; }

    void setKernel(FMMKernel k) { kernel = k; rebuildTables(); }
    void setOrder(int p);
    void setDepth(int d);
    // Cells within this many cells of a target are handled by direct
    // summation. Raising it to 2 pushes the nearest translated pair from a
    // 2-cell to a 3-cell separation, improving the expansion ratio from ~0.71
    // to ~0.47 at the cost of a 25-cell near field and a 75-entry
    // interaction list.
    void setNearFieldRadius(int ws);
    void setLeafCapacity(int cap);

    int depth() const { return max_level; }
    int order() const { return p; }
    int nearFieldRadius() const { return near_radius; }
    int leafCapacity() const { return leaf_capacity; }
    int cellCount() const;
    FMMKernel kernelMode() const { return kernel; }

    // Depth that the constructor would pick for a given particle count.
    static int autoDepth(int n, int leaf_capacity);

private:
    // Morton codes are 16 bits per axis, so the tree can address 16 levels.
    static constexpr int MAX_DEPTH = 16;
    // Coarsest level with a non-empty interaction list: below this every cell
    // is a neighbour of every other, so there is nothing to translate.
    static constexpr int FIRST_M2L_LEVEL = 2;

    // Open-addressed key -> cell index map. The interaction lists do tens of
    // lookups per cell per level, which is far too hot for a binary search
    // over the level's key array.
    struct CellIndex {
        static constexpr uint32_t EMPTY = 0xFFFFFFFFu;

        std::vector<uint32_t> slot_keys;
        std::vector<int> slot_vals;
        uint32_t mask = 0;

        static uint32_t hash(uint32_t k) {
            k *= 0x9E3779B1u;
            k ^= k >> 15;
            k *= 0x85EBCA6Bu;
            k ^= k >> 13;
            return k;
        }

        void build(const std::vector<uint32_t>& keys);
        int find(uint32_t key) const {
            uint32_t h = hash(key) & mask;
            for (;;) {
                const uint32_t k = slot_keys[h];
                if (k == key) return slot_vals[h];
                if (k == EMPTY) return -1;
                h = (h + 1) & mask;
            }
        }
    };

    struct Level {
        std::vector<uint32_t> keys;      // ascending, one per non-empty cell
        std::vector<int> body_start;     // range into `sorted`
        std::vector<int> body_end;
        std::vector<int> parent;         // index into level l-1
        std::vector<int> child_begin;    // range into level l+1
        std::vector<int> child_end;
        CellIndex index;

        size_t size() const { return keys.size(); }
        void clear();
    };

    std::unique_ptr<Preset> preset;
    ThreadPool pool;
    CpuRadixSort sorter;

    FMMKernel kernel;
    int p;                  // expansion order
    int leaf_capacity;
    int max_level;
    int near_radius = 1;    // cells within this distance are summed directly
    float G;
    float softening;

    // Morton-sorted particle state, kept as separate arrays rather than a
    // vector<Particle>. The near-field direct sum dominates FMM cost at every
    // useful setting, and striding it through a 36-byte struct defeats
    // vectorisation entirely; contiguous float arrays let it use full SIMD
    // width. Accelerations are computed here and scattered back via order_map.
    std::vector<float> pos_x, pos_y, mass;
    std::vector<float> acc_x, acc_y;
    std::vector<uint32_t> codes;
    std::vector<int> order_map;      // sorted slot -> original particle index
    std::vector<uint64_t> sort_keys; // (morton << 32) | original index

    std::vector<Level> levels;
    glm::vec2 root_min{0.0f};
    float root_size = 1.0f;

    // --- Cartesian (softened) expansions -----------------------------
    // Flat per level: [cell * stride + coefficient].
    std::vector<std::vector<double>> cart_multipole;
    std::vector<std::vector<double>> cart_local;
    // Per level, per V-list offset: a stride x stride matrix mapping a
    // multipole to its contribution to the target's local expansion. The
    // translation only depends on the integer cell offset, so 40 matrices per
    // level cover every M2L in the tree.
    std::vector<std::vector<double>> m2l_matrices;
    std::vector<int> alpha_x, alpha_y;    // exponents per flat coefficient
    std::vector<double> binomial;         // (2p+3)^2 table

    // --- Complex (logarithmic) expansions ----------------------------
    std::vector<std::vector<std::complex<double>>> cx_multipole;
    std::vector<std::vector<std::complex<double>>> cx_local;

    int stride = 0;         // coefficients per cell

    void rebuildTables();
    void buildTree();
    void sortByMorton();
    void buildLevels();

    // Every per-cell pass goes through here so the coarse levels, which hold
    // only a handful of cells, run inline instead of paying dispatch.
    static constexpr int CELL_GRAIN = 64;
    void forEachCell(int cells, const std::function<void(int, int)>& fn) {
        pool.parallel_for(0, cells, fn, CELL_GRAIN);
    }

    void upwardPass();
    void interactionPass();
    void downwardPass();
    void nearFieldPass();

    // Cartesian
    void buildM2LMatrices();
    void cartP2M();
    void cartM2M(int level);
    void cartM2L(int level);
    void cartL2L(int level);
    void cartL2P();

    // Complex
    void cxP2M();
    void cxM2M(int level);
    void cxM2L(int level);
    void cxL2L(int level);
    void cxL2P();

    void verletStep1(float dt);
    void verletStep2(float dt);

    float cellSize(int level) const { return root_size / float(1 << level); }
    glm::vec2 cellCenter(int level, uint32_t key) const;
    // Index of the cell with `key` at `level`, or -1 if it holds no particles.
    int findCell(int level, uint32_t key) const { return levels[level].index.find(key); }

    // The largest cell offset an interaction-list entry can have, and the
    // width of the square translation table that covers it.
    int m2lReach() const { return 2 * near_radius + 1; }
    int m2lSpan() const { return 2 * m2lReach() + 1; }

    double binom(int n, int k) const { return binomial[n * (2 * p + 3) + k]; }
};
