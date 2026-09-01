#include "fmm_cpu.h"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace {

constexpr int MAX_ORDER = 15;

inline uint32_t expandBits16(uint32_t v) {
    v &= 0x0000FFFFu;
    v = (v | (v << 8)) & 0x00FF00FFu;
    v = (v | (v << 4)) & 0x0F0F0F0Fu;
    v = (v | (v << 2)) & 0x33333333u;
    v = (v | (v << 1)) & 0x55555555u;
    return v;
}

inline uint32_t compactBits(uint32_t v) {
    v &= 0x55555555u;
    v = (v | (v >> 1)) & 0x33333333u;
    v = (v | (v >> 2)) & 0x0F0F0F0Fu;
    v = (v | (v >> 4)) & 0x00FF00FFu;
    v = (v | (v >> 8)) & 0x0000FFFFu;
    return v;
}

inline uint32_t mortonEncode(uint32_t x, uint32_t y) {
    return expandBits16(x) | (expandBits16(y) << 1);
}

// Number of 2D multi-indices with total degree <= q, and the flat index of
// one such multi-index. Degrees are laid out in blocks: (0,0) | (1,0) (0,1) |
// (2,0) (1,1) (0,2) | ...
inline int triCount(int q) { return (q + 1) * (q + 2) / 2; }
inline int triIndex(int a, int b) { int d = a + b; return d * (d + 1) / 2 + b; }

inline uint32_t keyAtLevel(uint32_t code, int level, int max_depth) {
    int shift = 2 * (max_depth - level);
    return shift >= 32 ? 0u : (code >> shift);
}

// Taylor coefficients T_gamma = d^gamma K(R) / gamma! of the softened kernel
// K(r) = 1/sqrt(|r|^2 + eps^2), expanded about R, up to total degree q.
//
// With u(d) = |R+d|^2 + eps^2 = A + B and B(0) = 0, the binomial series
// (A+B)^(-1/2) = A^(-1/2) * sum_k C(-1/2,k) (B/A)^k is a formal power series
// in d whose k-th term has minimum degree k. Truncating at k = q therefore
// reproduces the degree-q Taylor polynomial exactly -- no convergence
// condition, and B has only four terms so each power costs almost nothing.
void taylorSoftened(double Rx, double Ry, double eps_sq, int q, std::vector<double>& T) {
    const int n = triCount(q);
    T.assign(n, 0.0);

    const double A = Rx * Rx + Ry * Ry + eps_sq;
    const double inv_A = 1.0 / A;
    const double base = 1.0 / std::sqrt(A);

    std::vector<double> P(n, 0.0), Q(n, 0.0);
    P[triIndex(0, 0)] = 1.0;          // B^0
    T[triIndex(0, 0)] = base;         // k = 0 term

    double binom = 1.0;               // C(-1/2, k)
    double inv_A_pow = 1.0;           // A^-k

    for (int k = 1; k <= q; k++) {
        // Q = P * B, where B = 2Rx*x + 2Ry*y + x^2 + y^2.
        std::fill(Q.begin(), Q.end(), 0.0);
        for (int d = 0; d < q; d++) {
            for (int b = 0; b <= d; b++) {
                const double v = P[triIndex(d - b, b)];
                if (v == 0.0) continue;
                const int ax = d - b, ay = b;
                if (ax + 1 + ay <= q) Q[triIndex(ax + 1, ay)] += v * 2.0 * Rx;
                if (ax + ay + 1 <= q) Q[triIndex(ax, ay + 1)] += v * 2.0 * Ry;
                if (ax + 2 + ay <= q) Q[triIndex(ax + 2, ay)] += v;
                if (ax + ay + 2 <= q) Q[triIndex(ax, ay + 2)] += v;
            }
        }
        P.swap(Q);

        binom *= (-0.5 - double(k - 1)) / double(k);
        inv_A_pow *= inv_A;
        const double coef = base * binom * inv_A_pow;

        for (int i = 0; i < n; i++) T[i] += coef * P[i];
    }
}

}  // namespace

void FMMCPU::CellIndex::build(const std::vector<uint32_t>& keys) {
    size_t cap = 16;
    while (cap < keys.size() * 2 + 1) cap <<= 1;
    mask = uint32_t(cap - 1);
    slot_keys.assign(cap, EMPTY);
    slot_vals.assign(cap, -1);

    for (size_t i = 0; i < keys.size(); i++) {
        uint32_t h = hash(keys[i]) & mask;
        while (slot_keys[h] != EMPTY) h = (h + 1) & mask;
        slot_keys[h] = keys[i];
        slot_vals[h] = int(i);
    }
}

void FMMCPU::Level::clear() {
    keys.clear();
    body_start.clear();
    body_end.clear();
    parent.clear();
    child_begin.clear();
    child_end.clear();
}

int FMMCPU::autoDepth(int n, int leaf_capacity) {
    if (n <= 0 || leaf_capacity <= 0) return 2;
    // One extra level per 4x growth in particle count.
    int d = 2;
    double cells = 16.0;   // 4^2
    while (d < 12 && double(n) / cells > double(leaf_capacity)) {
        d++;
        cells *= 4.0;
    }
    return d;
}

FMMCPU::FMMCPU(std::unique_ptr<Preset> preset, FMMKernel kernel, int order,
               int leaf_capacity, float G, float softening)
    : preset(std::move(preset)), kernel(kernel),
      p(std::clamp(order, 1, MAX_ORDER)),
      leaf_capacity(std::max(1, leaf_capacity)),
      G(G), softening(softening) {
    max_level = autoDepth(this->preset->getParticleCount(), this->leaf_capacity);
    reset();
}

void FMMCPU::reset() {
    current_time = 0.0f;
    preset->apply(particles);
    setParticleCount(static_cast<int>(particles.size()));

    const size_t n = particles.size();
    pos_x.resize(n); pos_y.resize(n); mass.resize(n);
    acc_x.resize(n); acc_y.resize(n);
    codes.resize(n);
    sort_keys.resize(n);
    order_map.resize(n);

    max_level = autoDepth(static_cast<int>(n), leaf_capacity);
    rebuildTables();
}

void FMMCPU::setOrder(int new_p) {
    p = std::clamp(new_p, 1, kernel == FMMKernel::Softened ? 6 : MAX_ORDER);
    rebuildTables();
}

void FMMCPU::setDepth(int d) {
    max_level = std::clamp(d, FIRST_M2L_LEVEL, 12);
    rebuildTables();
}

void FMMCPU::setNearFieldRadius(int ws) {
    near_radius = std::clamp(ws, 1, 2);
    rebuildTables();
}

void FMMCPU::setLeafCapacity(int cap) {
    leaf_capacity = std::max(1, cap);
    max_level = autoDepth(getParticleCount(), leaf_capacity);
    rebuildTables();
}

int FMMCPU::cellCount() const {
    int total = 0;
    for (const Level& l : levels) total += static_cast<int>(l.size());
    return total;
}

void FMMCPU::rebuildTables() {
    if (kernel == FMMKernel::Softened) {
        p = std::clamp(p, 1, 6);          // 2p-order derivative tables get big
        stride = triCount(p);

        alpha_x.resize(stride);
        alpha_y.resize(stride);
        for (int d = 0; d <= p; d++) {
            for (int b = 0; b <= d; b++) {
                const int i = triIndex(d - b, b);
                alpha_x[i] = d - b;
                alpha_y[i] = b;
            }
        }
    } else {
        p = std::clamp(p, 1, MAX_ORDER);
        stride = p + 1;                   // a_0 .. a_p
    }

    const int bn = 2 * p + 3;
    binomial.assign(size_t(bn) * bn, 0.0);
    for (int n = 0; n < bn; n++) {
        binomial[size_t(n) * bn] = 1.0;
        for (int k = 1; k <= n; k++) {
            binomial[size_t(n) * bn + k] =
                binomial[size_t(n - 1) * bn + k - 1] + binomial[size_t(n - 1) * bn + k];
        }
    }

    levels.assign(max_level + 1, Level{});
    cart_multipole.assign(max_level + 1, {});
    cart_local.assign(max_level + 1, {});
    cx_multipole.assign(max_level + 1, {});
    cx_local.assign(max_level + 1, {});
    m2l_matrices.assign(max_level + 1, {});
}

// ---------------------------------------------------------------------------
// Tree
// ---------------------------------------------------------------------------

void FMMCPU::sortByMorton() {
    const int n = static_cast<int>(particles.size());
    if (n == 0) return;

    glm::vec2 lo = particles[0].position;
    glm::vec2 hi = lo;
    for (const Particle& part : particles) {
        lo = glm::min(lo, part.position);
        hi = glm::max(hi, part.position);
    }

    // FMM needs square cells so that a translation depends only on the integer
    // cell offset, so the root box is a square, not the tight AABB.
    const glm::vec2 extent = hi - lo;
    float side = std::max(std::max(extent.x, extent.y), 1e-6f) * 1.02f;
    root_min = 0.5f * (lo + hi) - glm::vec2(side * 0.5f);
    root_size = side;

    const glm::vec2 base = root_min;
    const float inv_side = 1.0f / side;

    pool.parallel_for(0, n, [&](int start, int end) {
        for (int i = start; i < end; i++) {
            glm::vec2 u = glm::clamp((particles[i].position - base) * inv_side, 0.0f, 0.999999f);
            const uint32_t code = mortonEncode(uint32_t(u.x * 65536.0f), uint32_t(u.y * 65536.0f));
            sort_keys[i] = CpuRadixSort::pack(code, uint32_t(i));
        }
    });

    sorter.sort(sort_keys, n, pool);

    pool.parallel_for(0, n, [&](int start, int end) {
        for (int i = start; i < end; i++) {
            const int id = int(CpuRadixSort::payload(sort_keys[i]));
            pos_x[i] = particles[id].position.x;
            pos_y[i] = particles[id].position.y;
            mass[i] = particles[id].mass;
            order_map[i] = id;
            codes[i] = CpuRadixSort::key(sort_keys[i]);
        }
    });
}

void FMMCPU::buildLevels() {
    const int n = getParticleCount();

    for (int l = 0; l <= max_level; l++) {
        Level& lv = levels[l];
        lv.clear();

        int i = 0;
        while (i < n) {
            const uint32_t key = keyAtLevel(codes[i], l, MAX_DEPTH);
            int j = i + 1;
            while (j < n && keyAtLevel(codes[j], l, MAX_DEPTH) == key) j++;
            lv.keys.push_back(key);
            lv.body_start.push_back(i);
            lv.body_end.push_back(j);
            i = j;
        }

        lv.parent.assign(lv.size(), -1);
        lv.child_begin.assign(lv.size(), 0);
        lv.child_end.assign(lv.size(), 0);
        lv.index.build(lv.keys);
    }

    // Children of a cell share its key prefix and the key arrays are sorted,
    // so each parent's children form one contiguous run.
    for (int l = 1; l <= max_level; l++) {
        Level& child = levels[l];
        Level& parent = levels[l - 1];

        int pi = 0;
        for (size_t ci = 0; ci < child.size(); ci++) {
            const uint32_t pkey = child.keys[ci] >> 2;
            while (pi + 1 < static_cast<int>(parent.size()) && parent.keys[pi] < pkey) pi++;
            child.parent[ci] = pi;

            if (parent.child_end[pi] == parent.child_begin[pi])
                parent.child_begin[pi] = static_cast<int>(ci);
            parent.child_end[pi] = static_cast<int>(ci) + 1;
        }
    }
}

glm::vec2 FMMCPU::cellCenter(int level, uint32_t key) const {
    const uint32_t cx = compactBits(key);
    const uint32_t cy = compactBits(key >> 1);
    const float h = cellSize(level);
    return root_min + glm::vec2((float(cx) + 0.5f) * h, (float(cy) + 0.5f) * h);
}

void FMMCPU::buildTree() {
    sortByMorton();
    buildLevels();

    for (int l = 0; l <= max_level; l++) {
        const size_t cells = levels[l].size();
        if (kernel == FMMKernel::Softened) {
            cart_multipole[l].assign(cells * stride, 0.0);
            cart_local[l].assign(cells * stride, 0.0);
        } else {
            cx_multipole[l].assign(cells * stride, std::complex<double>(0.0, 0.0));
            cx_local[l].assign(cells * stride, std::complex<double>(0.0, 0.0));
        }
    }
}

// ---------------------------------------------------------------------------
// Cartesian (softened inverse-square) expansions
// ---------------------------------------------------------------------------

void FMMCPU::buildM2LMatrices() {
    const int q = 2 * p;
    const double eps_sq = double(softening) * double(softening);
    const int mat_size = stride * stride;

    const int reach = m2lReach();
    const int span = m2lSpan();

    for (int l = FIRST_M2L_LEVEL; l <= max_level; l++) {
        m2l_matrices[l].assign(size_t(span) * span * mat_size, 0.0);
        const double h = double(cellSize(l));

        std::vector<double> T;
        for (int dy = -reach; dy <= reach; dy++) {
            for (int dx = -reach; dx <= reach; dx++) {
                if (std::abs(dx) <= near_radius && std::abs(dy) <= near_radius) continue;

                // R points from the source centre to the target centre.
                taylorSoftened(-double(dx) * h, -double(dy) * h, eps_sq, q, T);

                double* mat = &m2l_matrices[l][size_t((dy + reach) * span + (dx + reach)) * mat_size];
                for (int j = 0; j < stride; j++) {          // beta (target)
                    const int bx = alpha_x[j], by = alpha_y[j];
                    for (int i = 0; i < stride; i++) {      // alpha (source)
                        const int ax = alpha_x[i], ay = alpha_y[i];
                        mat[j * stride + i] = T[triIndex(ax + bx, ay + by)]
                                            * binom(ax + bx, ax) * binom(ay + by, ay);
                    }
                }
            }
        }
    }
}

void FMMCPU::cartP2M() {
    Level& leaf = levels[max_level];
    std::vector<double>& M = cart_multipole[max_level];
    const int cells = static_cast<int>(leaf.size());

    forEachCell(cells, [&](int start, int end) {
        double px[MAX_ORDER + 1], py[MAX_ORDER + 1];
        for (int c = start; c < end; c++) {
            const glm::vec2 center = cellCenter(max_level, leaf.keys[c]);
            double* dst = &M[size_t(c) * stride];

            for (int b = leaf.body_start[c]; b < leaf.body_end[c]; b++) {
                const double tx = double(pos_x[b]) - double(center.x);
                const double ty = double(pos_y[b]) - double(center.y);
                const double m = double(mass[b]);

                px[0] = py[0] = 1.0;
                for (int k = 1; k <= p; k++) { px[k] = px[k - 1] * tx; py[k] = py[k - 1] * ty; }

                for (int i = 0; i < stride; i++) {
                    const int ax = alpha_x[i], ay = alpha_y[i];
                    const double sign = ((ax + ay) & 1) ? -1.0 : 1.0;
                    dst[i] += m * sign * px[ax] * py[ay];
                }
            }
        }
    });
}

void FMMCPU::cartM2M(int level) {
    Level& parent = levels[level];
    Level& child = levels[level + 1];
    std::vector<double>& Mp = cart_multipole[level];
    const std::vector<double>& Mc = cart_multipole[level + 1];
    const int cells = static_cast<int>(parent.size());

    forEachCell(cells, [&](int start, int end) {
        double vx[MAX_ORDER + 1], vy[MAX_ORDER + 1];
        for (int c = start; c < end; c++) {
            const glm::vec2 pc = cellCenter(level, parent.keys[c]);
            double* dst = &Mp[size_t(c) * stride];

            for (int ci = parent.child_begin[c]; ci < parent.child_end[c]; ci++) {
                const glm::vec2 cc = cellCenter(level + 1, child.keys[ci]);
                const double dx = double(pc.x) - double(cc.x);
                const double dy = double(pc.y) - double(cc.y);

                vx[0] = vy[0] = 1.0;
                for (int k = 1; k <= p; k++) { vx[k] = vx[k - 1] * dx; vy[k] = vy[k - 1] * dy; }

                const double* src = &Mc[size_t(ci) * stride];
                for (int i = 0; i < stride; i++) {
                    const int ax = alpha_x[i], ay = alpha_y[i];
                    double acc = 0.0;
                    for (int gx = 0; gx <= ax; gx++) {
                        for (int gy = 0; gy <= ay; gy++) {
                            acc += binom(ax, gx) * binom(ay, gy)
                                 * src[triIndex(gx, gy)] * vx[ax - gx] * vy[ay - gy];
                        }
                    }
                    dst[i] += acc;
                }
            }
        }
    });
}

void FMMCPU::cartM2L(int level) {
    Level& lv = levels[level];
    const std::vector<double>& M = cart_multipole[level];
    std::vector<double>& L = cart_local[level];
    const std::vector<double>& mats = m2l_matrices[level];
    const int cells = static_cast<int>(lv.size());
    const int mat_size = stride * stride;
    const uint32_t grid = 1u << level;
    const int ws = near_radius;
    const int reach = m2lReach();
    const int span = m2lSpan();

    forEachCell(cells, [&](int start, int end) {
        for (int c = start; c < end; c++) {
            const uint32_t key = lv.keys[c];
            const int tx = int(compactBits(key));
            const int ty = int(compactBits(key >> 1));
            double* dst = &L[size_t(c) * stride];

            // The interaction list is the children of the parent's neighbours
            // that are not themselves neighbours of this cell. Enumerating the
            // integer offsets directly is equivalent and avoids materialising
            // the list.
            for (int dy = -reach; dy <= reach; dy++) {
                const int sy = ty + dy;
                if (sy < 0 || sy >= int(grid)) continue;

                for (int dx = -reach; dx <= reach; dx++) {
                    if (std::abs(dx) <= ws && std::abs(dy) <= ws) continue;

                    const int sx = tx + dx;
                    if (sx < 0 || sx >= int(grid)) continue;

                    // Only cells sharing a parent-neighbour are well separated
                    // at this level; the rest are handled one level up.
                    if (std::abs((sx >> 1) - (tx >> 1)) > ws) continue;
                    if (std::abs((sy >> 1) - (ty >> 1)) > ws) continue;

                    const int src = findCell(level, mortonEncode(uint32_t(sx), uint32_t(sy)));
                    if (src < 0) continue;

                    const double* mat = &mats[size_t((dy + reach) * span + (dx + reach)) * mat_size];
                    const double* m = &M[size_t(src) * stride];

                    for (int j = 0; j < stride; j++) {
                        double acc = 0.0;
                        for (int i = 0; i < stride; i++) acc += mat[j * stride + i] * m[i];
                        dst[j] += acc;
                    }
                }
            }
        }
    });
}

void FMMCPU::cartL2L(int level) {
    Level& child = levels[level + 1];
    const std::vector<double>& Lp = cart_local[level];
    std::vector<double>& Lc = cart_local[level + 1];
    const int cells = static_cast<int>(child.size());

    forEachCell(cells, [&](int start, int end) {
        double wx[MAX_ORDER + 1], wy[MAX_ORDER + 1];
        for (int c = start; c < end; c++) {
            const int pi = child.parent[c];
            const glm::vec2 pc = cellCenter(level, levels[level].keys[pi]);
            const glm::vec2 cc = cellCenter(level + 1, child.keys[c]);
            const double dx = double(cc.x) - double(pc.x);
            const double dy = double(cc.y) - double(pc.y);

            wx[0] = wy[0] = 1.0;
            for (int k = 1; k <= p; k++) { wx[k] = wx[k - 1] * dx; wy[k] = wy[k - 1] * dy; }

            const double* src = &Lp[size_t(pi) * stride];
            double* dst = &Lc[size_t(c) * stride];

            for (int j = 0; j < stride; j++) {
                const int gx = alpha_x[j], gy = alpha_y[j];
                double acc = 0.0;
                for (int bx = gx; bx + gy <= p; bx++) {
                    for (int by = gy; bx + by <= p; by++) {
                        acc += binom(bx, gx) * binom(by, gy)
                             * src[triIndex(bx, by)] * wx[bx - gx] * wy[by - gy];
                    }
                }
                dst[j] += acc;
            }
        }
    });
}

void FMMCPU::cartL2P() {
    Level& leaf = levels[max_level];
    const std::vector<double>& L = cart_local[max_level];
    const int cells = static_cast<int>(leaf.size());
    const float g = G;

    forEachCell(cells, [&](int start, int end) {
        double sx_pow[MAX_ORDER + 2], sy_pow[MAX_ORDER + 2];
        for (int c = start; c < end; c++) {
            const glm::vec2 center = cellCenter(max_level, leaf.keys[c]);
            const double* src = &L[size_t(c) * stride];

            for (int b = leaf.body_start[c]; b < leaf.body_end[c]; b++) {
                const double sx = double(pos_x[b]) - double(center.x);
                const double sy = double(pos_y[b]) - double(center.y);

                sx_pow[0] = sy_pow[0] = 1.0;
                for (int k = 1; k <= p; k++) {
                    sx_pow[k] = sx_pow[k - 1] * sx;
                    sy_pow[k] = sy_pow[k - 1] * sy;
                }

                // phi(x) = sum_beta L_beta * s^beta, and a = G * grad(phi).
                double gx = 0.0, gy = 0.0;
                for (int i = 0; i < stride; i++) {
                    const int bx = alpha_x[i], by = alpha_y[i];
                    const double v = src[i];
                    if (bx > 0) gx += v * double(bx) * sx_pow[bx - 1] * sy_pow[by];
                    if (by > 0) gy += v * double(by) * sx_pow[bx] * sy_pow[by - 1];
                }

                acc_x[b] = float(gx) * g;
                acc_y[b] = float(gy) * g;
            }
        }
    });
}

// ---------------------------------------------------------------------------
// Complex (logarithmic) expansions -- classic Greengard-Rokhlin 2D FMM
// ---------------------------------------------------------------------------

void FMMCPU::cxP2M() {
    Level& leaf = levels[max_level];
    std::vector<std::complex<double>>& M = cx_multipole[max_level];
    const int cells = static_cast<int>(leaf.size());

    forEachCell(cells, [&](int start, int end) {
        for (int c = start; c < end; c++) {
            const glm::vec2 center = cellCenter(max_level, leaf.keys[c]);
            const std::complex<double> zc(center.x, center.y);
            std::complex<double>* dst = &M[size_t(c) * stride];

            for (int b = leaf.body_start[c]; b < leaf.body_end[c]; b++) {
                const std::complex<double> z(pos_x[b], pos_y[b]);
                const double q = double(mass[b]);
                const std::complex<double> d = z - zc;

                dst[0] += q;
                std::complex<double> pw = 1.0;
                for (int k = 1; k <= p; k++) {
                    pw *= d;
                    dst[k] -= q * pw / double(k);
                }
            }
        }
    });
}

void FMMCPU::cxM2M(int level) {
    Level& parent = levels[level];
    Level& child = levels[level + 1];
    std::vector<std::complex<double>>& Mp = cx_multipole[level];
    const std::vector<std::complex<double>>& Mc = cx_multipole[level + 1];
    const int cells = static_cast<int>(parent.size());

    forEachCell(cells, [&](int start, int end) {
        std::vector<std::complex<double>> dpow(size_t(p) + 1);
        for (int c = start; c < end; c++) {
            const glm::vec2 pc = cellCenter(level, parent.keys[c]);
            std::complex<double>* dst = &Mp[size_t(c) * stride];

            for (int ci = parent.child_begin[c]; ci < parent.child_end[c]; ci++) {
                const glm::vec2 cc = cellCenter(level + 1, child.keys[ci]);
                const std::complex<double> d(double(cc.x) - double(pc.x),
                                             double(cc.y) - double(pc.y));

                dpow[0] = 1.0;
                for (int k = 1; k <= p; k++) dpow[k] = dpow[k - 1] * d;

                const std::complex<double>* src = &Mc[size_t(ci) * stride];
                dst[0] += src[0];

                for (int l = 1; l <= p; l++) {
                    std::complex<double> acc = -src[0] * dpow[l] / double(l);
                    for (int k = 1; k <= l; k++)
                        acc += src[k] * dpow[l - k] * binom(l - 1, k - 1);
                    dst[l] += acc;
                }
            }
        }
    });
}

void FMMCPU::cxM2L(int level) {
    Level& lv = levels[level];
    const std::vector<std::complex<double>>& M = cx_multipole[level];
    std::vector<std::complex<double>>& L = cx_local[level];
    const int cells = static_cast<int>(lv.size());
    const uint32_t grid = 1u << level;
    const double h = double(cellSize(level));
    const int ws = near_radius;
    const int reach = m2lReach();

    forEachCell(cells, [&](int start, int end) {
        std::vector<std::complex<double>> inv_pow(size_t(p) + 2);
        for (int c = start; c < end; c++) {
            const uint32_t key = lv.keys[c];
            const int tx = int(compactBits(key));
            const int ty = int(compactBits(key >> 1));
            std::complex<double>* dst = &L[size_t(c) * stride];

            for (int dy = -reach; dy <= reach; dy++) {
                const int sy = ty + dy;
                if (sy < 0 || sy >= int(grid)) continue;

                for (int dx = -reach; dx <= reach; dx++) {
                    if (std::abs(dx) <= ws && std::abs(dy) <= ws) continue;

                    const int sx = tx + dx;
                    if (sx < 0 || sx >= int(grid)) continue;
                    if (std::abs((sx >> 1) - (tx >> 1)) > ws) continue;
                    if (std::abs((sy >> 1) - (ty >> 1)) > ws) continue;

                    const int src_cell = findCell(level, mortonEncode(uint32_t(sx), uint32_t(sy)));
                    if (src_cell < 0) continue;

                    // z0 = source centre - target centre, exactly on the grid.
                    const std::complex<double> z0(double(dx) * h, double(dy) * h);
                    const std::complex<double> inv_z0 = 1.0 / z0;

                    inv_pow[0] = 1.0;
                    for (int k = 1; k <= p + 1; k++) inv_pow[k] = inv_pow[k - 1] * inv_z0;

                    const std::complex<double>* a = &M[size_t(src_cell) * stride];

                    // b_0 only shifts the potential by a constant and never
                    // reaches the force, so it is skipped (which also avoids a
                    // complex log per interaction).
                    for (int l = 1; l <= p; l++) {
                        std::complex<double> acc = -a[0] * inv_pow[l] / double(l);
                        std::complex<double> inner(0.0, 0.0);
                        for (int k = 1; k <= p; k++) {
                            const double sign = (k & 1) ? -1.0 : 1.0;
                            inner += a[k] * inv_pow[k] * (sign * binom(k + l - 1, k - 1));
                        }
                        dst[l] += acc + inner * inv_pow[l];
                    }
                }
            }
        }
    });
}

void FMMCPU::cxL2L(int level) {
    Level& child = levels[level + 1];
    const std::vector<std::complex<double>>& Lp = cx_local[level];
    std::vector<std::complex<double>>& Lc = cx_local[level + 1];
    const int cells = static_cast<int>(child.size());

    forEachCell(cells, [&](int start, int end) {
        std::vector<std::complex<double>> dpow(size_t(p) + 1);
        for (int c = start; c < end; c++) {
            const int pi = child.parent[c];
            const glm::vec2 pc = cellCenter(level, levels[level].keys[pi]);
            const glm::vec2 cc = cellCenter(level + 1, child.keys[c]);
            const std::complex<double> d(double(cc.x) - double(pc.x),
                                         double(cc.y) - double(pc.y));

            dpow[0] = 1.0;
            for (int k = 1; k <= p; k++) dpow[k] = dpow[k - 1] * d;

            const std::complex<double>* src = &Lp[size_t(pi) * stride];
            std::complex<double>* dst = &Lc[size_t(c) * stride];

            for (int m = 1; m <= p; m++) {
                std::complex<double> acc(0.0, 0.0);
                for (int l = m; l <= p; l++)
                    acc += src[l] * dpow[l - m] * binom(l, m);
                dst[m] += acc;
            }
        }
    });
}

void FMMCPU::cxL2P() {
    Level& leaf = levels[max_level];
    const std::vector<std::complex<double>>& L = cx_local[max_level];
    const int cells = static_cast<int>(leaf.size());
    const float g = G;

    forEachCell(cells, [&](int start, int end) {
        for (int c = start; c < end; c++) {
            const glm::vec2 center = cellCenter(max_level, leaf.keys[c]);
            const std::complex<double> zc(center.x, center.y);
            const std::complex<double>* src = &L[size_t(c) * stride];

            for (int b = leaf.body_start[c]; b < leaf.body_end[c]; b++) {
                const std::complex<double> w = std::complex<double>(pos_x[b], pos_y[b]) - zc;

                // Phi'(z) = sum_{l>=1} l * b_l * w^(l-1)
                std::complex<double> deriv(0.0, 0.0);
                std::complex<double> pw = 1.0;
                for (int l = 1; l <= p; l++) {
                    deriv += double(l) * src[l] * pw;
                    pw *= w;
                }

                // phi_real = G * sum m log r, and grad(Re f) = (Re f', -Im f'),
                // so the attractive acceleration is -G * (Re Phi', -Im Phi').
                acc_x[b] = float(-g * deriv.real());
                acc_y[b] = float(g * deriv.imag());
            }
        }
    });
}

// ---------------------------------------------------------------------------
// Passes
// ---------------------------------------------------------------------------

void FMMCPU::upwardPass() {
    if (kernel == FMMKernel::Softened) {
        cartP2M();
        for (int l = max_level - 1; l >= FIRST_M2L_LEVEL; l--) cartM2M(l);
    } else {
        cxP2M();
        for (int l = max_level - 1; l >= FIRST_M2L_LEVEL; l--) cxM2M(l);
    }
}

void FMMCPU::interactionPass() {
    if (kernel == FMMKernel::Softened) {
        buildM2LMatrices();
        for (int l = FIRST_M2L_LEVEL; l <= max_level; l++) cartM2L(l);
    } else {
        for (int l = FIRST_M2L_LEVEL; l <= max_level; l++) cxM2L(l);
    }
}

void FMMCPU::downwardPass() {
    if (kernel == FMMKernel::Softened) {
        for (int l = FIRST_M2L_LEVEL; l < max_level; l++) cartL2L(l);
        cartL2P();
    } else {
        for (int l = FIRST_M2L_LEVEL; l < max_level; l++) cxL2L(l);
        cxL2P();
    }
}

void FMMCPU::nearFieldPass() {
    Level& leaf = levels[max_level];
    const int cells = static_cast<int>(leaf.size());
    const uint32_t grid = 1u << max_level;
    const float g = G;
    const float eps_sq = softening * softening;
    const bool log_kernel = (kernel == FMMKernel::Logarithmic);
    const int ws = near_radius;

    forEachCell(cells, [&](int start, int end) {
        for (int c = start; c < end; c++) {
            const uint32_t key = leaf.keys[c];
            const int tx = int(compactBits(key));
            const int ty = int(compactBits(key >> 1));

            for (int dy = -ws; dy <= ws; dy++) {
                const int sy = ty + dy;
                if (sy < 0 || sy >= int(grid)) continue;

                for (int dx = -ws; dx <= ws; dx++) {
                    const int sx = tx + dx;
                    if (sx < 0 || sx >= int(grid)) continue;

                    const int src = (dx == 0 && dy == 0)
                        ? c
                        : findCell(max_level, mortonEncode(uint32_t(sx), uint32_t(sy)));
                    if (src < 0) continue;

                    const int js = leaf.body_start[src], je = leaf.body_end[src];

                    const float* sxp = pos_x.data();
                    const float* syp = pos_y.data();
                    const float* smp = mass.data();

                    for (int i = leaf.body_start[c]; i < leaf.body_end[c]; i++) {
                        const float pix = pos_x[i], piy = pos_y[i];
                        float ax = 0.0f, ay = 0.0f;

                        if (log_kernel) {
                            for (int j = js; j < je; j++) {
                                const float dx = sxp[j] - pix, dy = syp[j] - piy;
                                const float w = smp[j] / (dx * dx + dy * dy + eps_sq);
                                ax += dx * w;
                                ay += dy * w;
                            }
                        } else {
                            for (int j = js; j < je; j++) {
                                const float dx = sxp[j] - pix, dy = syp[j] - piy;
                                const float r2 = dx * dx + dy * dy + eps_sq;
                                const float inv = 1.0f / std::sqrt(r2);
                                const float w = smp[j] * inv * inv * inv;
                                ax += dx * w;
                                ay += dy * w;
                            }
                        }

                        // Self-interaction contributes exactly zero (d = 0),
                        // matching brute force, which also does not skip it.
                        acc_x[i] += ax * g;
                        acc_y[i] += ay * g;
                    }
                }
            }
        }
    });
}

// ---------------------------------------------------------------------------
// Stepping
// ---------------------------------------------------------------------------

void FMMCPU::step(float dt) {
    if (particles.empty()) return;

    verletStep1(dt);

    buildTree();
    upwardPass();
    interactionPass();
    downwardPass();     // writes accelerations (far field)
    nearFieldPass();    // adds the direct near field

    const int n = static_cast<int>(particles.size());
    pool.parallel_for(0, n, [&](int start, int end) {
        for (int i = start; i < end; i++)
            particles[order_map[i]].acceleration = glm::vec2(acc_x[i], acc_y[i]);
    });

    verletStep2(dt);

    if (boundary.active()) {
        pool.parallel_for(0, n, [&](int a, int b) { applyBoundaryRange(a, b, dt); });
    }

    current_time += dt;
}

void FMMCPU::verletStep1(float dt) {
    const int n = static_cast<int>(particles.size());
    pool.parallel_for(0, n, [&](int start, int end) {
        for (int i = start; i < end; i++) {
            particles[i].prev_acceleration = particles[i].acceleration;
            particles[i].position += particles[i].velocity * dt
                                   + 0.5f * particles[i].acceleration * dt * dt;
        }
    });
}

void FMMCPU::verletStep2(float dt) {
    const int n = static_cast<int>(particles.size());
    pool.parallel_for(0, n, [&](int start, int end) {
        for (int i = start; i < end; i++)
            particles[i].velocity += 0.5f * (particles[i].prev_acceleration
                                           + particles[i].acceleration) * dt;
    });
}
