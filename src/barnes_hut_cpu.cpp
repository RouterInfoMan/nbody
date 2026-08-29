#include "barnes_hut_cpu.h"
#include <algorithm>
#include <cmath>
#include <iostream>

BarnesHutCPU::BarnesHutCPU(std::unique_ptr<Preset> preset, float theta,
                           float G, float softening, int leaf_cap, int thread_cap)
    : preset(std::move(preset)), theta(theta), theta_sq(theta * theta),
      G(G), softening_sq(softening * softening),
      leaf_cap(leaf_cap), thread_cap(thread_cap) {
    reset();

    int n = this->preset->getParticleCount();
    size_t estimated = std::max(size_t(n * 6), size_t(100000));
    nodes.resize(estimated);
    parent_map.resize(estimated / 4);
    indices.resize(n);
    sorted.resize(n);
    morton_codes.resize(n);
}

void BarnesHutCPU::reset() {
    current_time = 0.0f;
    preset->apply(particles);
}

void BarnesHutCPU::step(float dt) {
    verletStep1(dt);
    sortByMorton();
    buildTree();
    computeForces();
    verletStep2(dt);
    current_time += dt;
}

void BarnesHutCPU::sortByMorton() {
    if (particles.empty()) return;

    bounds_min = bounds_max = particles[0].position;
    for (const auto& p : particles) {
        bounds_min = glm::min(bounds_min, p.position);
        bounds_max = glm::max(bounds_max, p.position);
    }

    glm::vec2 pad = (bounds_max - bounds_min) * 0.01f + glm::vec2(1e-6f);
    bounds_min -= pad;
    bounds_max += pad;

    int n = static_cast<int>(particles.size());

    pool.parallel_for(0, n, [&](int start, int end) {
        for (int i = start; i < end; i++) {
            morton_codes[i] = Morton::fromPosition(particles[i].position, bounds_min, bounds_max);
            indices[i] = i;
        }
    });

    std::sort(indices.begin(), indices.begin() + n,
              [this](int a, int b) { return morton_codes[a] < morton_codes[b]; });

    pool.parallel_for(0, n, [&](int start, int end) {
        for (int i = start; i < end; i++)
            sorted[i] = particles[indices[i]];
    });
}

template<typename Pred>
int BarnesHutCPU::partitionRange(int start, int end, Pred pred) {
    int i = start;
    for (int j = start; j < end; j++) {
        if (pred(sorted[j])) {
            if (i != j) std::swap(sorted[i], sorted[j]);
            i++;
        }
    }
    return i - start;
}

void BarnesHutCPU::clearTree() {
    node_count.store(0, std::memory_order_relaxed);
}

int BarnesHutCPU::subdivide(int node_idx, int range_start, int range_end) {
    glm::vec2 center = nodes[node_idx].quad.center;

    int split[5];
    split[0] = range_start;
    split[4] = range_end;

    split[2] = range_start + partitionRange(range_start, range_end,
        [center](const Particle& p) { return p.position.y < center.y; });
    split[1] = range_start + partitionRange(range_start, split[2],
        [center](const Particle& p) { return p.position.x < center.x; });
    split[3] = split[2] + partitionRange(split[2], range_end,
        [center](const Particle& p) { return p.position.x < center.x; });

    int idx = node_count.fetch_add(1, std::memory_order_relaxed);
    int children = idx * 4 + 1;

    if (static_cast<size_t>(children + 3) >= nodes.size()) {
        size_t new_size = nodes.size() + nodes.size() / 2;
        nodes.resize(new_size);
        parent_map.resize(new_size / 4);
    }

    parent_map[idx] = node_idx;
    nodes[node_idx].children = children;

    int nexts[4] = { children + 1, children + 2, children + 3, nodes[node_idx].next };
    for (int i = 0; i < 4; i++) {
        nodes[children + i] = QuadNode();
        nodes[children + i].next = nexts[i];
        nodes[children + i].quad = nodes[node_idx].quad.child(i);
        nodes[children + i].body_start = split[i];
        nodes[children + i].body_end = split[i + 1];
    }

    return children;
}

void BarnesHutCPU::propagate() {
    int len = node_count.load(std::memory_order_relaxed);

    for (int idx = len - 1; idx >= 0; idx--) {
        int node = parent_map[idx];
        int c = nodes[node].children;
        nodes[node].com = nodes[c].com + nodes[c+1].com + nodes[c+2].com + nodes[c+3].com;
        nodes[node].mass = nodes[c].mass + nodes[c+1].mass + nodes[c+2].mass + nodes[c+3].mass;
    }

    int total = len * 4 + 1;
    pool.parallel_for(0, total, [&](int start, int end) {
        for (int i = start; i < end; i++) {
            if (nodes[i].mass > 0.0f)
                nodes[i].com /= nodes[i].mass;
        }
    });
}

void BarnesHutCPU::buildTree() {
    clearTree();
    if (sorted.empty()) return;

    glm::vec2 sz = bounds_max - bounds_min;
    float side = std::max(sz.x, sz.y);
    glm::vec2 center = (bounds_min + bounds_max) * 0.5f;

    nodes[ROOT] = QuadNode();
    nodes[ROOT].quad = Quad(center, side);
    nodes[ROOT].body_start = 0;
    nodes[ROOT].body_end = static_cast<int>(sorted.size());
    nodes[ROOT].next = 0;

    std::vector<int> work, next_work;
    work.reserve(1024);
    next_work.reserve(1024);
    work.push_back(ROOT);

    while (!work.empty()) {
        next_work.clear();

        auto makeLeaf = [&](int idx) {
            glm::vec2 wpos(0.0f);
            float total_mass = 0.0f;
            for (int j = nodes[idx].body_start; j < nodes[idx].body_end; j++) {
                wpos += sorted[j].position * sorted[j].mass;
                total_mass += sorted[j].mass;
            }
            nodes[idx].com = wpos;
            nodes[idx].mass = total_mass;
        };

        auto shouldBeLeaf = [&](int idx) {
            return nodes[idx].bodyCount() <= leaf_cap
                || nodes[idx].quad.size < 1e-5f;
        };

        for (int node : work) {
            if (shouldBeLeaf(node)) {
                makeLeaf(node);
                continue;
            }

            if (nodes[node].bodyCount() >= thread_cap) {
                int c = subdivide(node, nodes[node].body_start, nodes[node].body_end);
                for (int k = 0; k < 4; k++) {
                    if (nodes[c + k].bodyCount() > 0)
                        next_work.push_back(c + k);
                }
            } else {
                std::vector<int> stack;
                stack.reserve(128);
                stack.push_back(node);

                while (!stack.empty()) {
                    int cur = stack.back();
                    stack.pop_back();

                    if (shouldBeLeaf(cur)) {
                        makeLeaf(cur);
                        continue;
                    }

                    int c = subdivide(cur, nodes[cur].body_start, nodes[cur].body_end);
                    for (int k = 0; k < 4; k++) {
                        if (nodes[c + k].bodyCount() > 0)
                            stack.push_back(c + k);
                    }
                }
            }
        }

        work.swap(next_work);
    }

    propagate();
}

glm::vec2 BarnesHutCPU::computeAcceleration(glm::vec2 pos) {
    glm::vec2 acc(0.0f);
    int node = ROOT;

    while (true) {
        const QuadNode& n = nodes[node];
        glm::vec2 d = n.com - pos;
        float d_sq = glm::dot(d, d);

        if (n.quad.size * n.quad.size < d_sq * theta_sq) {
            if (d_sq > 1e-10f) {
                float r_soft = std::sqrt(d_sq + softening_sq);
                float denom = r_soft * r_soft * r_soft;
                acc += d * (G * n.mass / denom);
            }
            if (n.next == 0) break;
            node = n.next;
        } else if (n.isLeaf()) {
            for (int i = n.body_start; i < n.body_end; i++) {
                glm::vec2 bd = sorted[i].position - pos;
                float bd_sq = glm::dot(bd, bd);
                if (bd_sq > 1e-10f) {
                    float r_soft = std::sqrt(bd_sq + softening_sq);
                    float denom = r_soft * r_soft * r_soft;
                    acc += bd * (G * sorted[i].mass / denom);
                }
            }
            if (n.next == 0) break;
            node = n.next;
        } else {
            node = n.children;
        }
    }

    return acc;
}

void BarnesHutCPU::computeForces() {
    int n = static_cast<int>(particles.size());
    pool.parallel_for(0, n, [&](int start, int end) {
        for (int i = start; i < end; i++)
            particles[i].acceleration = computeAcceleration(particles[i].position);
    });
}

void BarnesHutCPU::verletStep1(float dt) {
    int n = static_cast<int>(particles.size());
    pool.parallel_for(0, n, [&](int start, int end) {
        for (int i = start; i < end; i++) {
            particles[i].prev_acceleration = particles[i].acceleration;
            particles[i].position += particles[i].velocity * dt
                                   + 0.5f * particles[i].acceleration * dt * dt;
        }
    });
}

void BarnesHutCPU::verletStep2(float dt) {
    int n = static_cast<int>(particles.size());
    pool.parallel_for(0, n, [&](int start, int end) {
        for (int i = start; i < end; i++)
            particles[i].velocity += 0.5f * (particles[i].prev_acceleration
                                           + particles[i].acceleration) * dt;
    });
}
