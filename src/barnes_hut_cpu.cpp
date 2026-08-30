#include "barnes_hut_cpu.h"
#include <algorithm>
#include <cmath>

BarnesHutCPU::BarnesHutCPU(std::unique_ptr<Preset> preset, float theta,
                           float G, float softening, int leaf_cap, int split_threshold)
    : preset(std::move(preset)), theta(theta), inv_theta(1.0f / theta),
      G(G), softening_sq(softening * softening),
      leaf_cap(leaf_cap), split_threshold(split_threshold) {
    reset();
}

void BarnesHutCPU::reset() {
    current_time = 0.0f;
    preset->apply(particles);
    setParticleCount(static_cast<int>(particles.size()));

    const int n = getParticleCount();
    sort_keys.resize(n);
    sorted.resize(n);
    sorted_ids.resize(n);

    const size_t estimated = std::max(size_t(n) * 2, size_t(4096));
    nodes.resize(estimated);
    parent_map.resize(estimated / 4 + 1);
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

    const int n = static_cast<int>(particles.size());

    bounds_min = bounds_max = particles[0].position;
    for (const auto& p : particles) {
        bounds_min = glm::min(bounds_min, p.position);
        bounds_max = glm::max(bounds_max, p.position);
    }

    glm::vec2 pad = (bounds_max - bounds_min) * 0.01f + glm::vec2(1e-6f);
    bounds_min -= pad;
    bounds_max += pad;

    pool.parallel_for(0, n, [&](int start, int end) {
        for (int i = start; i < end; i++) {
            const uint32_t code = Morton::fromPosition(particles[i].position,
                                                       bounds_min, bounds_max);
            sort_keys[i] = CpuRadixSort::pack(code, uint32_t(i));
        }
    });

    sorter.sort(sort_keys, n, pool);

    pool.parallel_for(0, n, [&](int start, int end) {
        for (int i = start; i < end; i++) {
            const int id = int(CpuRadixSort::payload(sort_keys[i]));
            sorted[i] = particles[id];
            sorted_ids[i] = id;
        }
    });
}

template<typename Pred>
int BarnesHutCPU::partitionRange(int start, int end, Pred pred) {
    int i = start;
    for (int j = start; j < end; j++) {
        if (pred(sorted[j])) {
            if (i != j) {
                std::swap(sorted[i], sorted[j]);
                std::swap(sorted_ids[i], sorted_ids[j]);
            }
            i++;
        }
    }
    return i - start;
}

void BarnesHutCPU::clearTree() {
    node_count.store(0, std::memory_order_relaxed);
    node_overflow.store(false, std::memory_order_relaxed);
}

int BarnesHutCPU::subdivide(int node_idx, int range_start, int range_end) {
    const glm::vec2 center = nodes[node_idx].quad.center;

    int split[5];
    split[0] = range_start;
    split[4] = range_end;

    split[2] = range_start + partitionRange(range_start, range_end,
        [center](const Particle& p) { return p.position.y < center.y; });
    split[1] = range_start + partitionRange(range_start, split[2],
        [center](const Particle& p) { return p.position.x < center.x; });
    split[3] = split[2] + partitionRange(split[2], range_end,
        [center](const Particle& p) { return p.position.x < center.x; });

    const int idx = node_count.fetch_add(1, std::memory_order_relaxed);
    const int children = idx * 4 + 1;

    // Subtrees are expanded concurrently, so the node pool cannot be grown
    // here. Overflow is recorded and buildTree retries with a bigger pool.
    if (static_cast<size_t>(children + 3) >= nodes.size()) {
        node_overflow.store(true, std::memory_order_relaxed);
        return -1;
    }

    parent_map[idx] = node_idx;
    nodes[node_idx].children = children;

    const int nexts[4] = { children + 1, children + 2, children + 3, nodes[node_idx].next };
    for (int i = 0; i < 4; i++) {
        nodes[children + i] = QuadNode();
        nodes[children + i].next = nexts[i];
        nodes[children + i].quad = nodes[node_idx].quad.child(i);
        nodes[children + i].body_start = split[i];
        nodes[children + i].body_end = split[i + 1];
    }

    return children;
}

bool BarnesHutCPU::shouldBeLeaf(int node_idx) const {
    return nodes[node_idx].bodyCount() <= leaf_cap
        || nodes[node_idx].quad.size < 1e-5f;
}

void BarnesHutCPU::makeLeaf(int idx) {
    glm::vec2 wpos(0.0f);
    float total_mass = 0.0f;
    for (int j = nodes[idx].body_start; j < nodes[idx].body_end; j++) {
        wpos += sorted[j].position * sorted[j].mass;
        total_mass += sorted[j].mass;
    }

    const glm::vec2 com = total_mass > 0.0f ? wpos / total_mass : glm::vec2(0.0f);
    nodes[idx].com = com;
    nodes[idx].mass = total_mass;

    float qxx = 0.0f, qxy = 0.0f, qyy = 0.0f;
    for (int j = nodes[idx].body_start; j < nodes[idx].body_end; j++) {
        const glm::vec2 t = sorted[j].position - com;
        const float m = sorted[j].mass;
        qxx += m * t.x * t.x;
        qxy += m * t.x * t.y;
        qyy += m * t.y * t.y;
    }
    nodes[idx].qxx = qxx;
    nodes[idx].qxy = qxy;
    nodes[idx].qyy = qyy;
    nodes[idx].com_offset = total_mass > 0.0f
        ? glm::length(com - nodes[idx].quad.center) : 0.0f;
}

void BarnesHutCPU::expandSubtree(int root_node) {
    int stack[128];
    int sp = 0;
    stack[sp++] = root_node;

    while (sp > 0) {
        const int cur = stack[--sp];

        if (shouldBeLeaf(cur)) {
            makeLeaf(cur);
            continue;
        }

        const int c = subdivide(cur, nodes[cur].body_start, nodes[cur].body_end);
        if (c < 0) return;   // pool exhausted; buildTree will retry

        for (int k = 0; k < 4; k++) {
            if (nodes[c + k].bodyCount() > 0) {
                if (sp == 128) { makeLeaf(c + k); continue; }
                stack[sp++] = c + k;
            }
        }
    }
}

void BarnesHutCPU::propagate() {
    const int len = node_count.load(std::memory_order_relaxed);

    // parent_map is filled in subdivision order, and a node is always
    // subdivided before its children are, so walking it backwards visits every
    // child before its parent.
    for (int idx = len - 1; idx >= 0; idx--) {
        const int node = parent_map[idx];
        const int c = nodes[node].children;

        float mass = 0.0f;
        glm::vec2 wsum(0.0f);
        for (int k = 0; k < 4; k++) {
            mass += nodes[c + k].mass;
            wsum += nodes[c + k].com * nodes[c + k].mass;
        }

        const glm::vec2 com = mass > 0.0f ? wsum / mass : glm::vec2(0.0f);
        nodes[node].com = com;
        nodes[node].mass = mass;

        // Parallel axis theorem: shifting each child's second moment from its
        // own centre of mass to the parent's.
        float qxx = 0.0f, qxy = 0.0f, qyy = 0.0f;
        for (int k = 0; k < 4; k++) {
            const QuadNode& ch = nodes[c + k];
            const glm::vec2 s = ch.com - com;
            qxx += ch.qxx + ch.mass * s.x * s.x;
            qxy += ch.qxy + ch.mass * s.x * s.y;
            qyy += ch.qyy + ch.mass * s.y * s.y;
        }
        nodes[node].qxx = qxx;
        nodes[node].qxy = qxy;
        nodes[node].qyy = qyy;
        nodes[node].com_offset = mass > 0.0f
            ? glm::length(com - nodes[node].quad.center) : 0.0f;
    }
}

void BarnesHutCPU::buildTree() {
    if (sorted.empty()) return;

    const glm::vec2 sz = bounds_max - bounds_min;
    const float side = std::max(sz.x, sz.y);
    const glm::vec2 center = (bounds_min + bounds_max) * 0.5f;

    std::vector<int> frontier, next_frontier;
    frontier.reserve(1024);
    next_frontier.reserve(1024);

    // The parallel phase cannot grow the node pool, so on overflow the whole
    // build restarts against a larger one. In steady state this never fires.
    for (int attempt = 0; attempt < 8; attempt++) {
        clearTree();

        nodes[ROOT] = QuadNode();
        nodes[ROOT].quad = Quad(center, side);
        nodes[ROOT].body_start = 0;
        nodes[ROOT].body_end = static_cast<int>(sorted.size());
        nodes[ROOT].next = 0;

        frontier.clear();
        frontier.push_back(ROOT);

        // Split breadth-first while nodes are large, to expose enough
        // independent subtrees to keep every worker busy.
        const size_t target_tasks = size_t(pool.size()) * 4;
        bool overflowed = false;

        while (!overflowed) {
            bool any_split = false;
            next_frontier.clear();

            for (int node : frontier) {
                if (shouldBeLeaf(node)) {
                    makeLeaf(node);
                    continue;
                }
                if (frontier.size() >= target_tasks
                    && nodes[node].bodyCount() < split_threshold) {
                    next_frontier.push_back(node);   // hand to the parallel phase
                    continue;
                }

                const int c = subdivide(node, nodes[node].body_start, nodes[node].body_end);
                if (c < 0) { overflowed = true; break; }
                any_split = true;

                for (int k = 0; k < 4; k++)
                    if (nodes[c + k].bodyCount() > 0) next_frontier.push_back(c + k);
            }

            frontier.swap(next_frontier);
            if (!any_split || frontier.empty()) break;
        }

        if (!overflowed) {
            const int tasks = static_cast<int>(frontier.size());
            pool.parallel_for(0, tasks, [&](int a, int b) {
                for (int i = a; i < b; i++) expandSubtree(frontier[i]);
            });
        }

        if (!node_overflow.load(std::memory_order_relaxed)) {
            propagate();
            return;
        }

        nodes.resize(nodes.size() * 2);
        parent_map.resize(nodes.size() / 4 + 1);
    }
}

glm::vec2 BarnesHutCPU::computeAcceleration(glm::vec2 pos) const {
    glm::vec2 acc(0.0f);
    int node = ROOT;

    while (true) {
        const QuadNode& n = nodes[node];
        const glm::vec2 d = n.com - pos;
        const float d_sq = glm::dot(d, d);

        // Accept when d > size/theta + |com - cell centre|, squared to keep
        // the traversal free of square roots.
        const float limit = n.quad.size * inv_theta + n.com_offset;
        const bool accept = limit * limit < d_sq;

        if (accept || n.isLeaf()) {
            if (accept) {
                if (n.mass > 0.0f) {
                    const float u = d_sq + softening_sq;
                    const float inv = 1.0f / std::sqrt(u);
                    const float inv2 = inv * inv;
                    const float inv3 = inv2 * inv;
                    acc += d * (n.mass * inv3);

                    if (use_quadrupole) {
                        // a += G[ -3 (Qd) u^-5/2 - (3/2) tr(Q) d u^-5/2
                        //         + (15/2) d (d.Qd) u^-7/2 ]
                        const glm::vec2 qd(n.qxx * d.x + n.qxy * d.y,
                                           n.qxy * d.x + n.qyy * d.y);
                        const float dqd = d.x * qd.x + d.y * qd.y;
                        const float trq = n.qxx + n.qyy;
                        const float inv5 = inv3 * inv2;
                        const float inv7 = inv5 * inv2;
                        acc += qd * (-3.0f * inv5)
                             + d * (-1.5f * trq * inv5 + 7.5f * dqd * inv7);
                    }
                }
            } else {
                for (int i = n.body_start; i < n.body_end; i++) {
                    const glm::vec2 bd = sorted[i].position - pos;
                    const float bd_sq = glm::dot(bd, bd) + softening_sq;
                    const float inv = 1.0f / std::sqrt(bd_sq);
                    acc += bd * (sorted[i].mass * inv * inv * inv);
                }
            }
            if (n.next == 0) break;
            node = n.next;
        } else {
            node = n.children;
        }
    }

    return acc * G;
}

void BarnesHutCPU::computeForces() {
    const int n = static_cast<int>(particles.size());

    // Forces are evaluated at the Morton-sorted positions, where traversals of
    // neighbouring indices touch the same nodes, then scattered back.
    pool.parallel_for(0, n, [&](int start, int end) {
        for (int i = start; i < end; i++)
            particles[sorted_ids[i]].acceleration = computeAcceleration(sorted[i].position);
    });
}

void BarnesHutCPU::verletStep1(float dt) {
    const int n = static_cast<int>(particles.size());
    pool.parallel_for(0, n, [&](int start, int end) {
        for (int i = start; i < end; i++) {
            particles[i].prev_acceleration = particles[i].acceleration;
            particles[i].position += particles[i].velocity * dt
                                   + 0.5f * particles[i].acceleration * dt * dt;
        }
    });
}

void BarnesHutCPU::verletStep2(float dt) {
    const int n = static_cast<int>(particles.size());
    pool.parallel_for(0, n, [&](int start, int end) {
        for (int i = start; i < end; i++)
            particles[i].velocity += 0.5f * (particles[i].prev_acceleration
                                           + particles[i].acceleration) * dt;
    });
}
