#pragma once
#include <cmath>
#include <glm/glm.hpp>

// Optional containment applied after each integration step, so the system can
// be held together instead of slowly evaporating as escapers fly off.
enum BoundaryMode {
    BOUNDARY_OFF = 0,
    BOUNDARY_BOUNCE,   // hard wall: reflect the outward velocity component
    BOUNDARY_DRAG,     // soft absorbing layer: damp velocity beyond the wall
    BOUNDARY_MODE_COUNT
};

enum BoundaryShape {
    BOUNDARY_CIRCLE = 0,
    BOUNDARY_BOX,
    BOUNDARY_SHAPE_COUNT
};

struct Boundary {
    int mode = BOUNDARY_OFF;
    int shape = BOUNDARY_CIRCLE;

    // Circle radius, or box half-extent.
    float radius = 600.0f;

    // Fraction of the normal velocity kept on impact. 1 is perfectly elastic,
    // 0 makes the wall completely absorbing.
    float restitution = 0.6f;

    // Damping rate outside the wall, per unit time, scaled by how far out the
    // particle is. Zero at the wall itself so nothing discontinuous happens
    // to particles skimming it.
    float drag = 4.0f;

    bool active() const { return mode != BOUNDARY_OFF; }
};

// Kept in one place because shaders/boundary.comp mirrors it exactly; the CPU
// and GPU solvers have to agree or switching algorithms would change the
// physics.
inline void applyBoundary(glm::vec2& pos, glm::vec2& vel, const Boundary& b, float dt) {
    if (b.mode == BOUNDARY_OFF || b.radius <= 0.0f) return;

    const float R = b.radius;

    if (b.shape == BOUNDARY_CIRCLE) {
        const float r = glm::length(pos);
        if (r <= R || r < 1e-9f) return;

        const glm::vec2 n = pos / r;
        if (b.mode == BOUNDARY_BOUNCE) {
            // Clamp just inside rather than mirroring the overshoot: a
            // particle that crossed by more than the wall radius in one step
            // would otherwise be reflected past the centre.
            pos = n * (R * 0.999f);
            const float vn = glm::dot(vel, n);
            if (vn > 0.0f) vel -= (1.0f + b.restitution) * vn * n;
        } else {
            vel *= std::exp(-b.drag * ((r - R) / R) * dt);
        }
        return;
    }

    if (b.mode == BOUNDARY_BOUNCE) {
        for (int i = 0; i < 2; i++) {
            if (pos[i] > R) {
                pos[i] = R * 0.999f;
                if (vel[i] > 0.0f) vel[i] = -vel[i] * b.restitution;
            } else if (pos[i] < -R) {
                pos[i] = -R * 0.999f;
                if (vel[i] < 0.0f) vel[i] = -vel[i] * b.restitution;
            }
        }
    } else {
        const float out = std::max(std::abs(pos.x), std::abs(pos.y)) - R;
        if (out > 0.0f) vel *= std::exp(-b.drag * (out / R) * dt);
    }
}
