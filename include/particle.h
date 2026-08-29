#pragma once
#include <glm/glm.hpp>

struct Particle {
    glm::vec2 position{0.0f};
    glm::vec2 velocity{0.0f};
    glm::vec2 acceleration{0.0f};
    glm::vec2 prev_acceleration{0.0f};
    float mass = 1.0f;

    Particle() = default;

    Particle(glm::vec2 pos, glm::vec2 vel, float m)
        : position(pos), velocity(vel), acceleration(0.0f),
          prev_acceleration(0.0f), mass(m) {}
};
