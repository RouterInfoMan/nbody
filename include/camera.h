#pragma once
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

class Camera2D {
public:
    Camera2D(float width = 1280.0f, float height = 720.0f);

    void move(float dx, float dy);
    void zoom(float factor);
    void setScreenSize(float width, float height);

    void setHomeRadius(float radius, bool recenter = true);
    void reset();

    glm::mat4 getProjectionMatrix() const;
    glm::vec2 getPosition() const { return position; }
    float getZoom() const { return zoom_level; }

private:
    glm::vec2 position{0.0f};
    float zoom_level = 1.0f;
    // Radius the default view height of 100 world units happens to frame.
    float home_radius = 43.0f;
    float screen_width, screen_height;
    float view_width, view_height;

    void updateViewSize();
};
