#include "camera.h"

Camera2D::Camera2D(float width, float height)
    : screen_width(width), screen_height(height) {
    updateViewSize();
}

void Camera2D::move(float dx, float dy) {
    position.x += dx / zoom_level;
    position.y += dy / zoom_level;
}

void Camera2D::zoom(float factor) {
    zoom_level *= factor;
    zoom_level = glm::clamp(zoom_level, 0.001f, 100.0f);
    updateViewSize();
}

void Camera2D::setScreenSize(float width, float height) {
    screen_width = width;
    screen_height = height;
    updateViewSize();
}

void Camera2D::reset() {
    position = glm::vec2(0.0f);
    zoom_level = 1.0f;
    updateViewSize();
}

glm::mat4 Camera2D::getProjectionMatrix() const {
    float hw = view_width * 0.5f;
    float hh = view_height * 0.5f;
    return glm::ortho(position.x - hw, position.x + hw,
                      position.y - hh, position.y + hh, -1.0f, 1.0f);
}

void Camera2D::updateViewSize() {
    float aspect = screen_width / screen_height;
    view_height = 100.0f / zoom_level;
    view_width = view_height * aspect;
}
