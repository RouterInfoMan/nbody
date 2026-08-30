#pragma once
#include <string>
#include <GL/glew.h>
#include <glm/glm.hpp>

class Shader {
public:
    Shader(const std::string& vert_path, const std::string& frag_path);
    Shader(const std::string& compute_path);
    ~Shader();

    Shader(const Shader&) = delete;
    Shader& operator=(const Shader&) = delete;

    void use() const;
    GLuint id() const { return program; }

    void set_mat4(const std::string& name, const glm::mat4& mat) const;
    void set_float(const std::string& name, float value) const;
    void set_int(const std::string& name, int value) const;
    void set_vec2(const std::string& name, float x, float y) const;

private:
    GLuint program;
    GLuint compile(const std::string& source, GLenum type);
    std::string read_file(const std::string& path);
};
