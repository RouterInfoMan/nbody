#include "shader.h"
#include <fstream>
#include <sstream>
#include <stdexcept>

Shader::Shader(const std::string& vert_path, const std::string& frag_path) {
    GLuint vs = compile(read_file(vert_path), GL_VERTEX_SHADER);
    GLuint fs = compile(read_file(frag_path), GL_FRAGMENT_SHADER);

    program = glCreateProgram();
    glAttachShader(program, vs);
    glAttachShader(program, fs);
    glLinkProgram(program);

    GLint ok;
    glGetProgramiv(program, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[512];
        glGetProgramInfoLog(program, 512, nullptr, log);
        glDeleteShader(vs);
        glDeleteShader(fs);
        throw std::runtime_error(std::string("Shader link failed: ") + log);
    }

    glDeleteShader(vs);
    glDeleteShader(fs);
}

Shader::Shader(const std::string& compute_path) {
    GLuint cs = compile(read_file(compute_path), GL_COMPUTE_SHADER);

    program = glCreateProgram();
    glAttachShader(program, cs);
    glLinkProgram(program);

    GLint ok;
    glGetProgramiv(program, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[512];
        glGetProgramInfoLog(program, 512, nullptr, log);
        glDeleteShader(cs);
        throw std::runtime_error(std::string("Compute shader link failed: ") + log);
    }

    glDeleteShader(cs);
}

Shader::~Shader() {
    glDeleteProgram(program);
}

void Shader::use() const { glUseProgram(program); }

void Shader::set_mat4(const std::string& name, const glm::mat4& mat) const {
    glUniformMatrix4fv(glGetUniformLocation(program, name.c_str()), 1, GL_FALSE, &mat[0][0]);
}

void Shader::set_vec2(const std::string& name, float x, float y) const {
    glUniform2f(glGetUniformLocation(program, name.c_str()), x, y);
}

void Shader::set_vec3(const std::string& name, float x, float y, float z) const {
    glUniform3f(glGetUniformLocation(program, name.c_str()), x, y, z);
}

void Shader::set_float(const std::string& name, float value) const {
    glUniform1f(glGetUniformLocation(program, name.c_str()), value);
}

void Shader::set_int(const std::string& name, int value) const {
    glUniform1i(glGetUniformLocation(program, name.c_str()), value);
}

GLuint Shader::compile(const std::string& source, GLenum type) {
    GLuint shader = glCreateShader(type);
    const char* src = source.c_str();
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);

    GLint ok;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[512];
        glGetShaderInfoLog(shader, 512, nullptr, log);
        const char* type_name = type == GL_VERTEX_SHADER ? "vertex" :
                                type == GL_FRAGMENT_SHADER ? "fragment" : "compute";
        throw std::runtime_error(std::string(type_name) + " shader compile failed: " + log);
    }
    return shader;
}

std::string Shader::read_file(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open())
        throw std::runtime_error("Failed to open shader: " + path);
    std::stringstream buf;
    buf << file.rdbuf();
    return buf.str();
}
