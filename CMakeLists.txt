cmake_minimum_required(VERSION 3.16)
project(nbody VERSION 2.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-g -Wall -Wextra")

if(CMAKE_CROSSCOMPILING)
    set(NBODY_NATIVE_ARCH_DEFAULT OFF)
else()
    set(NBODY_NATIVE_ARCH_DEFAULT ON)
endif()
option(NBODY_NATIVE_ARCH "Tune for the building CPU (not portable)"
       ${NBODY_NATIVE_ARCH_DEFAULT})

if(NBODY_NATIVE_ARCH)
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")
else()
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=x86-64-v3")
endif()

option(NBODY_BUNDLE_DEPS "Fetch and statically link GLFW/GLEW/GLM" ${WIN32})

find_package(OpenGL REQUIRED)

if(NBODY_BUNDLE_DEPS)
    include(FetchContent)
    set(FETCHCONTENT_QUIET OFF)

    set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(glfw
        GIT_REPOSITORY https://github.com/glfw/glfw.git
        GIT_TAG 3.4
        GIT_SHALLOW TRUE
    )

    FetchContent_Declare(glm
        GIT_REPOSITORY https://github.com/g-truc/glm.git
        GIT_TAG 1.0.1
        GIT_SHALLOW TRUE
    )

    set(glew-cmake_BUILD_SHARED OFF CACHE BOOL "" FORCE)
    set(ONLY_LIBS ON CACHE BOOL "" FORCE)
    FetchContent_Declare(glew
        URL https://github.com/Perlmint/glew-cmake/archive/refs/tags/glew-cmake-2.2.0.tar.gz
    )

    FetchContent_MakeAvailable(glfw glm glew)

    add_library(nbody_deps INTERFACE)
    target_link_libraries(nbody_deps INTERFACE glfw libglew_static glm::glm OpenGL::GL)
    target_compile_definitions(nbody_deps INTERFACE GLEW_STATIC)
    if(MINGW)
        target_link_options(nbody_deps INTERFACE -static -static-libgcc -static-libstdc++)
    else()
        target_link_options(nbody_deps INTERFACE -static-libgcc -static-libstdc++)
    endif()
else()
    find_package(glfw3 REQUIRED)
    find_package(GLEW REQUIRED)
    find_package(glm REQUIRED)

    add_library(nbody_deps INTERFACE)
    target_link_libraries(nbody_deps INTERFACE OpenGL::GL glfw GLEW::GLEW glm::glm)
endif()

# ImGui via FetchContent
set(IMGUI_DIR ${CMAKE_SOURCE_DIR}/external/imgui)
if(NOT EXISTS ${IMGUI_DIR}/imgui.cpp)
    include(FetchContent)
    FetchContent_Declare(imgui
        GIT_REPOSITORY https://github.com/ocornut/imgui.git
        GIT_TAG v1.91.8
        SOURCE_DIR ${IMGUI_DIR}
    )
    FetchContent_MakeAvailable(imgui)
endif()

set(IMGUI_SOURCES
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
    ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
)

# Main executable
add_executable(nbody
    src/main.cpp
    src/shader.cpp
    src/camera.cpp
    src/renderer.cpp
    src/brute_force_cpu.cpp
    src/brute_force_gpu.cpp
    src/barnes_hut_cpu.cpp
    src/barnes_hut_gpu.cpp
    src/gpu_radix_sort.cpp
    src/fmm_cpu.cpp
    src/galaxy_preset.cpp
    src/collapse_preset.cpp
    src/cloud_presets.cpp
    ${IMGUI_SOURCES}
)

target_include_directories(nbody PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${IMGUI_DIR}
    ${IMGUI_DIR}/backends
)

target_link_libraries(nbody PRIVATE nbody_deps)

# Correctness harness: checks every tree solver against a direct N-body sum.
add_executable(solver_check
    tests/solver_check.cpp
    src/shader.cpp
    src/camera.cpp
    src/renderer.cpp
    src/brute_force_cpu.cpp
    src/brute_force_gpu.cpp
    src/barnes_hut_cpu.cpp
    src/barnes_hut_gpu.cpp
    src/gpu_radix_sort.cpp
    src/fmm_cpu.cpp
    src/galaxy_preset.cpp
    src/collapse_preset.cpp
    src/cloud_presets.cpp
)

target_include_directories(solver_check PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(solver_check PRIVATE nbody_deps)

# Accuracy-vs-cost sweeps used to choose solver defaults.
add_executable(sweep
    tests/sweep.cpp
    src/shader.cpp
    src/camera.cpp
    src/renderer.cpp
    src/brute_force_cpu.cpp
    src/brute_force_gpu.cpp
    src/barnes_hut_cpu.cpp
    src/barnes_hut_gpu.cpp
    src/gpu_radix_sort.cpp
    src/fmm_cpu.cpp
    src/galaxy_preset.cpp
    src/collapse_preset.cpp
    src/cloud_presets.cpp
)

target_include_directories(sweep PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(sweep PRIVATE nbody_deps)

# Speedup benchmark across solver variants and backends.
add_executable(benchmark
    tests/benchmark.cpp
    src/shader.cpp
    src/camera.cpp
    src/renderer.cpp
    src/brute_force_cpu.cpp
    src/brute_force_gpu.cpp
    src/barnes_hut_cpu.cpp
    src/barnes_hut_gpu.cpp
    src/gpu_radix_sort.cpp
    src/fmm_cpu.cpp
    src/galaxy_preset.cpp
    src/collapse_preset.cpp
    src/cloud_presets.cpp
)

target_include_directories(benchmark PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(benchmark PRIVATE nbody_deps)

add_custom_target(copy_shaders ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_SOURCE_DIR}/shaders ${CMAKE_BINARY_DIR}/shaders
    COMMENT "Syncing shaders to build directory"
)
add_dependencies(nbody copy_shaders)
add_dependencies(solver_check copy_shaders)
add_dependencies(sweep copy_shaders)
add_dependencies(benchmark copy_shaders)

# Print config
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CXX: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
