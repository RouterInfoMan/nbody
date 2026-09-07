# nbody

Interactive 2D N-body gravity simulation with five solvers — brute force
(CPU/GPU), Barnes-Hut (CPU/GPU), and a CPU fast multipole method — rendered
through an HDR pipeline with bloom and blackbody star colouring.

## Requirements

- A GPU and driver supporting **OpenGL 4.3** (compute shaders and SSBOs).
- CMake 3.16+ and a C++17 compiler.

On Ubuntu/Debian:

```sh
sudo apt install build-essential cmake libglfw3-dev libglew-dev libglm-dev libgl-dev
```

Dear ImGui is fetched automatically on the first `cmake` run if `external/imgui`
is missing, so that run needs `git` and network access.

## Build

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

Release is the default and matters: it enables `-O3 -march=native`, without
which the CPU solvers are several times slower.

To cross-compile a self-contained Windows `.exe` from Linux or WSL:

```sh
sudo apt install g++-mingw-w64-x86-64
cmake -S . -B build-win -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-mingw64.cmake
cmake --build build-win -j$(nproc)
```

## Run

```sh
./build/nbody
```

Shaders load relative to the working directory, so run from either the project
root or `build/`.

| Input | Action |
| --- | --- |
| Space | play / pause |
| Right mouse drag | pan |
| Scroll wheel, or Q / E | zoom |
| R | reframe the camera on the simulation |
| Esc | quit |

Everything else is in the control panel: solver and accuracy settings, preset
and its initial conditions, gravity and containment, and the rendering
controls. Barnes-Hut can also overlay the acceleration structure it is actually
using — quadtree cells on the CPU, LBVH node AABBs on the GPU.

## Other binaries

The build also produces development tools in `build/`:

```sh
cd build
./solver_check            # every solver against a direct N-body sum; --no-gpu to skip OpenGL
./benchmark               # time per step and force error vs N; --help for options
```
