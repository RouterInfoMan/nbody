# nbody

Interactive 2D N-body gravity simulation with five solvers — brute force
(CPU/GPU), Barnes-Hut (CPU/GPU), and a CPU fast multipole method — rendered
through an HDR pipeline with bloom and blackbody star colouring.

## Requirements

- A GPU and driver supporting **OpenGL 4.3** (compute shaders and SSBOs).
- CMake 3.16+ and a C++17 compiler.
- GLFW, GLEW, GLM, and OpenGL development headers.

On Ubuntu/Debian:

```sh
sudo apt install build-essential cmake libglfw3-dev libglew-dev libglm-dev libgl-dev
```

Dear ImGui is not vendored in the repository. If `external/imgui` is missing,
CMake fetches it (v1.91.8) during configuration, so the first `cmake` run needs
`git` and network access.

## Build

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

Release is the default if `CMAKE_BUILD_TYPE` is unset, and matters here: it
enables `-O3 -march=native`, without which the CPU solvers are several times
slower.

This produces three binaries in `build/`:

| Binary | Purpose |
| --- | --- |
| `nbody` | the interactive simulation |
| `solver_check` | correctness suite: every solver against a direct N-body sum, plus shader compilation, preset invariants, and CPU/GPU parity |
| `sweep` | accuracy-versus-cost sweeps used to pick solver defaults |

## Run

```sh
./build/nbody
```

Shaders are loaded relative to the working directory, so run from either the
project root or from `build/` — the build copies `shaders/` into `build/` on
every build, so both stay current.

```sh
cd build && ./solver_check     # expect "all checks passed"
cd build && ./sweep [N]        # N sets the particle count used for timings
```

## Controls

| Input | Action |
| --- | --- |
| Space | play / pause |
| Right mouse drag | pan |
| Scroll wheel, or Q / E | zoom |
| R | reframe the camera on the simulation |
| Esc | quit |

Everything else is in the control panel: solver and its accuracy parameters,
preset and its initial conditions, gravity and containment, and the rendering
controls. Sections are collapsible; the panel scrolls if it outgrows the
window.

## What is in the panel

- **Algorithm** — pick a solver. Barnes-Hut exposes the opening angle `theta`,
  a quadrupole toggle, and an overlay that draws the acceleration structure it
  actually uses: quadtree cells on the CPU, LBVH node AABBs on the GPU.
- **Preset** — Galaxy, Collapse, Binary Clouds, Cloud Cluster, each with its own
  radii, masses, orbital and dispersion parameters, and a seed.
- **Physics** — `G`, softening, time step, and an optional boundary that either
  bounces particles off a wall or damps them beyond it.
- **Appearance** — colour source and temperature range, brightness, exposure,
  bloom, saturation, and per-particle colour variety.
- **Performance / Accuracy** — timings, and a button to measure the current
  solver's force error against a direct sum.

## Layout

```
include/, src/    solvers, presets, renderer, camera
shaders/          GLSL: compute solvers, sorting, rendering, post-processing
tests/            solver_check and sweep
external/imgui/   fetched at configure time if absent
```
