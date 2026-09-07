# Cross-compile to a native Windows binary using MinGW-w64.
#
#   cmake -S . -B build-win -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-mingw64.cmake \
#         -DCMAKE_BUILD_TYPE=Release
#   cmake --build build-win -j
#
# The point is to get off Mesa's OpenGL-over-D3D12 translation layer. Under
# WSL that layer is what the GPU solvers actually run on, and it costs far more
# than the hardware does; a native .exe uses the vendor's own OpenGL driver.

set(CMAKE_SYSTEM_NAME Windows)
set(CMAKE_SYSTEM_PROCESSOR x86_64)

set(TOOLCHAIN_PREFIX x86_64-w64-mingw32)

# Debian and Ubuntu ship two threading variants of this toolchain. The default
# is usually win32, whose libstdc++ has no std::thread, std::mutex or
# std::condition_variable -- all of which the CPU solvers and the thread pool
# depend on. Prefer the posix variant, which has them; static linking means the
# winpthreads dependency does not follow the binary around.
find_program(MINGW_CXX NAMES ${TOOLCHAIN_PREFIX}-g++-posix ${TOOLCHAIN_PREFIX}-g++)
find_program(MINGW_CC  NAMES ${TOOLCHAIN_PREFIX}-gcc-posix ${TOOLCHAIN_PREFIX}-gcc)

if(NOT MINGW_CXX)
    message(FATAL_ERROR
        "MinGW-w64 not found. Install it with:\n"
        "  sudo apt install g++-mingw-w64-x86-64")
endif()

set(CMAKE_C_COMPILER   ${MINGW_CC})
set(CMAKE_CXX_COMPILER ${MINGW_CXX})
set(CMAKE_RC_COMPILER  ${TOOLCHAIN_PREFIX}-windres)

set(CMAKE_FIND_ROOT_PATH /usr/${TOOLCHAIN_PREFIX})

# Look for programs on the host, but headers and libraries only in the target
# sysroot -- otherwise CMake cheerfully finds the Linux GLFW and the link fails
# in a thoroughly confusing way.
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
