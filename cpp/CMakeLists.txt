cmake_minimum_required(VERSION 3.20)
project(slope_preprocess LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Open3D (pre-built) ---
# Auto-detect extracted location or accept -DOpen3D_DIR=...
if(NOT Open3D_DIR)
    set(_O3D_LOCAL "${CMAKE_CURRENT_SOURCE_DIR}/open3d/open3d-devel-windows-amd64-0.19.0/CMake")
    if(EXISTS "${_O3D_LOCAL}")
        set(Open3D_DIR "${_O3D_LOCAL}")
    endif()
endif()
find_package(Open3D REQUIRED)

# --- OpenMP ---
find_package(OpenMP REQUIRED)

# --- Header-only deps via FetchContent ---
include(FetchContent)

FetchContent_Declare(nanoflann
    GIT_REPOSITORY https://github.com/jlblancoc/nanoflann.git
    GIT_TAG        v1.6.2)
FetchContent_Declare(nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3)
FetchContent_Declare(fast_float
    GIT_REPOSITORY https://github.com/fastfloat/fast_float.git
    GIT_TAG        v8.0.2)

FetchContent_MakeAvailable(nanoflann nlohmann_json fast_float)

add_executable(slope_preprocess
    src/main.cpp
    src/pipeline.cpp
    src/pts_loader.cpp
    src/displacement.cpp
    src/ply_writer.cpp
)

target_include_directories(slope_preprocess PRIVATE src)

target_link_libraries(slope_preprocess PRIVATE
    Open3D::Open3D
    OpenMP::OpenMP_CXX
    nanoflann::nanoflann
    nlohmann_json::nlohmann_json
    FastFloat::fast_float
)

if(MSVC)
    target_compile_definitions(slope_preprocess PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
    target_compile_options(slope_preprocess PRIVATE /O2 /fp:fast)
else()
    target_compile_options(slope_preprocess PRIVATE -O3 -march=native)
endif()
