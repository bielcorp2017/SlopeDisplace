#pragma once
#include "types.h"
#include <filesystem>

// Write binary PLY matching Open3D format (double XYZ + uchar RGB).
void write_binary_ply(const std::filesystem::path& path,
                      const float* xyz, size_t n,
                      const uint8_t* rgb = nullptr);
