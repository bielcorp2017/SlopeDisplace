#pragma once
#include "types.h"
#include <filesystem>

// Load a raw scan PLY file via Open3D, returning float32 arrays.
PointCloudF load_scan(const std::filesystem::path& path,
                      bool with_rgb = false,
                      ProgressCallback progress = nullptr);
