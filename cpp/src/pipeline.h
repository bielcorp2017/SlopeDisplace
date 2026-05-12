#pragma once
#include "types.h"
#include <filesystem>
#include <string>
#include <vector>
#include <Eigen/Dense>

namespace pipeline {

struct Config {
    std::filesystem::path data_root;
    std::string dataset;
    std::string target_file;
    int target_points = 500000;
    std::vector<double> icp_voxels = {1.0, 0.4, 0.1, 0.05, 0.01, 0.005, 0.002};
    double fgr_voxel = 0.2;
};

// Convert PointCloudF (float32) to Open3D PointCloud (float64)
// for registration operations on downsampled clouds.
namespace detail {
    // Voxel-downsample a float32 cloud, returning an Open3D PointCloud.
    // Iteratively adjusts voxel size to hit target count.
    struct DownsampleResult {
        std::vector<float> xyz;   // downsampled float32
        std::vector<uint8_t> rgb; // downsampled rgb (if input has rgb)
        double voxel_size;
    };
    DownsampleResult voxel_downsample_to_target(const PointCloudF& input, int target);
}

// Run full preprocessing pipeline. Returns 0 on success.
int run(const Config& cfg, ProgressCallback progress = nullptr);

} // namespace pipeline
