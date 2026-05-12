#pragma once
#include <cstdint>
#include <cstddef>
#include <vector>
#include <string>
#include <functional>

// Bulk point cloud stored as float32 for memory efficiency.
// 120M points: float32 = 1.44 GB vs float64 = 2.88 GB.
struct PointCloudF {
    std::vector<float> xyz;      // interleaved [x0,y0,z0, x1,y1,z1, ...], N*3
    std::vector<uint8_t> rgb;    // interleaved [r0,g0,b0, ...], N*3 (optional)
    std::vector<float> normals;  // interleaved [nx,ny,nz, ...], N*3 (optional)

    size_t size() const { return xyz.size() / 3; }
    bool has_rgb() const { return rgb.size() == xyz.size(); }
    bool has_normals() const { return normals.size() == xyz.size(); }
};

// Progress callback: stage name, detail string, progress fraction [0,1]
using ProgressCallback = std::function<void(const std::string& stage,
                                            const std::string& detail,
                                            float progress)>;

struct ICPScaleResult {
    double voxel;
    double fitness;
    double inlier_rmse;
    int src_count;
    int tgt_count;
};
