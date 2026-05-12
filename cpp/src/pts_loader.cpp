#include "pts_loader.h"
#include <open3d/Open3D.h>
#include <stdexcept>
#include <algorithm>

PointCloudF load_scan(const std::filesystem::path& path, bool with_rgb,
                      ProgressCallback progress) {
    if (progress) progress("load", "reading " + path.filename().string(), 0.0f);

    auto pcd = open3d::io::CreatePointCloudFromFile(path.string());
    if (!pcd || pcd->points_.empty()) {
        throw std::runtime_error("Cannot read PLY: " + path.string());
    }

    size_t n = pcd->points_.size();
    if (progress) progress("load", std::to_string(n) + " points", 0.5f);

    PointCloudF cloud;
    cloud.xyz.resize(n * 3);
    for (size_t i = 0; i < n; ++i) {
        cloud.xyz[i*3+0] = static_cast<float>(pcd->points_[i][0]);
        cloud.xyz[i*3+1] = static_cast<float>(pcd->points_[i][1]);
        cloud.xyz[i*3+2] = static_cast<float>(pcd->points_[i][2]);
    }

    if (with_rgb && pcd->HasColors()) {
        cloud.rgb.resize(n * 3);
        for (size_t i = 0; i < n; ++i) {
            cloud.rgb[i*3+0] = static_cast<uint8_t>(std::clamp(pcd->colors_[i][0] * 255.0, 0.0, 255.0));
            cloud.rgb[i*3+1] = static_cast<uint8_t>(std::clamp(pcd->colors_[i][1] * 255.0, 0.0, 255.0));
            cloud.rgb[i*3+2] = static_cast<uint8_t>(std::clamp(pcd->colors_[i][2] * 255.0, 0.0, 255.0));
        }
    }

    if (progress) progress("load", "done", 1.0f);
    return cloud;
}
