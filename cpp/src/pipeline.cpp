#include "pipeline.h"
#include "pts_loader.h"
#include "displacement.h"
#include "ply_writer.h"

#include <open3d/Open3D.h>
#include <nlohmann/json.hpp>
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>

namespace fs = std::filesystem;
using json = nlohmann::json;
using namespace open3d;

namespace pipeline {

// ---------- helpers ----------

static double now_sec() {
    using C = std::chrono::high_resolution_clock;
    static auto t0 = C::now();
    return std::chrono::duration<double>(C::now() - t0).count();
}

static std::string fmt_count(size_t n) {
    // simple thousands separator
    auto s = std::to_string(n);
    std::string r;
    int c = 0;
    for (int i = (int)s.size() - 1; i >= 0; --i) {
        if (c && c % 3 == 0) r += ',';
        r += s[i]; ++c;
    }
    std::reverse(r.begin(), r.end());
    return r;
}

// Convert float32 cloud to Open3D PointCloud (float64)
static geometry::PointCloud to_o3d(const float* xyz, size_t n,
                                   const uint8_t* rgb = nullptr) {
    geometry::PointCloud pcd;
    pcd.points_.resize(n);
    for (size_t i = 0; i < n; ++i) {
        pcd.points_[i] = {xyz[i*3], xyz[i*3+1], xyz[i*3+2]};
    }
    if (rgb) {
        pcd.colors_.resize(n);
        for (size_t i = 0; i < n; ++i) {
            pcd.colors_[i] = {rgb[i*3] / 255.0, rgb[i*3+1] / 255.0, rgb[i*3+2] / 255.0};
        }
    }
    return pcd;
}

// Convert Open3D PointCloud back to float32 arrays
static void from_o3d(const geometry::PointCloud& pcd,
                     std::vector<float>& xyz, std::vector<uint8_t>& rgb) {
    size_t n = pcd.points_.size();
    xyz.resize(n * 3);
    for (size_t i = 0; i < n; ++i) {
        xyz[i*3+0] = static_cast<float>(pcd.points_[i][0]);
        xyz[i*3+1] = static_cast<float>(pcd.points_[i][1]);
        xyz[i*3+2] = static_cast<float>(pcd.points_[i][2]);
    }
    if (pcd.HasColors()) {
        rgb.resize(n * 3);
        for (size_t i = 0; i < n; ++i) {
            rgb[i*3+0] = static_cast<uint8_t>(std::clamp(pcd.colors_[i][0] * 255.0, 0.0, 255.0));
            rgb[i*3+1] = static_cast<uint8_t>(std::clamp(pcd.colors_[i][1] * 255.0, 0.0, 255.0));
            rgb[i*3+2] = static_cast<uint8_t>(std::clamp(pcd.colors_[i][2] * 255.0, 0.0, 255.0));
        }
    }
}

// ---------- Downsample ----------

detail::DownsampleResult detail::voxel_downsample_to_target(
        const PointCloudF& input, int target) {
    auto pcd = to_o3d(input.xyz.data(), input.size(),
                      input.has_rgb() ? input.rgb.data() : nullptr);
    auto bb = pcd.GetAxisAlignedBoundingBox();
    auto ext = bb.GetExtent();
    double volume = std::max(ext[0] * ext[1] * ext[2], 1e-6);
    double voxel = std::pow(volume / target, 1.0 / 3.0);

    std::shared_ptr<geometry::PointCloud> ds;
    for (int iter = 0; iter < 8; ++iter) {
        ds = pcd.VoxelDownSample(voxel);
        int c = static_cast<int>(ds->points_.size());
        if (c >= target * 0.8 && c <= target * 1.2) break;
        if (c == 0) { voxel *= 0.5; continue; }
        voxel *= std::pow(static_cast<double>(c) / target, 1.0 / 3.0);
    }

    DownsampleResult res;
    res.voxel_size = voxel;
    from_o3d(*ds, res.xyz, res.rgb);
    return res;
}

// ---------- Registration ----------

static Eigen::Matrix4d fgr_global(geometry::PointCloud& src,
                                  geometry::PointCloud& tgt,
                                  double voxel) {
    auto s = src.VoxelDownSample(voxel);
    auto t = tgt.VoxelDownSample(voxel);

    s->EstimateNormals(geometry::KDTreeSearchParamHybrid(voxel * 2.0, 30));
    t->EstimateNormals(geometry::KDTreeSearchParamHybrid(voxel * 2.0, 30));

    auto fs = pipelines::registration::ComputeFPFHFeature(
        *s, geometry::KDTreeSearchParamHybrid(voxel * 5.0, 100));
    auto ft = pipelines::registration::ComputeFPFHFeature(
        *t, geometry::KDTreeSearchParamHybrid(voxel * 5.0, 100));

    auto result = pipelines::registration::
        FastGlobalRegistrationBasedOnFeatureMatching(
            *s, *t, *fs, *ft,
            pipelines::registration::FastGlobalRegistrationOption(voxel * 1.5));

    return result.transformation_;
}

static std::pair<Eigen::Matrix4d, std::vector<ICPScaleResult>>
multiscale_icp(const PointCloudF& src_full, const PointCloudF& tgt_full,
               const Eigen::Matrix4d& init_T,
               const std::vector<double>& voxels,
               ProgressCallback progress) {
    Eigen::Matrix4d T = init_T;
    std::vector<ICPScaleResult> history;

    for (size_t si = 0; si < voxels.size(); ++si) {
        double v = voxels[si];

        // Build O3D clouds from full float32 via voxel downsampling
        auto s_o3d = to_o3d(src_full.xyz.data(), src_full.size());
        auto t_o3d = to_o3d(tgt_full.xyz.data(), tgt_full.size());
        auto s = s_o3d.VoxelDownSample(v);
        auto t = t_o3d.VoxelDownSample(v);

        s->EstimateNormals(geometry::KDTreeSearchParamHybrid(v * 2.0, 30));
        t->EstimateNormals(geometry::KDTreeSearchParamHybrid(v * 2.0, 30));

        if (progress) {
            progress("icp",
                "scale " + std::to_string(si+1) + "/" + std::to_string(voxels.size()) +
                ", voxel=" + std::to_string(v) + "m, pts=" + fmt_count(s->points_.size()),
                static_cast<float>(si) / voxels.size());
        }

        auto criteria = pipelines::registration::ICPConvergenceCriteria(50, 1e-7, 1e-7);
        auto result = pipelines::registration::RegistrationICP(
            *s, *t, v * 2.0, T,
            pipelines::registration::TransformationEstimationPointToPlane(),
            criteria);

        T = result.transformation_;

        ICPScaleResult sr;
        sr.voxel = v;
        sr.fitness = result.fitness_;
        sr.inlier_rmse = result.inlier_rmse_;
        sr.src_count = static_cast<int>(s->points_.size());
        sr.tgt_count = static_cast<int>(t->points_.size());
        history.push_back(sr);

        if (progress) {
            char buf[128];
            snprintf(buf, sizeof(buf), "scale %zu/%zu: fitness=%.4f, RMSE=%.6fm",
                     si+1, voxels.size(), sr.fitness, sr.inlier_rmse);
            progress("icp_result", buf, static_cast<float>(si+1) / voxels.size());
        }
    }
    return {T, history};
}

// Apply 4x4 transform to float32 XYZ array in-place (OpenMP)
static void apply_transform(float* xyz, size_t n, const Eigen::Matrix4d& T) {
    // Extract as float for speed
    float r00 = (float)T(0,0), r01 = (float)T(0,1), r02 = (float)T(0,2), t0 = (float)T(0,3);
    float r10 = (float)T(1,0), r11 = (float)T(1,1), r12 = (float)T(1,2), t1 = (float)T(1,3);
    float r20 = (float)T(2,0), r21 = (float)T(2,1), r22 = (float)T(2,2), t2 = (float)T(2,3);

    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(n); ++i) {
        float x = xyz[i*3+0], y = xyz[i*3+1], z = xyz[i*3+2];
        xyz[i*3+0] = r00*x + r01*y + r02*z + t0;
        xyz[i*3+1] = r10*x + r11*y + r12*z + t1;
        xyz[i*3+2] = r20*x + r21*y + r22*z + t2;
    }
}

// Displacement stats
static json stats(const float* arr, size_t n, int stride, int offset) {
    if (n == 0) return {};
    double sum = 0, abs_sum = 0, sq_sum = 0;
    float mn = arr[offset], mx = arr[offset];
    std::vector<float> abs_vals(n);

    for (size_t i = 0; i < n; ++i) {
        float v = arr[i * stride + offset];
        sum += v;
        abs_sum += std::abs(v);
        sq_sum += static_cast<double>(v) * v;
        mn = std::min(mn, v);
        mx = std::max(mx, v);
        abs_vals[i] = std::abs(v);
    }

    std::nth_element(abs_vals.begin(), abs_vals.begin() + (size_t)(n * 0.95), abs_vals.end());

    return {
        {"min", mn}, {"max", mx},
        {"mean", sum / n}, {"abs_mean", abs_sum / n},
        {"rms", std::sqrt(sq_sum / n)},
        {"p95_abs", abs_vals[(size_t)(n * 0.95)]}
    };
}

// List raw scan .ply files (excluding *_simple.ply) sorted by name
static std::vector<fs::path> list_scans(const fs::path& folder) {
    std::vector<fs::path> v;
    for (auto& e : fs::directory_iterator(folder)) {
        if (e.path().extension() == ".ply") {
            auto stem = e.path().stem().string();
            if (stem.size() < 7 || stem.substr(stem.size() - 7) != "_simple") {
                v.push_back(e.path());
            }
        }
    }
    std::sort(v.begin(), v.end());
    return v;
}

// ---------- Main pipeline ----------

int run(const Config& cfg, ProgressCallback progress) {
    auto P = [&](const std::string& s, const std::string& d, float p = -1) {
        if (progress) progress(s, d, p);
    };

    fs::path folder = cfg.data_root / cfg.dataset;
    fs::path target_path = folder / cfg.target_file;
    if (!fs::exists(target_path)) {
        P("error", "File not found: " + target_path.string());
        return 1;
    }

    auto scan_files = list_scans(folder);
    if (scan_files.empty()) {
        P("error", "No scan .ply files in " + folder.string());
        return 1;
    }

    fs::path ref_path = scan_files[0];
    bool is_reference = (target_path.filename() == ref_path.filename());
    std::string ref_stem = ref_path.stem().string();
    std::string tgt_stem = target_path.stem().string();

    json timing;
    auto t_start = std::chrono::high_resolution_clock::now();
    auto elapsed = [&]() {
        return std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - t_start).count();
    };

    // --- Reference simple ---
    fs::path ref_simple_path = folder / (ref_stem + "_simple.ply");
    detail::DownsampleResult ref_ds;
    PointCloudF ref_full;

    if (!fs::exists(ref_simple_path)) {
        P("load_ref", ref_path.filename().string());
        auto t0 = elapsed();
        ref_full = load_scan(ref_path, true, progress);
        timing["load_ref"] = elapsed() - t0;

        P("downsample_ref", fmt_count(ref_full.size()) + " pts");
        t0 = elapsed();
        ref_ds = detail::voxel_downsample_to_target(ref_full, cfg.target_points);
        timing["downsample_ref"] = elapsed() - t0;

        write_binary_ply(ref_simple_path, ref_ds.xyz.data(), ref_ds.xyz.size() / 3,
                         ref_ds.rgb.empty() ? nullptr : ref_ds.rgb.data());

        if (!ref_ds.rgb.empty()) {
            auto rgb_path = folder / (ref_stem + "_rgb.bin");
            std::ofstream(rgb_path, std::ios::binary)
                .write(reinterpret_cast<const char*>(ref_ds.rgb.data()), ref_ds.rgb.size());
        }

        json ref_meta;
        ref_meta["is_reference"] = true;
        ref_meta["source"] = ref_path.filename().string();
        ref_meta["simple_voxel"] = ref_ds.voxel_size;
        ref_meta["simple_count"] = ref_ds.xyz.size() / 3;
        ref_meta["transform"] = std::vector<std::vector<double>>{
            {1,0,0,0},{0,1,0,0},{0,0,1,0},{0,0,0,1}};
        ref_meta["icp_history"] = json::array();
        ref_meta["global_voxel"] = nullptr;
        ref_meta["timing"] = {{"downsample", timing.value("downsample_ref", 0.0)}};
        std::ofstream(folder / (ref_stem + "_meta.json")) << ref_meta.dump(2);
    }

    if (is_reference) {
        P("done", "reference preprocessed");
        return 0;
    }

    // --- Load target ---
    P("load_target", target_path.filename().string());
    auto t0 = elapsed();
    PointCloudF tgt_full = load_scan(target_path, true, progress);
    timing["load_target"] = elapsed() - t0;

    P("downsample_target", fmt_count(tgt_full.size()) + " pts");
    t0 = elapsed();
    auto tgt_ds = detail::voxel_downsample_to_target(tgt_full, cfg.target_points);
    timing["downsample_target"] = elapsed() - t0;

    // Save untransformed simple + RGB
    fs::path tgt_simple_path = folder / (tgt_stem + "_simple.ply");
    write_binary_ply(tgt_simple_path, tgt_ds.xyz.data(), tgt_ds.xyz.size() / 3,
                     tgt_ds.rgb.empty() ? nullptr : tgt_ds.rgb.data());
    if (!tgt_ds.rgb.empty()) {
        auto rgb_path = folder / (tgt_stem + "_rgb.bin");
        std::ofstream(rgb_path, std::ios::binary)
            .write(reinterpret_cast<const char*>(tgt_ds.rgb.data()), tgt_ds.rgb.size());
    }

    // --- Load reference if needed ---
    if (ref_full.size() == 0) {
        P("load_ref_full", ref_path.filename().string());
        t0 = elapsed();
        ref_full = load_scan(ref_path, false, progress);
        timing["load_ref_full"] = elapsed() - t0;
    }

    // Load ref simple for FGR
    if (ref_ds.xyz.empty()) {
        auto ref_o3d = io::CreatePointCloudFromFile(ref_simple_path.string());
        from_o3d(*ref_o3d, ref_ds.xyz, ref_ds.rgb);
    }

    // --- FGR on simple clouds ---
    P("fgr", "voxel=" + std::to_string(cfg.fgr_voxel) + " on simple clouds");
    t0 = elapsed();
    auto src_o3d = to_o3d(tgt_ds.xyz.data(), tgt_ds.xyz.size() / 3);
    auto tgt_o3d = to_o3d(ref_ds.xyz.data(), ref_ds.xyz.size() / 3);
    Eigen::Matrix4d T0 = fgr_global(src_o3d, tgt_o3d, cfg.fgr_voxel);
    timing["fgr"] = elapsed() - t0;

    // --- Multi-scale ICP on full clouds ---
    P("icp", "voxels=" + std::to_string(cfg.icp_voxels.size()) + " scales");
    t0 = elapsed();
    auto [T_final, history] = multiscale_icp(tgt_full, ref_full, T0, cfg.icp_voxels, progress);
    timing["icp"] = elapsed() - t0;

    // --- Apply transform to simple cloud and save ---
    apply_transform(tgt_ds.xyz.data(), tgt_ds.xyz.size() / 3, T_final);
    write_binary_ply(tgt_simple_path, tgt_ds.xyz.data(), tgt_ds.xyz.size() / 3,
                     tgt_ds.rgb.empty() ? nullptr : tgt_ds.rgb.data());

    // --- Apply transform to full target ---
    P("transform_full", fmt_count(tgt_full.size()) + " pts");
    t0 = elapsed();
    apply_transform(tgt_full.xyz.data(), tgt_full.size(), T_final);
    timing["transform_full"] = elapsed() - t0;

    // --- Estimate normals on downsampled reference, then map to full via NN ---
    P("normals", "estimating normals on simple ref (" + fmt_count(ref_ds.xyz.size()/3) + " pts)");
    t0 = elapsed();
    {
        auto ref_ds_o3d = to_o3d(ref_ds.xyz.data(), ref_ds.xyz.size() / 3);
        ref_ds_o3d.EstimateNormals(geometry::KDTreeSearchParamHybrid(0.5, 30));
        ref_ds_o3d.OrientNormalsConsistentTangentPlane(20);

        size_t n_ds = ref_ds_o3d.points_.size();
        std::vector<float> ds_xyz(n_ds * 3);
        std::vector<float> ds_normals(n_ds * 3);
        for (size_t i = 0; i < n_ds; ++i) {
            ds_xyz[i*3+0] = (float)ref_ds_o3d.points_[i][0];
            ds_xyz[i*3+1] = (float)ref_ds_o3d.points_[i][1];
            ds_xyz[i*3+2] = (float)ref_ds_o3d.points_[i][2];
            ds_normals[i*3+0] = (float)ref_ds_o3d.normals_[i][0];
            ds_normals[i*3+1] = (float)ref_ds_o3d.normals_[i][1];
            ds_normals[i*3+2] = (float)ref_ds_o3d.normals_[i][2];
        }

        P("normals", "mapping normals to full ref (" + fmt_count(ref_full.size()) + " pts)");
        ref_full.normals.resize(ref_full.size() * 3);
        map_normals_nn(ds_xyz.data(), ds_normals.data(), n_ds,
                       ref_full.xyz.data(), ref_full.normals.data(), ref_full.size());
    }
    timing["normals"] = elapsed() - t0;

    // --- Displacement (full vs full) ---
    P("displacement", fmt_count(tgt_full.size()) + " vs " + fmt_count(ref_full.size()) + " pts");
    t0 = elapsed();
    auto disp_full = compute_displacement(
        tgt_full.xyz.data(), tgt_full.size(),
        ref_full.xyz.data(), ref_full.normals.data(), ref_full.size(),
        progress);
    timing["displacement_full"] = elapsed() - t0;

    // --- Map to simple ---
    P("disp_map", "mapping to simple " + fmt_count(tgt_ds.xyz.size() / 3) + " pts");
    t0 = elapsed();
    auto disp_simple = map_disp_to_simple(
        tgt_full.xyz.data(), tgt_full.size(),
        tgt_ds.xyz.data(), tgt_ds.xyz.size() / 3,
        disp_full.data(), progress);
    timing["displacement_map"] = elapsed() - t0;

    // --- Write disp.bin ---
    fs::path disp_path = folder / (tgt_stem + "_disp.bin");
    {
        std::ofstream f(disp_path, std::ios::binary);
        f.write(reinterpret_cast<const char*>(disp_simple.data()),
                disp_simple.size() * sizeof(float));
    }

    // --- Write meta.json ---
    size_t n_simple = tgt_ds.xyz.size() / 3;
    json icp_hist = json::array();
    for (auto& h : history) {
        icp_hist.push_back({
            {"voxel", h.voxel}, {"fitness", h.fitness},
            {"inlier_rmse", h.inlier_rmse},
            {"src_count", h.src_count}, {"tgt_count", h.tgt_count}
        });
    }

    auto to_list = [](const Eigen::Matrix4d& M) {
        json rows = json::array();
        for (int r = 0; r < 4; ++r) {
            json row = json::array();
            for (int c = 0; c < 4; ++c) row.push_back(M(r, c));
            rows.push_back(row);
        }
        return rows;
    };

    double total_sec = elapsed();
    json meta = {
        {"is_reference", false},
        {"reference", ref_path.filename().string()},
        {"source", target_path.filename().string()},
        {"simple_voxel", tgt_ds.voxel_size},
        {"simple_count", (int)n_simple},
        {"global_voxel", cfg.fgr_voxel},
        {"icp_voxels", cfg.icp_voxels},
        {"icp_history", icp_hist},
        {"transform_global", to_list(T0)},
        {"transform", to_list(T_final)},
        {"displacement_stats", {
            {"signed_normal", stats(disp_simple.data(), n_simple, 4, 0)},
            {"magnitude",     stats(disp_simple.data(), n_simple, 4, 1)},
            {"horizontal",    stats(disp_simple.data(), n_simple, 4, 2)},
            {"vertical",      stats(disp_simple.data(), n_simple, 4, 3)},
        }},
        {"timing", timing},
        {"total_seconds", total_sec},
    };
    std::ofstream(folder / (tgt_stem + "_meta.json")) << meta.dump(2);

    P("done", std::to_string(total_sec) + "s");
    return 0;
}

} // namespace pipeline
