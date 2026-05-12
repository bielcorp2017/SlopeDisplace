#pragma once
#include "types.h"
#include <vector>

// Compute per-point displacement (N x 4 float32):
//   [signed_normal, magnitude, horizontal_xy, vertical_z]
// target_xyz: transformed target points (float32, N*3)
// ref_xyz: reference points (float32, M*3)
// ref_normals: reference normals (float32, M*3)
std::vector<float> compute_displacement(
    const float* target_xyz, size_t n_target,
    const float* ref_xyz, const float* ref_normals, size_t n_ref,
    ProgressCallback progress = nullptr);

// Estimate normals for a point cloud using nanoflann NN queries.
// radius: search radius, max_nn: max neighbors for PCA.
void estimate_normals(PointCloudF& cloud, float radius = 0.5f, int max_nn = 30);

// Map 3-channel normals from downsampled cloud to full cloud via NN.
void map_normals_nn(
    const float* ds_xyz, const float* ds_normals, size_t n_ds,
    const float* full_xyz, float* full_normals, size_t n_full);

// Map displacement from full cloud to simple cloud via NN.
// full_xyz: full transformed target (N_full * 3)
// simple_xyz: simple cloud points (N_simple * 3)
// disp_full: full displacement (N_full * 4)
// Returns: N_simple * 4 displacement mapped via nearest neighbor.
std::vector<float> map_disp_to_simple(
    const float* full_xyz, size_t n_full,
    const float* simple_xyz, size_t n_simple,
    const float* disp_full,
    ProgressCallback progress = nullptr);
