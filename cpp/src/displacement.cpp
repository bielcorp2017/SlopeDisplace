#include "displacement.h"
#include <nanoflann.hpp>
#include <omp.h>
#include <cmath>
#include <algorithm>
#include <Eigen/Dense>

namespace {

// Zero-copy nanoflann adaptor over interleaved float32 array
struct CloudAdaptor {
    const float* pts;
    size_t n;
    size_t kdtree_get_point_count() const { return n; }
    float kdtree_get_pt(size_t idx, size_t dim) const { return pts[idx * 3 + dim]; }
    template <class BBOX> bool kdtree_get_bbox(BBOX&) const { return false; }
};

using KDTree = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<float, CloudAdaptor>,
    CloudAdaptor, 3, size_t>;

} // anonymous namespace

std::vector<float> compute_displacement(
    const float* target_xyz, size_t n_target,
    const float* ref_xyz, const float* ref_normals, size_t n_ref,
    ProgressCallback progress)
{
    if (progress) progress("displacement", "building KD-tree", 0.0f);

    CloudAdaptor adaptor{ref_xyz, n_ref};
    KDTree tree(3, adaptor, nanoflann::KDTreeSingleIndexAdaptorParams(32));
    tree.buildIndex();

    if (progress) progress("displacement", "computing NN displacement", 0.2f);

    std::vector<float> result(n_target * 4);

    #pragma omp parallel for schedule(dynamic, 4096)
    for (long long i = 0; i < static_cast<long long>(n_target); ++i) {
        const float* tp = target_xyz + i * 3;
        float query[3] = {tp[0], tp[1], tp[2]};

        size_t nn_idx;
        float nn_dist_sq;
        nanoflann::KNNResultSet<float> resultSet(1);
        resultSet.init(&nn_idx, &nn_dist_sq);
        tree.findNeighbors(resultSet, query);

        const float* rp = ref_xyz + nn_idx * 3;
        const float* rn = ref_normals + nn_idx * 3;

        float dx = tp[0] - rp[0];
        float dy = tp[1] - rp[1];
        float dz = tp[2] - rp[2];

        // Signed normal: delta dot normal
        float signed_normal = dx * rn[0] + dy * rn[1] + dz * rn[2];

        // Magnitude: Euclidean NN distance
        float magnitude = std::sqrt(nn_dist_sq);

        // Horizontal: project XY displacement onto outward XY direction of normal
        float nxy_len = std::sqrt(rn[0] * rn[0] + rn[1] * rn[1]);
        float horiz;
        if (nxy_len > 1e-6f) {
            float nxy_dx = rn[0] / nxy_len;
            float nxy_dy = rn[1] / nxy_len;
            horiz = dx * nxy_dx + dy * nxy_dy;
        } else {
            horiz = std::sqrt(dx * dx + dy * dy);
        }

        // Vertical: Z component of normal-projected displacement
        float vert = signed_normal * rn[2];

        result[i * 4 + 0] = signed_normal;
        result[i * 4 + 1] = magnitude;
        result[i * 4 + 2] = horiz;
        result[i * 4 + 3] = vert;
    }

    if (progress) progress("displacement", "done", 1.0f);
    return result;
}


void estimate_normals(PointCloudF& cloud, float radius, int max_nn) {
    size_t n = cloud.size();
    cloud.normals.resize(n * 3);

    CloudAdaptor adaptor{cloud.xyz.data(), n};
    KDTree tree(3, adaptor, nanoflann::KDTreeSingleIndexAdaptorParams(32));
    tree.buildIndex();

    #pragma omp parallel for schedule(dynamic, 1024)
    for (long long i = 0; i < static_cast<long long>(n); ++i) {
        float query[3] = {cloud.xyz[i*3], cloud.xyz[i*3+1], cloud.xyz[i*3+2]};

        std::vector<nanoflann::ResultItem<size_t, float>> matches;
        nanoflann::SearchParameters params;
        tree.radiusSearch(query, radius * radius, matches, params);

        if (matches.size() < 3) {
            cloud.normals[i*3+0] = 0; cloud.normals[i*3+1] = 0; cloud.normals[i*3+2] = 1;
            continue;
        }

        size_t nn = std::min(static_cast<size_t>(max_nn), matches.size());

        // Compute centroid
        Eigen::Vector3f centroid(0, 0, 0);
        for (size_t j = 0; j < nn; ++j) {
            size_t idx = matches[j].first;
            centroid += Eigen::Vector3f(cloud.xyz[idx*3], cloud.xyz[idx*3+1], cloud.xyz[idx*3+2]);
        }
        centroid /= static_cast<float>(nn);

        // Covariance matrix
        Eigen::Matrix3f cov = Eigen::Matrix3f::Zero();
        for (size_t j = 0; j < nn; ++j) {
            size_t idx = matches[j].first;
            Eigen::Vector3f p(cloud.xyz[idx*3], cloud.xyz[idx*3+1], cloud.xyz[idx*3+2]);
            Eigen::Vector3f d = p - centroid;
            cov += d * d.transpose();
        }

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(cov);
        Eigen::Vector3f normal = solver.eigenvectors().col(0); // smallest eigenvalue

        cloud.normals[i*3+0] = normal[0];
        cloud.normals[i*3+1] = normal[1];
        cloud.normals[i*3+2] = normal[2];
    }
}


void map_normals_nn(
    const float* ds_xyz, const float* ds_normals, size_t n_ds,
    const float* full_xyz, float* full_normals, size_t n_full)
{
    CloudAdaptor adaptor{ds_xyz, n_ds};
    KDTree tree(3, adaptor, nanoflann::KDTreeSingleIndexAdaptorParams(32));
    tree.buildIndex();

    #pragma omp parallel for schedule(dynamic, 4096)
    for (long long i = 0; i < static_cast<long long>(n_full); ++i) {
        float query[3] = {full_xyz[i*3], full_xyz[i*3+1], full_xyz[i*3+2]};
        size_t nn_idx;
        float nn_dist_sq;
        nanoflann::KNNResultSet<float> rs(1);
        rs.init(&nn_idx, &nn_dist_sq);
        tree.findNeighbors(rs, query);

        full_normals[i*3+0] = ds_normals[nn_idx*3+0];
        full_normals[i*3+1] = ds_normals[nn_idx*3+1];
        full_normals[i*3+2] = ds_normals[nn_idx*3+2];
    }
}


std::vector<float> map_disp_to_simple(
    const float* full_xyz, size_t n_full,
    const float* simple_xyz, size_t n_simple,
    const float* disp_full,
    ProgressCallback progress)
{
    if (progress) progress("disp_map", "building KD-tree on full target", 0.0f);

    CloudAdaptor adaptor{full_xyz, n_full};
    KDTree tree(3, adaptor, nanoflann::KDTreeSingleIndexAdaptorParams(32));
    tree.buildIndex();

    if (progress) progress("disp_map", "mapping to simple cloud", 0.3f);

    std::vector<float> result(n_simple * 4);

    #pragma omp parallel for schedule(dynamic, 1024)
    for (long long i = 0; i < static_cast<long long>(n_simple); ++i) {
        float query[3] = {simple_xyz[i*3], simple_xyz[i*3+1], simple_xyz[i*3+2]};

        size_t nn_idx;
        float nn_dist_sq;
        nanoflann::KNNResultSet<float> rs(1);
        rs.init(&nn_idx, &nn_dist_sq);
        tree.findNeighbors(rs, query);

        result[i*4+0] = disp_full[nn_idx*4+0];
        result[i*4+1] = disp_full[nn_idx*4+1];
        result[i*4+2] = disp_full[nn_idx*4+2];
        result[i*4+3] = disp_full[nn_idx*4+3];
    }

    if (progress) progress("disp_map", "done", 1.0f);
    return result;
}
