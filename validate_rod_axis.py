"""수직봉 축 추정 결과를 다중 방법으로 교차검증한다.

3가지 독립적인 방법으로 축을 추정하고 일치도를 비교한다:
    A. PCA   최대 분산 방향 (analyze_rod_axis.py 와 동일)
    B. 슬라이스 중심 직선   각 단면에 2D 원을 피팅 → 중심들에 직선 피팅
    C. 원기둥 LSQ           A를 초기값으로 (center, axis, radius) 비선형 최소제곱 피팅

세 축이 모두 같은 방향이면 PCA 결과가 진짜 봉 축임이 증명된다.

추가 진단:
    - 선형성 L = λ1/(λ1+λ2+λ3)            1에 가까우면 막대 모양
    - planarity P = (λ2-λ3)/λ1            ~0 이어야 함 (납작하면 안 됨)
    - radial std / mean radius             원기둥 표면 잡음 비율
    - per-slice 반경 평균/표준편차          축 방향으로 반경이 일정한지

사용:
    python validate_rod_axis.py <ply_path> [--slices 30] [--no-vis]
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import open3d as o3d
from scipy.optimize import least_squares


# ---------- A. PCA ----------
def pca_axis(points: np.ndarray):
    c = points.mean(axis=0)
    centered = points - c
    cov = np.cov(centered.T)
    w, v = np.linalg.eigh(cov)
    order = np.argsort(w)[::-1]
    w = w[order]
    v = v[:, order]
    axis = v[:, 0]
    if axis[2] < 0:
        axis = -axis
    return c, axis / np.linalg.norm(axis), w


# ---------- B. 슬라이스별 원 피팅 → 중심 직선 ----------
def _orthonormal_basis(axis: np.ndarray):
    a = axis / np.linalg.norm(axis)
    helper = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(a, helper)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(a, e1)
    return e1, e2


def _fit_circle_2d(xy: np.ndarray):
    """Kasa algebraic circle fit. Returns (cx, cy, r)."""
    x, y = xy[:, 0], xy[:, 1]
    A = np.column_stack([2 * x, 2 * y, np.ones_like(x)])
    b = x * x + y * y
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = sol[0], sol[1]
    r = float(np.sqrt(max(sol[2] + cx * cx + cy * cy, 0.0)))
    return float(cx), float(cy), r


def slice_circle_axis(points, centroid, axis, n_slices=30):
    e1, e2 = _orthonormal_basis(axis)
    centered = points - centroid
    s = centered @ axis
    s_min, s_max = s.min(), s.max()
    edges = np.linspace(s_min, s_max, n_slices + 1)
    centers_world, radii, s_mids = [], [], []
    for i in range(n_slices):
        mask = (s >= edges[i]) & (s < edges[i + 1])
        if mask.sum() < 30:
            continue
        slab = centered[mask]
        xy = np.column_stack([slab @ e1, slab @ e2])
        cx, cy, r = _fit_circle_2d(xy)
        s_mid = 0.5 * (edges[i] + edges[i + 1])
        centers_world.append(centroid + s_mid * axis + cx * e1 + cy * e2)
        radii.append(r)
        s_mids.append(s_mid)
    centers_world = np.asarray(centers_world)
    radii = np.asarray(radii)
    # 중심 점들에 직선 피팅
    c0 = centers_world.mean(axis=0)
    _, _, vt = np.linalg.svd(centers_world - c0, full_matrices=False)
    d = vt[0]
    if d[2] < 0:
        d = -d
    return c0, d / np.linalg.norm(d), centers_world, radii


# ---------- C. 원기둥 LSQ ----------
def _cyl_residuals(params, points):
    c = params[:3]
    d = params[3:6]
    d = d / (np.linalg.norm(d) + 1e-12)
    r = params[6]
    v = points - c
    return np.linalg.norm(np.cross(v, d), axis=1) - r


def cylinder_lsq(points, init_c, init_axis, init_r, sample=200_000):
    if len(points) > sample:
        idx = np.random.default_rng(0).choice(len(points), sample, replace=False)
        pts = points[idx]
    else:
        pts = points
    x0 = np.concatenate([init_c, init_axis, [init_r]])
    res = least_squares(_cyl_residuals, x0, args=(pts,), method="lm", max_nfev=200)
    c = res.x[:3]
    d = res.x[3:6]
    d = d / np.linalg.norm(d)
    if d[2] < 0:
        d = -d
    r = abs(float(res.x[6]))
    final_resid = _cyl_residuals(res.x, pts)
    return c, d, r, final_resid


# ---------- 보조 ----------
def angle_deg(a, b):
    return float(np.degrees(np.arccos(np.clip(abs(np.dot(a, b)), -1.0, 1.0))))


def point_line_dist(points, c, d):
    return np.linalg.norm(np.cross(points - c, d), axis=1)


def make_line(c, d, length, color):
    p0 = c - d * length / 2
    p1 = c + d * length / 2
    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.vstack([p0, p1])),
        lines=o3d.utility.Vector2iVector([[0, 1]]),
    )
    ls.colors = o3d.utility.Vector3dVector([color])
    return ls


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="수직봉 축 다중 방법 교차검증")
    ap.add_argument("ply_path")
    ap.add_argument("--slices", type=int, default=30)
    ap.add_argument("--no-vis", action="store_true")
    args = ap.parse_args()

    pcd = o3d.io.read_point_cloud(args.ply_path)
    pts = np.asarray(pcd.points)
    if pts.size == 0:
        print(f"[ERROR] empty point cloud: {args.ply_path}", file=sys.stderr)
        sys.exit(1)

    # A. PCA
    c_pca, axis_pca, eigvals = pca_axis(pts)
    L = float(eigvals[0] / eigvals.sum())
    P = float((eigvals[1] - eigvals[2]) / eigvals[0])
    proj = (pts - c_pca) @ axis_pca
    rod_length = float(proj.max() - proj.min())
    dist_pca = point_line_dist(pts, c_pca, axis_pca)
    r_pca_mean = float(dist_pca.mean())
    r_pca_std = float(dist_pca.std())

    # B. 슬라이스 원 피팅
    c_slc, axis_slc, slc_centers, slc_radii = slice_circle_axis(
        pts, c_pca, axis_pca, n_slices=args.slices
    )

    # C. 원기둥 LSQ
    c_cyl, axis_cyl, r_cyl, cyl_resid = cylinder_lsq(
        pts, c_pca, axis_pca, r_pca_mean
    )

    # 각도 차이
    ang_pca_slc = angle_deg(axis_pca, axis_slc)
    ang_pca_cyl = angle_deg(axis_pca, axis_cyl)
    ang_slc_cyl = angle_deg(axis_slc, axis_cyl)

    print(f"파일                : {args.ply_path}")
    print(f"점 개수             : {len(pts):,}")
    print()
    print("[A] PCA")
    print(f"  중심              : {c_pca}")
    print(f"  축 벡터           : {axis_pca}")
    print(f"  점군 길이         : {rod_length:.4f}")
    print(f"  선형성 L          : {L:.4f}   (1에 가까울수록 막대)")
    print(f"  planarity P       : {P:.4f}   (~0 이어야 함)")
    print(f"  축까지 거리 mean  : {r_pca_mean:.5f}")
    print(f"  축까지 거리 std   : {r_pca_std:.5f}   (std/mean = {r_pca_std/max(r_pca_mean,1e-9):.4f})")
    print()
    print("[B] 슬라이스 원 피팅 → 중심 직선")
    print(f"  유효 슬라이스 수  : {len(slc_centers)}")
    print(f"  축 벡터           : {axis_slc}")
    print(f"  슬라이스 반경 mean: {slc_radii.mean():.5f}")
    print(f"  슬라이스 반경 std : {slc_radii.std():.5f}")
    print()
    print("[C] 원기둥 LSQ")
    print(f"  중심              : {c_cyl}")
    print(f"  축 벡터           : {axis_cyl}")
    print(f"  반경              : {r_cyl:.5f}")
    print(f"  잔차 std          : {cyl_resid.std():.5f}")
    print(f"  잔차 max          : {np.abs(cyl_resid).max():.5f}")
    print()
    print("[축 일치도] (각도, deg — 작을수록 같은 축)")
    print(f"  PCA vs Slice      : {ang_pca_slc:.4f}")
    print(f"  PCA vs Cylinder   : {ang_pca_cyl:.4f}")
    print(f"  Slice vs Cylinder : {ang_slc_cyl:.4f}")
    print()

    verdict_axis = max(ang_pca_slc, ang_pca_cyl) < 1.0
    verdict_shape = L > 0.95 and P < 0.05
    verdict_cyl = (cyl_resid.std() / max(r_cyl, 1e-9)) < 0.10
    overall = verdict_axis and verdict_shape and verdict_cyl
    print(f"[판정]")
    print(f"  축 일치 (<1°)            : {'PASS' if verdict_axis else 'FAIL'}")
    print(f"  막대 형태 (L>0.95,P<0.05): {'PASS' if verdict_shape else 'FAIL'}")
    print(f"  원기둥성 (resid<10% r)   : {'PASS' if verdict_cyl else 'FAIL'}")
    print(f"  >>> 종합                  : {'PROVEN'  if overall else 'NOT PROVEN'}")

    if args.no_vis:
        return

    # 시각화: 세 축을 모두 길게 그어 겹치는지 본다
    if not pcd.has_colors():
        pcd.paint_uniform_color([0.70, 0.72, 0.78])
    L_draw = rod_length * 2.0
    line_pca = make_line(c_pca, axis_pca, L_draw, [1.0, 0.1, 0.1])  # red
    line_slc = make_line(c_slc, axis_slc, L_draw, [0.1, 1.0, 0.1])  # green
    line_cyl = make_line(c_cyl, axis_cyl, L_draw, [0.1, 0.3, 1.0])  # blue

    # 슬라이스 중심점 표시
    ctr_pcd = o3d.geometry.PointCloud()
    ctr_pcd.points = o3d.utility.Vector3dVector(slc_centers)
    ctr_pcd.paint_uniform_color([1.0, 0.9, 0.1])

    o3d.visualization.draw_geometries(
        [pcd, line_pca, line_slc, line_cyl, ctr_pcd],
        window_name="Axis validation — red=PCA  green=slice  blue=cylinderLSQ",
    )


if __name__ == "__main__":
    main()
