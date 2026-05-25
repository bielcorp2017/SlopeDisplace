"""수직봉(vertical rod) PLY의 방향 벡터를 PCA로 추출하고 시각화한다.

사용:
    python analyze_rod_axis.py <ply_path> [--scale 2.0] [--no-vis]

PCA의 최대 고유벡터가 막대(rod)의 주축 방향이 된다.
출력에는 중심, 단위벡터, 점군 길이, Z축 대비 기울기(deg)가 포함된다.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import open3d as o3d


def compute_rod_axis(points: np.ndarray):
    """점군의 주축(가장 큰 분산 방향)을 PCA로 계산.

    Returns
    -------
    centroid : (3,) float64
    axis     : (3,) float64, 단위벡터, +Z 쪽으로 향하도록 부호 정렬
    length   : float, 점군이 축에 투영된 길이
    """
    centroid = points.mean(axis=0)
    centered = points - centroid
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, np.argmax(eigvals)]
    if axis[2] < 0:
        axis = -axis
    axis = axis / np.linalg.norm(axis)
    proj = centered @ axis
    length = float(proj.max() - proj.min())
    return centroid, axis, length


def make_axis_line(centroid, axis, length, scale=2.0, color=(1.0, 0.1, 0.1)):
    """중심을 지나며 점군 길이의 `scale`배인 LineSet 생성."""
    half = (length * scale) / 2.0
    p0 = centroid - axis * half
    p1 = centroid + axis * half
    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.vstack([p0, p1])),
        lines=o3d.utility.Vector2iVector([[0, 1]]),
    )
    ls.colors = o3d.utility.Vector3dVector([color])
    return ls


def main():
    parser = argparse.ArgumentParser(description="수직봉 방향 벡터 추출 및 시각화")
    parser.add_argument("ply_path", help="입력 PLY 파일 경로")
    parser.add_argument("--scale", type=float, default=2.0,
                        help="축선을 점군 길이 대비 몇 배로 그릴지 (기본 2.0)")
    parser.add_argument("--no-vis", action="store_true",
                        help="시각화 없이 벡터 정보만 출력")
    args = parser.parse_args()

    pcd = o3d.io.read_point_cloud(args.ply_path)
    pts = np.asarray(pcd.points)
    if pts.size == 0:
        print(f"[ERROR] 점군이 비어 있습니다: {args.ply_path}", file=sys.stderr)
        sys.exit(1)

    centroid, axis, length = compute_rod_axis(pts)

    z = np.array([0.0, 0.0, 1.0])
    cos_a = float(np.clip(np.dot(axis, z), -1.0, 1.0))
    tilt_deg = float(np.degrees(np.arccos(cos_a)))

    print(f"파일        : {args.ply_path}")
    print(f"점 개수     : {len(pts):,}")
    print(f"중심        : ({centroid[0]:.4f}, {centroid[1]:.4f}, {centroid[2]:.4f})")
    print(f"방향벡터    : ({axis[0]:.6f}, {axis[1]:.6f}, {axis[2]:.6f})")
    print(f"점군 길이   : {length:.4f}")
    print(f"Z축 기울기  : {tilt_deg:.3f} deg")

    if args.no_vis:
        return

    line = make_axis_line(centroid, axis, length, scale=args.scale)

    if not pcd.has_colors():
        pcd.paint_uniform_color([0.70, 0.72, 0.78])

    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=max(length * 0.01, 1e-3))
    sphere.translate(centroid)
    sphere.paint_uniform_color([1.0, 0.9, 0.1])

    world_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=max(length * 0.3, 0.1), origin=centroid - axis * (length / 2.0)
    )

    o3d.visualization.draw_geometries(
        [pcd, line, sphere, world_axes],
        window_name="Rod Axis (red line = PCA principal direction)",
    )


if __name__ == "__main__":
    main()
