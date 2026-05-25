"""tripod 스캔에서 수직봉만 분리해 정밀 축벡터를 구한다.

결과를 `--save-json <path>` 로 JSON 저장 가능 (옹벽 기울기 분석에 사용).

핵심 아이디어
-------------
1) Ground removal : 가장 두꺼운 수평 슬라브를 지면으로 보고 제거
2) XY 중심 탐지   : 지면 제거 후 점들을 XY로 누적해 가장 길게 수직으로 뻗은 셀 위치 찾기
3) Z 클리핑       : 봉 중간 60% 구간만 사용 — 머리·다리 분기점 회피
4) 좁은 반경 필터 : XY 중심에서 r_xy 이내만 추출 (1차 봉 후보)
5) IRLS 원기둥 피팅: 잔차에 Tukey 가중치, 인라이어 재선정·재피팅 반복
6) 검증           : PCA / Slice / Cylinder LSQ 세 축 일치도, 선형성, 잔차

출력: 축벡터, 중심, 반경, Z기울기, 검증지표
시각화(기본 on): 전체점군(회색) + 봉 인라이어(주황) + 축선(빨강, 4배 연장) + 중심구
"""

from __future__ import annotations
import argparse
import sys

import numpy as np
import open3d as o3d
from scipy.optimize import least_squares


# ---------- 1. Ground removal ----------
def remove_ground(pts, slab_height=0.10, min_density_ratio=0.40):
    """가장 두꺼운 수평 슬라브를 지면으로 간주해 제거.
    슬라브 두께 안에 전체 점의 min_density_ratio 이상이 들어있으면 지면."""
    z = pts[:, 2]
    z_min, z_max = z.min(), z.max()
    edges = np.arange(z_min, z_max + 0.01, 0.01)  # 1 cm bin
    hist, _ = np.histogram(z, bins=edges)
    # slab_height 길이의 윈도우에서 누적 카운트가 최대인 z 위치
    win = max(int(round(slab_height / 0.01)), 1)
    csum = np.concatenate([[0], np.cumsum(hist)])
    win_counts = csum[win:] - csum[:-win]
    if len(win_counts) == 0:
        return np.ones(len(pts), dtype=bool), z_min - 1
    k = int(np.argmax(win_counts))
    if win_counts[k] < min_density_ratio * len(pts):
        # 두꺼운 슬라브 없음 → 지면 없음
        return np.ones(len(pts), dtype=bool), z_min - 1
    z_ground_top = edges[k + win]
    mask = z > z_ground_top
    return mask, float(z_ground_top)


# ---------- 2. XY 중심 (봉 후보) 탐지 ----------
def find_rod_xy(pts, nbin=200, n_z_bins=20, occ_thresh=3, top_k=5):
    """수직 연속성(coverage) 기준으로 XY 셀 선택.

    각 XY 셀의 z 컬럼을 n_z_bins로 나눠, 점이 occ_thresh개 이상 들어있는
    z-bin 개수(coverage)를 센다. coverage가 가장 큰 셀이 진짜 수직봉.
    상위 top_k개를 출력해 디버깅에 활용.
    """
    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    xe = np.linspace(mn[0], mx[0], nbin + 1)
    ye = np.linspace(mn[1], mx[1], nbin + 1)
    ix = np.clip(np.searchsorted(xe, pts[:, 0]) - 1, 0, nbin - 1)
    iy = np.clip(np.searchsorted(ye, pts[:, 1]) - 1, 0, nbin - 1)
    z_lo, z_hi = mn[2], mx[2]
    iz = np.clip(((pts[:, 2] - z_lo) / max(z_hi - z_lo, 1e-9) * n_z_bins).astype(int),
                 0, n_z_bins - 1)
    cnt3 = np.zeros((nbin, nbin, n_z_bins), dtype=np.int32)
    np.add.at(cnt3, (ix, iy, iz), 1)
    coverage = (cnt3 >= occ_thresh).sum(axis=2)
    # 각 셀의 총 점수 — 같은 coverage라면 점 많은 쪽 선호
    cnt2 = cnt3.sum(axis=2)
    score = coverage.astype(np.float64) + 1e-6 * cnt2
    flat = np.argsort(score.ravel())[::-1][:top_k]
    cands = []
    for fi in flat:
        ii, jj = np.unravel_index(fi, score.shape)
        cands.append({
            "ix": int(ii), "iy": int(jj),
            "cx": float(0.5 * (xe[ii] + xe[ii + 1])),
            "cy": float(0.5 * (ye[jj] + ye[jj + 1])),
            "coverage": int(coverage[ii, jj]),
            "count": int(cnt2[ii, jj]),
        })
    best = cands[0]
    return best["cx"], best["cy"], best["coverage"], cands


# ---------- 3. 원기둥 IRLS 피팅 ----------
def _cyl_dist(c, d, r, P):
    return np.linalg.norm(np.cross(P - c, d), axis=1) - r


def _cyl_res(params, P, weights):
    c = params[:3]
    d = params[3:6]
    d = d / (np.linalg.norm(d) + 1e-12)
    r = params[6]
    return weights * _cyl_dist(c, d, r, P)


def fit_cylinder_irls(points, init_c, init_axis, init_r,
                     n_iter=5, tukey_c=2.5):
    """Tukey biweight IRLS. 매 iter마다 가중치 갱신."""
    c = np.array(init_c, dtype=float)
    d = np.array(init_axis, dtype=float)
    d /= np.linalg.norm(d)
    r = float(init_r)
    sample = points
    if len(sample) > 200_000:
        idx = np.random.default_rng(0).choice(len(sample), 200_000, replace=False)
        sample = sample[idx]

    weights = np.ones(len(sample))
    for it in range(n_iter):
        x0 = np.concatenate([c, d, [r]])
        res = least_squares(_cyl_res, x0, args=(sample, weights),
                            method="lm", max_nfev=200)
        c = res.x[:3]
        d = res.x[3:6]
        d /= np.linalg.norm(d)
        r = abs(float(res.x[6]))
        # 잔차 (가중치 없는)
        e = _cyl_dist(c, d, r, sample)
        mad = np.median(np.abs(e - np.median(e))) + 1e-9
        sigma = 1.4826 * mad
        u = e / (tukey_c * sigma)
        weights = np.where(np.abs(u) < 1, (1 - u * u) ** 2, 0.0)
    if d[2] < 0:
        d = -d
    return c, d, r, weights, sample


# ---------- 4. 검증 ----------
def pca_axis(P):
    c = P.mean(axis=0)
    w, v = np.linalg.eigh(np.cov((P - c).T))
    order = np.argsort(w)[::-1]
    w = w[order]; v = v[:, order]
    a = v[:, 0]
    if a[2] < 0:
        a = -a
    return c, a / np.linalg.norm(a), w


def _orth_basis(a):
    a = a / np.linalg.norm(a)
    h = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(a, h); e1 /= np.linalg.norm(e1)
    e2 = np.cross(a, e1)
    return e1, e2


def slice_axis(P, c, a, n=20):
    e1, e2 = _orth_basis(a)
    Pc = P - c
    s = Pc @ a
    edges = np.linspace(s.min(), s.max(), n + 1)
    centers = []
    for i in range(n):
        m = (s >= edges[i]) & (s < edges[i + 1])
        if m.sum() < 30:
            continue
        slab = Pc[m]
        xy = np.column_stack([slab @ e1, slab @ e2])
        # Kasa
        x, y = xy[:, 0], xy[:, 1]
        A = np.column_stack([2 * x, 2 * y, np.ones_like(x)])
        b = x * x + y * y
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        cx, cy = sol[0], sol[1]
        sm = 0.5 * (edges[i] + edges[i + 1])
        centers.append(c + sm * a + cx * e1 + cy * e2)
    centers = np.asarray(centers)
    if len(centers) < 3:
        return None, None
    c0 = centers.mean(axis=0)
    _, _, vt = np.linalg.svd(centers - c0, full_matrices=False)
    d = vt[0]
    if d[2] < 0:
        d = -d
    return c0, d / np.linalg.norm(d)


def angle_deg(a, b):
    return float(np.degrees(np.arccos(np.clip(abs(np.dot(a, b)), -1, 1))))


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ply_path")
    ap.add_argument("--r-xy", type=float, default=0.05,
                    help="XY 봉 반경 초기 추정 (m). 기본 5cm")
    ap.add_argument("--z-clip", type=float, default=0.20,
                    help="Z 양끝에서 잘라낼 비율 (머리·다리 분기점 회피). 기본 20%")
    ap.add_argument("--no-vis", action="store_true")
    ap.add_argument("--save", default="",
                    help="추출된 봉 인라이어를 PLY로 저장할 경로")
    ap.add_argument("--xy", nargs=2, type=float, default=None,
                    metavar=("CX", "CY"),
                    help="자동탐지 대신 수동으로 봉의 XY 중심을 지정")
    ap.add_argument("--save-json", default="",
                    help="plumb 결과를 JSON 으로 저장")
    args = ap.parse_args()

    pcd_all = o3d.io.read_point_cloud(args.ply_path)
    pts_all = np.asarray(pcd_all.points)
    print(f"[입력] {args.ply_path}   점 {len(pts_all):,}")

    # 1. ground removal
    mask_g, z_g = remove_ground(pts_all)
    pts_above = pts_all[mask_g]
    print(f"[ground] z_top = {z_g:.3f},  지면 위 점 {len(pts_above):,}"
          f"  ({100*len(pts_above)/len(pts_all):.1f}%)")

    # 2. XY 중심 탐지 — 연속성 기준
    if args.xy is None:
        cx, cy, cov, cands = find_rod_xy(pts_above, nbin=200, n_z_bins=20, occ_thresh=3, top_k=5)
        print(f"[XY 중심] ({cx:.3f}, {cy:.3f})  연속 z-bin coverage={cov}/20")
        print("  top-5 후보:")
        for k, c_ in enumerate(cands):
            print(f"    #{k} cx={c_['cx']:7.3f} cy={c_['cy']:7.3f} "
                  f"cov={c_['coverage']:>3}/20  count={c_['count']:>6}")
    else:
        cx, cy = args.xy
        print(f"[XY 중심] 수동 지정 ({cx:.3f}, {cy:.3f})")

    # 3. XY 좁은 반경 + Z 가운데 클리핑
    d_xy = np.hypot(pts_above[:, 0] - cx, pts_above[:, 1] - cy)
    m_xy = d_xy < args.r_xy
    cand = pts_above[m_xy]
    if len(cand) < 100:
        # 반경 자동 확장
        for r in (0.08, 0.12, 0.20):
            cand = pts_above[d_xy < r]
            if len(cand) >= 100:
                args.r_xy = r
                break
    print(f"[xy 필터] r_xy={args.r_xy:.3f}  점 {len(cand):,}")

    z = cand[:, 2]
    z_lo, z_hi = np.quantile(z, [args.z_clip, 1 - args.z_clip])
    cand_mid = cand[(z >= z_lo) & (z <= z_hi)]
    print(f"[z 클립] {z_lo:.3f}~{z_hi:.3f}  중간구간 점 {len(cand_mid):,}")

    # 4. 초기 축 = 수직, 중심 = 후보 평균
    init_c = cand_mid.mean(axis=0)
    init_axis = np.array([0.0, 0.0, 1.0])
    init_r = float(np.median(np.hypot(cand_mid[:, 0] - init_c[0],
                                       cand_mid[:, 1] - init_c[1])))
    print(f"[초기] center={init_c},  r={init_r:.4f}")

    # 5. IRLS
    c, axis, r, w, sample = fit_cylinder_irls(
        cand_mid, init_c, init_axis, init_r, n_iter=6, tukey_c=2.5
    )
    print(f"[IRLS cylinder] axis={axis}  r={r:.5f}")

    # 6. 인라이어 = 가중치 > 0.5 인 점 + 원본에서 같은 기준으로 재선정
    e_all = np.linalg.norm(np.cross(cand_mid - c, axis), axis=1) - r
    mad = np.median(np.abs(e_all - np.median(e_all))) + 1e-9
    sigma = 1.4826 * mad
    inlier_mask = np.abs(e_all) < 2.5 * sigma
    inliers = cand_mid[inlier_mask]

    # 원기둥 LSQ는 축 방향 위치를 결정하지 않으므로 중심을 인라이어 평균으로 재투영
    c = c + axis * float(((inliers.mean(axis=0) - c) @ axis))
    print(f"[inliers] {len(inliers):,} / {len(cand_mid):,}"
          f"  (σ={sigma*1000:.2f} mm)")

    # 7. 인라이어 PCA + 슬라이스 축 (검증)
    c_pca, axis_pca, eigvals = pca_axis(inliers)
    L = float(eigvals[0] / eigvals.sum())
    P_anis = float((eigvals[1] - eigvals[2]) / eigvals[0])
    c_slc, axis_slc = slice_axis(inliers, c_pca, axis_pca, n=20)

    ang_cyl_pca = angle_deg(axis, axis_pca)
    ang_cyl_slc = angle_deg(axis, axis_slc) if axis_slc is not None else float("nan")
    ang_pca_slc = angle_deg(axis_pca, axis_slc) if axis_slc is not None else float("nan")

    # 최종 봉 길이: 인라이어를 축에 투영
    proj = (inliers - c) @ axis
    rod_len = float(proj.max() - proj.min())
    tilt = angle_deg(axis, np.array([0, 0, 1.0]))

    print()
    print("==================== 결과 ====================")
    print(f"중심 (cylinder)       : ({c[0]:.4f}, {c[1]:.4f}, {c[2]:.4f})")
    print(f"축벡터               : ({axis[0]:.6f}, {axis[1]:.6f}, {axis[2]:.6f})")
    print(f"반경                 : {r*1000:.2f} mm")
    print(f"봉 길이(inlier 투영) : {rod_len:.3f} m")
    print(f"Z축 대비 기울기      : {tilt:.4f} deg")
    print(f"[검증] PCA-축          : ({axis_pca[0]:.6f}, {axis_pca[1]:.6f}, {axis_pca[2]:.6f})")
    if axis_slc is not None:
        print(f"[검증] Slice-축        : ({axis_slc[0]:.6f}, {axis_slc[1]:.6f}, {axis_slc[2]:.6f})")
    print(f"[일치도] Cyl↔PCA      : {ang_cyl_pca:.4f} deg")
    print(f"[일치도] Cyl↔Slice    : {ang_cyl_slc:.4f} deg")
    print(f"[일치도] PCA↔Slice    : {ang_pca_slc:.4f} deg")
    print(f"[형태] 선형성 L       : {L:.4f}   planarity P : {P_anis:.4f}")
    print(f"[표면] 잔차 σ         : {sigma*1000:.3f} mm")
    pass_axis = max(ang_cyl_pca, ang_cyl_slc) < 1.0
    pass_shape = L > 0.95 and P_anis < 0.05
    pass_surf = sigma < 0.5 * r
    overall = pass_axis and pass_shape and pass_surf
    print(f"[판정] 축 일치<1°       : {'PASS' if pass_axis else 'FAIL'}")
    print(f"[판정] 막대 형태        : {'PASS' if pass_shape else 'FAIL'}")
    print(f"[판정] 원기둥 표면      : {'PASS' if pass_surf else 'FAIL'}")
    print(f">>> 종합                : {'PROVEN' if overall else 'NOT PROVEN'}")

    if args.save:
        out = o3d.geometry.PointCloud()
        out.points = o3d.utility.Vector3dVector(inliers)
        out.paint_uniform_color([1.0, 0.5, 0.0])
        o3d.io.write_point_cloud(args.save, out)
        print(f"[saved] inliers → {args.save}")

    if args.save_json:
        import json
        result = {
            "source_ply": args.ply_path,
            "method": "irls_cylinder_lsq",
            "axis": [float(axis[0]), float(axis[1]), float(axis[2])],
            "center": [float(c[0]), float(c[1]), float(c[2])],
            "radius_m": float(r),
            "rod_length_m": float(rod_len),
            "tilt_from_z_deg": float(tilt),
            "n_inliers": int(len(inliers)),
            "surface_sigma_m": float(sigma),
            "validation": {
                "cylinder_vs_pca_deg": float(ang_cyl_pca),
                "cylinder_vs_slice_deg": float(ang_cyl_slc),
                "pca_vs_slice_deg": float(ang_pca_slc),
                "linearity": float(L),
                "planarity": float(P_anis),
                "axis_match_pass": bool(pass_axis),
                "shape_pass": bool(pass_shape),
                "surface_pass": bool(pass_surf),
                "overall_proven": bool(overall),
            },
        }
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"[saved] plumb JSON → {args.save_json}")

    if args.no_vis:
        return

    # 시각화
    # 1) 전체 점군 회색(다운샘플)
    full = pcd_all
    if len(pts_all) > 500_000:
        full = full.voxel_down_sample(0.01)
    full.paint_uniform_color([0.55, 0.57, 0.60])
    # 2) 인라이어 주황
    in_pcd = o3d.geometry.PointCloud()
    in_pcd.points = o3d.utility.Vector3dVector(inliers)
    in_pcd.paint_uniform_color([1.0, 0.45, 0.0])
    # 3) 축선 (인라이어 길이 4배)
    L_draw = rod_len * 2.0
    p0 = c - axis * L_draw
    p1 = c + axis * L_draw
    line = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.vstack([p0, p1])),
        lines=o3d.utility.Vector2iVector([[0, 1]]),
    )
    line.colors = o3d.utility.Vector3dVector([[1, 0, 0]])
    # 4) 중심 구
    sph = o3d.geometry.TriangleMesh.create_sphere(radius=max(r * 2.0, 0.01))
    sph.translate(c)
    sph.paint_uniform_color([1.0, 1.0, 0.0])

    o3d.visualization.draw_geometries(
        [full, in_pcd, line, sph],
        window_name="rod extraction — gray=all  orange=rod inliers  red=axis",
    )


if __name__ == "__main__":
    main()
