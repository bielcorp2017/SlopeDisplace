"""tripod PLY 내부 구조 파악용 진단 스크립트.

목적: PCA/슬라이스/원기둥 LSQ가 모두 실패한 이유를 데이터로 확인하고,
어떤 부분이 '수직봉'에 해당하는지 분리 가능성을 본다.
"""
from __future__ import annotations
import sys
import numpy as np
import open3d as o3d


def main(path: str):
    pcd = o3d.io.read_point_cloud(path)
    pts = np.asarray(pcd.points)
    has_color = pcd.has_colors()
    cols = np.asarray(pcd.colors) if has_color else None
    print(f"파일                 : {path}")
    print(f"점 개수              : {len(pts):,}")
    print(f"색상 보유            : {has_color}")
    print()

    mn, mx = pts.min(axis=0), pts.max(axis=0)
    ext = mx - mn
    print(f"AABB min             : ({mn[0]:.3f}, {mn[1]:.3f}, {mn[2]:.3f})")
    print(f"AABB max             : ({mx[0]:.3f}, {mx[1]:.3f}, {mx[2]:.3f})")
    print(f"AABB extent (XYZ)    : ({ext[0]:.3f}, {ext[1]:.3f}, {ext[2]:.3f})")
    print()

    # Z 분포 — 수직봉이 길게 뻗어있다면 Z 폭이 X/Y보다 클 것
    for name, arr in [("X", pts[:, 0]), ("Y", pts[:, 1]), ("Z", pts[:, 2])]:
        q = np.quantile(arr, [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0])
        print(f"{name} 분포  min/5%/Q1/med/Q3/95%/max:  "
              + " / ".join(f"{v:7.3f}" for v in q))
    print()

    # XY 히스토그램 → 봉의 평면상 footprint 위치 추정
    nbin = 80
    H, xe, ye = np.histogram2d(
        pts[:, 0], pts[:, 1], bins=nbin,
        range=[[mn[0], mx[0]], [mn[1], mx[1]]],
    )
    # 각 XY 셀의 Z 폭(extent) — 봉처럼 길게 수직인 셀은 큰 값
    zmin_grid = np.full((nbin, nbin), np.inf)
    zmax_grid = np.full((nbin, nbin), -np.inf)
    ix = np.clip(np.searchsorted(xe, pts[:, 0]) - 1, 0, nbin - 1)
    iy = np.clip(np.searchsorted(ye, pts[:, 1]) - 1, 0, nbin - 1)
    np.minimum.at(zmin_grid, (ix, iy), pts[:, 2])
    np.maximum.at(zmax_grid, (ix, iy), pts[:, 2])
    z_ext = np.where(np.isfinite(zmin_grid), zmax_grid - zmin_grid, 0.0)

    # 가장 큰 Z 폭을 가진 셀 찾기 (봉 후보)
    flat_idx = np.argsort(z_ext.ravel())[::-1][:10]
    print("XY 셀별 Z 폭 상위 10개 (봉/세로 구조 후보)")
    print(f"{'rank':<5}{'(ix,iy)':<12}{'cx':>8}{'cy':>8}{'count':>8}{'z_extent':>10}")
    for rk, fi in enumerate(flat_idx):
        ii, jj = np.unravel_index(fi, z_ext.shape)
        cx = 0.5 * (xe[ii] + xe[ii + 1])
        cy = 0.5 * (ye[jj] + ye[jj + 1])
        print(f"{rk:<5}({ii:>3},{jj:>3})   {cx:8.3f}{cy:8.3f}{int(H[ii,jj]):>8}{z_ext[ii,jj]:>10.3f}")
    print()

    # 추정된 봉 footprint 안에 들어가는 점만 모아 다시 PCA 해보기
    best = flat_idx[0]
    ii, jj = np.unravel_index(best, z_ext.shape)
    cx0 = 0.5 * (xe[ii] + xe[ii + 1])
    cy0 = 0.5 * (ye[jj] + ye[jj + 1])
    # 셀 + 인접 8셀 또는 일정 반경
    bin_w = max((mx[0] - mn[0]) / nbin, (mx[1] - mn[1]) / nbin)
    r_xy = bin_w * 1.5
    mask = (np.hypot(pts[:, 0] - cx0, pts[:, 1] - cy0) < r_xy)
    sub = pts[mask]
    print(f"봉 후보 footprint  (cx={cx0:.3f}, cy={cy0:.3f}, r_xy={r_xy:.3f})")
    print(f"  포함 점 개수       : {len(sub):,}   ({100*len(sub)/len(pts):.2f}%)")
    if len(sub) >= 50:
        c = sub.mean(axis=0)
        cov = np.cov((sub - c).T)
        w, v = np.linalg.eigh(cov)
        order = np.argsort(w)[::-1]
        w = w[order]; v = v[:, order]
        ax = v[:, 0]
        if ax[2] < 0: ax = -ax
        L = w[0] / w.sum()
        P = (w[1] - w[2]) / w[0]
        z = np.array([0, 0, 1.0])
        tilt = np.degrees(np.arccos(np.clip(abs(ax @ z), -1, 1)))
        zmin = sub[:, 2].min()
        zmax = sub[:, 2].max()
        print(f"  부분 PCA 축         : ({ax[0]:.4f}, {ax[1]:.4f}, {ax[2]:.4f})")
        print(f"  Z축 대비 기울기     : {tilt:.3f} deg")
        print(f"  선형성 L            : {L:.4f}")
        print(f"  planarity P         : {P:.4f}")
        print(f"  Z 범위              : {zmin:.3f} ~ {zmax:.3f}  (높이 {zmax-zmin:.3f})")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/CH2_RETAINWALL/20260522_tripod.ply")
