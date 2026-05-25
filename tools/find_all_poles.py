"""점군 안의 모든 vertical pole 후보를 분리해 찾는다.

자동탐지가 한 곳만 잡지 않게, 일정 거리 이상 떨어진 후보들을
non-maximum suppression 으로 모아 보여준다.
"""
from __future__ import annotations
import sys
import numpy as np
import open3d as o3d


def remove_ground(pts, slab=0.10, ratio=0.40):
    z = pts[:, 2]
    edges = np.arange(z.min(), z.max() + 0.01, 0.01)
    h, _ = np.histogram(z, bins=edges)
    win = max(int(round(slab / 0.01)), 1)
    cs = np.concatenate([[0], np.cumsum(h)])
    wc = cs[win:] - cs[:-win]
    if len(wc) == 0 or wc.max() < ratio * len(pts):
        return pts, None
    k = int(np.argmax(wc))
    z_top = edges[k + win]
    return pts[z > z_top], float(z_top)


def main(path: str):
    pcd = o3d.io.read_point_cloud(path)
    pts = np.asarray(pcd.points)
    above, z_top = remove_ground(pts)
    print(f"전체 {len(pts):,}, 지면제거 후 {len(above):,} (z_top={z_top})")

    nbin = 240
    n_z = 25
    occ_thresh = 3

    mn = above.min(axis=0); mx = above.max(axis=0)
    xe = np.linspace(mn[0], mx[0], nbin + 1)
    ye = np.linspace(mn[1], mx[1], nbin + 1)
    ix = np.clip(np.searchsorted(xe, above[:, 0]) - 1, 0, nbin - 1)
    iy = np.clip(np.searchsorted(ye, above[:, 1]) - 1, 0, nbin - 1)
    iz = np.clip(((above[:, 2] - mn[2]) / max(mx[2] - mn[2], 1e-9) * n_z).astype(int),
                 0, n_z - 1)
    cnt3 = np.zeros((nbin, nbin, n_z), dtype=np.int32)
    np.add.at(cnt3, (ix, iy, iz), 1)
    coverage = (cnt3 >= occ_thresh).sum(axis=2)
    cnt2 = cnt3.sum(axis=2)

    # NMS: 6cm 이내 다른 봉 후보 억제
    bin_w = max((mx[0] - mn[0]) / nbin, (mx[1] - mn[1]) / nbin)
    nms_r = int(np.ceil(0.06 / bin_w))
    used = np.zeros_like(coverage, dtype=bool)
    cands = []
    for _ in range(20):
        score = np.where(used, -1, coverage)
        fi = int(np.argmax(score))
        ii, jj = np.unravel_index(fi, score.shape)
        if coverage[ii, jj] < 5:
            break
        cands.append({
            "cx": float(0.5 * (xe[ii] + xe[ii + 1])),
            "cy": float(0.5 * (ye[jj] + ye[jj + 1])),
            "coverage": int(coverage[ii, jj]),
            "count": int(cnt2[ii, jj]),
        })
        i0 = max(0, ii - nms_r); i1 = min(nbin, ii + nms_r + 1)
        j0 = max(0, jj - nms_r); j1 = min(nbin, jj + nms_r + 1)
        used[i0:i1, j0:j1] = True

    print(f"\nNMS r = {nms_r * bin_w:.3f} m, 발견된 후보 {len(cands)}개")
    print(f"{'rank':<5}{'cx':>8}{'cy':>8}{'cov':>6}{'count':>8}")
    for k, c in enumerate(cands):
        print(f"{k:<5}{c['cx']:>8.3f}{c['cy']:>8.3f}{c['coverage']:>6}{c['count']:>8}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/CH2_RETAINWALL/20260522_tripod.ply")
