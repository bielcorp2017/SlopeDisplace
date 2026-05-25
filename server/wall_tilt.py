"""옹벽 기울기 변화 회귀 — server endpoint·CLI 공용 모듈.

핵심: 같은 옹벽을 두 번 스캔하고 ICP로 정합한 후, 옹벽 표면점의 변위를
plumb 축 높이에 대해 회귀해 "단위 높이당 outward 변위 증가율"(α, mm/m)을
구한다. α 가 0 이면 평행이동, α≠0 이면 추가 lean.

`signed_normal` 채널은 이미 *_disp.bin 에 들어있으므로, 따로 계산할
필요 없이 reference 노멀만 마스킹용으로 다시 추정한다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import open3d as o3d
from scipy.optimize import least_squares
from scipy.spatial import cKDTree


def load_plumb(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    axis = np.asarray(data["axis"], dtype=np.float64)
    origin = np.asarray(data["center"], dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    return axis, origin, data


def estimate_ref_normals(ref_simple_path: Path, radius: float = 0.5, k: int = 30):
    pcd = o3d.io.read_point_cloud(str(ref_simple_path))
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=k)
    )
    pcd.orient_normals_consistent_tangent_plane(k=20)
    pts = np.asarray(pcd.points, dtype=np.float64)
    n = np.asarray(pcd.normals, dtype=np.float64)
    return pts, n


def huber_linear_fit(x: np.ndarray, y: np.ndarray, c: float = 1.345):
    def resid(p):
        return (p[0] * x + p[1]) - y
    p0 = np.array([0.0, float(np.median(y))])
    f_scale = c * float(np.std(y)) + 1e-9
    res = least_squares(resid, p0, loss="huber", f_scale=f_scale, max_nfev=200)
    alpha, beta = float(res.x[0]), float(res.x[1])
    e = y - (alpha * x + beta)
    sigma = 1.4826 * float(np.median(np.abs(e - np.median(e))))
    return alpha, beta, sigma


def bootstrap_ci(x: np.ndarray, y: np.ndarray, n_boot: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = len(x)
    alphas = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        a, _, _ = huber_linear_fit(x[idx], y[idx])
        alphas[i] = a
    lo, hi = np.quantile(alphas, [0.025, 0.975])
    return float(lo), float(hi), float(alphas.std())


def compute_wall_tilt(
    dataset: str,
    target_stem: str,
    data_root: Path,
    mask_nz: float = 0.3,
    mag_min_m: float = 0.001,
    mag_max_m: float = 0.50,
    n_bootstrap: int = 200,
    n_scatter_sample: int = 3000,
    plumb_path: Optional[Path] = None,
) -> dict:
    """옹벽 기울기 변화를 회귀로 계산해 dict 반환.

    raises
    ------
    FileNotFoundError, ValueError
    """
    root = Path(data_root) / dataset
    if plumb_path is None:
        plumb_path = root / "plumb.json"
    target_simple = root / f"{target_stem}_simple.ply"
    disp_path = root / f"{target_stem}_disp.bin"
    meta_path = root / f"{target_stem}_meta.json"

    for p in (plumb_path, target_simple, disp_path, meta_path):
        if not p.exists():
            raise FileNotFoundError(str(p))

    axis, origin, plumb_meta = load_plumb(plumb_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    ref_name = meta.get("reference")
    if not ref_name:
        raise ValueError(f"{target_stem} 은 reference 가 없습니다 (자기 자신이 reference?)")
    ref_stem = Path(ref_name).stem
    ref_simple = root / f"{ref_stem}_simple.ply"
    if not ref_simple.exists():
        raise FileNotFoundError(str(ref_simple))

    # registration quality (from meta) — None 이면 C++ pipeline 처리물
    icp_fitness = meta.get("final_icp_fitness")
    icp_history = meta.get("icp_history") or []
    if icp_fitness is None and icp_history:
        icp_fitness = icp_history[-1].get("fitness")
    registration_reliable = meta.get("registration_reliable")
    if registration_reliable is None and icp_fitness is not None:
        registration_reliable = icp_fitness >= 0.30

    # target points (in ref frame)
    pcd_t = o3d.io.read_point_cloud(str(target_simple))
    P_t = np.asarray(pcd_t.points, dtype=np.float64)
    n_pts = len(P_t)

    disp = np.fromfile(disp_path, dtype=np.float32).reshape(-1, 4)
    if len(disp) != n_pts:
        raise ValueError(f"disp count mismatch: {len(disp)} vs {n_pts}")
    signed_n = disp[:, 0].astype(np.float64)   # m
    mag = disp[:, 1].astype(np.float64)        # m

    # reference normals for wall mask
    P_r, N_r = estimate_ref_normals(ref_simple)
    tree = cKDTree(P_r)
    _, idx = tree.query(P_t, k=1, workers=-1)
    n_at_t = N_r[idx]
    dot_plumb = np.abs(n_at_t @ axis)
    mask_wall = dot_plumb < mask_nz
    mask_mag = (mag > mag_min_m) & (mag < mag_max_m)
    mask = mask_wall & mask_mag

    if int(mask.sum()) < 100:
        raise ValueError(f"too few wall points after masking (n={int(mask.sum())})")

    h_all = (P_t - origin) @ axis          # m
    h_w = h_all[mask]
    d_w = signed_n[mask] * 1000.0          # mm

    # 옹벽 중심(인라이어 평균) + 평균 facing 방향(수평 노멀 평균)
    P_wall = P_t[mask]
    wall_centroid = P_wall.mean(axis=0)
    n_wall = n_at_t[mask]
    n_xy = n_wall.copy()
    n_xy[:, 2] = 0.0
    n_xy_norm = np.linalg.norm(n_xy, axis=1)
    valid_n = n_xy_norm > 1e-6
    n_xy_unit = np.zeros_like(n_xy)
    n_xy_unit[valid_n] = n_xy[valid_n] / n_xy_norm[valid_n, None]
    # facing 방향: 노멀의 평균. 옹벽 표면이 보는 방향(outward).
    facing_mean = n_xy_unit.mean(axis=0)
    fn = float(np.linalg.norm(facing_mean))
    facing_dir = facing_mean / fn if fn > 1e-6 else np.array([1.0, 0.0, 0.0])
    h_ref = float(np.median(h_w))
    x = h_w - h_ref
    y = d_w

    alpha, beta, resid_sigma = huber_linear_fit(x, y)
    ci_lo, ci_hi, alpha_std = bootstrap_ci(x, y, n_boot=n_bootstrap)
    alpha_deg = float(np.degrees(alpha / 1000.0))
    ci_lo_deg = float(np.degrees(ci_lo / 1000.0))
    ci_hi_deg = float(np.degrees(ci_hi / 1000.0))
    tilt_dir = "OUTWARD" if alpha > 0 else "INWARD"
    significant = not (ci_lo <= 0 <= ci_hi)
    verdict = (f"SIGNIFICANT TILT CHANGE ({tilt_dir})"
               if significant else "NO SIGNIFICANT TILT CHANGE (0 within 95% CI)")
    ss_res = float(np.sum((y - (alpha * x + beta)) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2)) + 1e-12
    r2 = 1.0 - ss_res / ss_tot

    # subsample scatter for client-side plotting
    if len(x) > n_scatter_sample:
        s_idx = np.random.default_rng(0).choice(len(x), n_scatter_sample, replace=False)
    else:
        s_idx = np.arange(len(x))
    scatter_h = (h_w[s_idx]).astype(np.float32)
    scatter_d = d_w[s_idx].astype(np.float32)

    result = {
        "dataset": dataset,
        "reference": ref_stem,
        "target": target_stem,
        "icp_fitness": float(icp_fitness) if icp_fitness is not None else None,
        "registration_reliable": (None if registration_reliable is None
                                  else bool(registration_reliable)),
        "plumb_axis": axis.tolist(),
        "plumb_origin": origin.tolist(),
        "wall_face_point_count": int(mask.sum()),
        "total_point_count": int(n_pts),
        "wall_centroid_xyz": [float(v) for v in wall_centroid],
        "wall_facing_xy": [float(facing_dir[0]), float(facing_dir[1]), float(facing_dir[2])],
        "height_range_m": [float(h_w.min()), float(h_w.max())],
        "h_ref_m": h_ref,
        "alpha_mm_per_m": alpha,
        "alpha_ci95_mm_per_m": [ci_lo, ci_hi],
        "alpha_bootstrap_std_mm_per_m": alpha_std,
        "beta_mm": beta,
        "residual_robust_sigma_mm": resid_sigma,
        "r_squared": r2,
        "tilt_change_deg": alpha_deg,
        "tilt_change_ci95_deg": [ci_lo_deg, ci_hi_deg],
        "tilt_direction": tilt_dir,
        "significant": bool(significant),
        "verdict": verdict,
        "params": {
            "mask_nz": mask_nz,
            "mag_min_m": mag_min_m,
            "mag_max_m": mag_max_m,
            "n_bootstrap": n_bootstrap,
        },
        # scatter for chart: parallel arrays
        "scatter_h_m": scatter_h.tolist(),
        "scatter_d_mm": scatter_d.tolist(),
    }
    return result


def save_result(result: dict, out_path: Path) -> None:
    out_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
