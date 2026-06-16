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


def local_patch_tilts(
    P: np.ndarray,
    axis: np.ndarray,
    facing_dir: np.ndarray,
    cell: float = 0.6,
    min_pts: int = 40,
    planar_ratio: float = 0.10,
):
    """벽면을 voxel 셀로 나눠 셀마다 국소 평면(PCA)을 피팅하고, 각 셀의
    '수직 대비 outward 기울기(deg)'를 반환한다.

    평면도상 사행(위에서 봤을 때 꼬불꼬불)하는 벽이라도 각 셀은 국소적으로
    평면이므로 단일 전역 평면보다 정확하다. 수직 기울기 성분(n·axis)은 수평
    방위에 불변이라 방위가 달라지는 셀들의 값도 그대로 집계할 수 있다.
    사행 변곡부(두 방위가 섞인 비평면) 셀은 평면성 검사로 자동 제외된다.

    returns
    -------
    tilts_deg : (M,) 셀별 기울기
    counts    : (M,) 셀별 점 개수
    rms_mm    : (M,) 셀별 평면 잔차 RMS(mm) — 타일 요철 지표
    cents     : (M,3) 셀별 점 무게중심 — 상/하부 구간 분리용 높이 계산에 사용
    """
    keys = np.floor(P / cell).astype(np.int64)
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    inv_s = inv[order]
    P_s = P[order]
    bounds = np.searchsorted(inv_s, np.arange(len(uniq) + 1))

    tilts, counts, rms, cents = [], [], [], []
    for ci in range(len(uniq)):
        s, e = int(bounds[ci]), int(bounds[ci + 1])
        n_pts = e - s
        if n_pts < min_pts:
            continue
        pts = P_s[s:e]
        ctr = pts.mean(axis=0)
        Q = pts - ctr
        w, V = np.linalg.eigh(Q.T @ Q)  # 고유값 오름차순
        # 평면성: 최소 고유값 << 중간 고유값 이어야 국소 평면. 사행 변곡 셀 제외.
        if w[0] > planar_ratio * w[1]:
            continue
        n = V[:, 0]                      # 최소 고유벡터 = 평면 노멀
        if (n @ facing_dir) < 0:         # outward 로 부호 정렬
            n = -n
        vc = float(np.clip(n @ axis, -1.0, 1.0))
        tilts.append(np.degrees(np.arcsin(vc)))
        counts.append(n_pts)
        rms.append(np.sqrt(max(w[0], 0.0) / n_pts) * 1000.0)  # off-plane RMS(mm)
        cents.append(ctr)
    return (np.array(tilts), np.array(counts), np.array(rms),
            np.array(cents).reshape(-1, 3))


def split_upper_lower(
    h: np.ndarray,
    tilts: np.ndarray,
    min_frac: float = 0.15,
    min_cells: int = 15,
    min_gap_deg: float = 0.5,
) -> Optional[dict]:
    """셀 (높이, 기울기) 에서 상/하부 경계를 changepoint 로 탐지.

    옹벽이 '하부 설계 경사 + 상부 수직' 2단 구성일 때, 높이순으로 정렬한 셀
    기울기의 계단 변화를 2-구간 robust(중앙값 절대편차) 비용 최소화로 찾는다.
    두 구간의 median 차이가 min_gap_deg 미만이면 설계 경사 구분이 없다고 보고
    None 을 반환한다(전체가 한 구간).

    returns: {"split_h_m", "lower": {...}, "upper": {...}} 또는 None
    """
    n = len(h)
    k = max(min_cells, int(n * min_frac))
    if n < 2 * k:
        return None
    order = np.argsort(h)
    hs, ts = h[order], tilts[order]
    best_i, best_cost = None, np.inf
    for i in range(k, n - k):
        lo, hi = ts[:i], ts[i:]
        cost = (np.abs(lo - np.median(lo)).sum()
                + np.abs(hi - np.median(hi)).sum())
        if cost < best_cost:
            best_cost, best_i = cost, i
    lo, hi = ts[:best_i], ts[best_i:]
    med_lo, med_hi = float(np.median(lo)), float(np.median(hi))
    if abs(med_hi - med_lo) < min_gap_deg:
        return None

    def seg(vals: np.ndarray) -> dict:
        return {
            "tilt_deg": float(np.median(vals)),
            "iqr_deg": float(np.percentile(vals, 75) - np.percentile(vals, 25)),
            "cell_count": int(len(vals)),
        }

    return {
        "split_h_m": float((hs[best_i - 1] + hs[best_i]) / 2.0),
        "lower": seg(lo),
        "upper": seg(hi),
    }


def compute_wall_tilt(
    dataset: str,
    target_stem: str,
    data_root: Path,
    mask_nz: float = 0.3,
    mag_min_m: float = 0.001,
    mag_max_m: float = 0.50,
    n_bootstrap: int = 200,
    n_scatter_sample: int = 3000,
    patch_cell_m: float = 0.6,
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

    # 절대 기울기 (수직 대비) — 국소 패치별 평면 피팅 + 강건 집계.
    # 벽이 평면도상 사행(위에서 봤을 때 꼬불꼬불)해도 각 voxel 셀은 국소 평면
    # 이므로 단일 전역 평면보다 정확. 수직 기울기 성분은 수평 방위에 불변이라
    # 방위가 변하는 셀들의 값도 그대로 집계 가능.
    # 부호: 양수 = OUTWARD lean (상부가 base 보다 바깥으로 기울어짐).
    #
    # 평면은 reference scan 의 벽면 점에 피팅한다(기준 형상). reference 벽 점이
    # 부족하면 정합된 target 벽 점으로 fallback.
    mask_wall_ref = np.abs(N_r @ axis) < mask_nz
    P_plane = P_r[mask_wall_ref]
    if len(P_plane) < 100:
        P_plane = P_wall  # fallback: 정합된 target 벽 점

    patch_tilts, patch_counts, patch_rms, patch_cents = local_patch_tilts(
        P_plane, axis, facing_dir, cell=patch_cell_m
    )
    abs_tilt_method = "local_patch_planes"
    if len(patch_tilts) >= 3:
        ref_abs_tilt_deg = float(np.median(patch_tilts))
        ref_abs_tilt_iqr_deg = float(
            np.percentile(patch_tilts, 75) - np.percentile(patch_tilts, 25)
        )
        patch_offplane_rms_mm = float(np.median(patch_rms))
    else:
        # 패치가 너무 적으면 점별 노멀 median 으로 fallback
        abs_tilt_method = "per_point_median"
        sign_outward = np.sign(n_wall @ facing_dir)
        sign_outward[sign_outward == 0] = 1.0
        n_oriented = n_wall * sign_outward[:, None]
        vert_comp = np.clip(n_oriented @ axis, -1.0, 1.0)
        tpp = np.degrees(np.arcsin(vert_comp))
        ref_abs_tilt_deg = float(np.median(tpp))
        ref_abs_tilt_iqr_deg = float(
            np.percentile(tpp, 75) - np.percentile(tpp, 25)
        )
        patch_offplane_rms_mm = float("nan")
    patch_count = int(len(patch_tilts))

    # 상/하부 구간 분리 — 하부는 설계상 경사(batter), 상부는 수직 설계.
    # 셀 높이에 따른 기울기 계단 변화를 탐지해 상부만의 절대 기울기를 따로 구한다.
    if abs_tilt_method == "local_patch_planes" and len(patch_tilts) >= 30:
        h_cells = (patch_cents - origin) @ axis
        sp = split_upper_lower(h_cells, patch_tilts)
        if sp is not None:
            sections = {"detected": True, **sp}
        else:
            sections = {"detected": False, "reason": "no_breakpoint"}
    else:
        sections = {"detected": False, "reason": "too_few_patches"}

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

    # 상부 전용 변화량 회귀 — 경계 위 점만으로 α 를 다시 피팅해
    # 상부의 "현재(target) 절대 기울기"를 상부 기준값 + 상부 변화량으로 구한다.
    if sections.get("detected"):
        m_up = h_w >= sections["split_h_m"]
        n_up = int(m_up.sum())
        sections["upper_point_count"] = n_up
        if n_up >= 200:
            a_u, _, _ = huber_linear_fit(
                h_w[m_up] - float(np.median(h_w[m_up])), d_w[m_up]
            )
            deg_u = float(np.degrees(a_u / 1000.0))
            sections["upper"]["alpha_mm_per_m"] = float(a_u)
            sections["upper"]["tilt_change_deg"] = deg_u
            sections["upper"]["target_tilt_deg"] = (
                sections["upper"]["tilt_deg"] + deg_u
            )
        else:
            # 상부 점 부족 — 전체 벽 변화량으로 근사
            sections["upper"]["alpha_mm_per_m"] = None
            sections["upper"]["tilt_change_deg"] = None
            sections["upper"]["target_tilt_deg"] = (
                sections["upper"]["tilt_deg"] + alpha_deg
            )

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
        # 절대 기울기 (수직 대비). reference = 기준 스캔, target = reference + 변화량.
        # 양수 = OUTWARD lean (상부가 base 보다 바깥으로), 음수 = INWARD lean.
        "reference_absolute_tilt_deg": ref_abs_tilt_deg,
        "reference_absolute_tilt_iqr_deg": ref_abs_tilt_iqr_deg,
        "target_absolute_tilt_deg": ref_abs_tilt_deg + alpha_deg,
        # 상/하부 2단 구성 분리 결과 — 상부(수직 설계부)만의 절대 기울기.
        # detected=False 면 경계 미탐지(단일 구간) 또는 패치 부족.
        "wall_sections": sections,
        # 절대 기울기 산출 방식 + 국소 패치 평면 피팅 품질
        "abs_tilt_method": abs_tilt_method,
        "patch_cell_m": patch_cell_m,
        "patch_count": patch_count,
        "patch_offplane_rms_mm": patch_offplane_rms_mm,
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


def compute_wall_tilt_debug(
    dataset: str,
    target_stem: str,
    data_root: Path,
    mask_nz: float = 0.3,
    cell: float = 0.6,
    min_pts: int = 40,
    planar_ratio: float = 0.10,
    max_wall_sample: int = 25000,
    max_other_sample: int = 12000,
    plumb_path: Optional[Path] = None,
) -> dict:
    """기울기 계산 과정 시각화용 디버그 데이터.

    compute_wall_tilt 의 절대-기울기 경로(local_patch_planes)를 그대로 재현하되,
    각 voxel 셀의 기하(박스 위치·노멀·기울기·포함/제외 사유)와 샘플 점을 함께
    내보내 프론트가 3D 로 단계별 시각화할 수 있게 한다. local_patch_tilts 와
    동일한 임계값을 쓰며, 그 함수가 진실의 출처(source of truth)다.

    절대 기울기는 reference 스캔의 벽면 점에 피팅하므로 여기서도 reference 를
    사용한다(정합·변위와 무관하게 한 스캔의 형상만 본다).
    """
    root = Path(data_root) / dataset
    if plumb_path is None:
        plumb_path = root / "plumb.json"
    meta_path = root / f"{target_stem}_meta.json"
    for p in (plumb_path, meta_path):
        if not p.exists():
            raise FileNotFoundError(str(p))

    axis, origin, _ = load_plumb(plumb_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    ref_name = meta.get("reference")
    if not ref_name:
        raise ValueError(f"{target_stem} 은 reference 가 없습니다 (자기 자신이 reference?)")
    ref_stem = Path(ref_name).stem
    ref_simple = root / f"{ref_stem}_simple.ply"
    if not ref_simple.exists():
        raise FileNotFoundError(str(ref_simple))

    P_r, N_r = estimate_ref_normals(ref_simple)
    dot_plumb = np.abs(N_r @ axis)
    mask_wall = dot_plumb < mask_nz          # 노멀이 수평 → 벽면
    P_wall = P_r[mask_wall]
    N_wall = N_r[mask_wall]
    if len(P_wall) < 100:
        raise ValueError(f"too few wall points after masking (n={len(P_wall)})")

    # facing 방향: 벽면 노멀의 수평성분 평균(outward)
    n_xy = N_wall.copy()
    n_xy[:, 2] = 0.0
    nn = np.linalg.norm(n_xy, axis=1)
    valid = nn > 1e-6
    u = np.zeros_like(n_xy)
    u[valid] = n_xy[valid] / nn[valid, None]
    facing = u.mean(axis=0)
    fn = float(np.linalg.norm(facing))
    facing_dir = facing / fn if fn > 1e-6 else np.array([1.0, 0.0, 0.0])

    # voxelize (local_patch_tilts 와 동일한 키/정렬)
    keys = np.floor(P_wall / cell).astype(np.int64)
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    inv_s = inv[order]
    P_s = P_wall[order]
    bounds = np.searchsorted(inv_s, np.arange(len(uniq) + 1))

    cells = []
    tilts_included = []
    h_included = []
    for ci in range(len(uniq)):
        s, e = int(bounds[ci]), int(bounds[ci + 1])
        n_pts = e - s
        key = uniq[ci]
        rec = {"key": [int(key[0]), int(key[1]), int(key[2])], "n_pts": int(n_pts)}
        if n_pts < min_pts:
            rec["included"] = False
            rec["reason"] = "too_few"
            cells.append(rec)
            continue
        pts = P_s[s:e]
        ctr = pts.mean(axis=0)
        Q = pts - ctr
        w, V = np.linalg.eigh(Q.T @ Q)       # 고유값 오름차순
        n = V[:, 0]                          # 최소 고유벡터 = 평면 노멀
        if (n @ facing_dir) < 0:             # outward 부호 정렬
            n = -n
        vc = float(np.clip(n @ axis, -1.0, 1.0))
        tilt = float(np.degrees(np.arcsin(vc)))
        rms = float(np.sqrt(max(w[0], 0.0) / n_pts) * 1000.0)
        planarity = float(w[0] / (w[1] + 1e-12))
        rec["centroid"] = [float(ctr[0]), float(ctr[1]), float(ctr[2])]
        rec["normal"] = [float(n[0]), float(n[1]), float(n[2])]
        rec["tilt_deg"] = tilt
        rec["rms_mm"] = rms
        rec["planarity"] = planarity
        if w[0] > planar_ratio * w[1]:       # 비평면(사행 변곡 등) → 제외
            rec["included"] = False
            rec["reason"] = "non_planar"
        else:
            rec["included"] = True
            rec["reason"] = "ok"
            tilts_included.append(tilt)
            h_included.append(float((ctr - origin) @ axis))
        cells.append(rec)

    median_tilt = float(np.median(tilts_included)) if len(tilts_included) >= 3 else None
    iqr = (float(np.percentile(tilts_included, 75) - np.percentile(tilts_included, 25))
           if len(tilts_included) >= 3 else None)

    # 상/하부 경계 탐지 (compute_wall_tilt 와 동일 로직)
    sections = None
    if len(tilts_included) >= 30:
        sections = split_upper_lower(
            np.asarray(h_included), np.asarray(tilts_included)
        )

    # 샘플 점 — 벽면(셀 인덱스 태그) + 비벽면(컨텍스트)
    rng = np.random.default_rng(0)
    nw = len(P_wall)
    wsel = rng.choice(nw, max_wall_sample, replace=False) if nw > max_wall_sample else np.arange(nw)
    wall_xyz = np.round(P_wall[wsel], 3).astype(np.float32)
    wall_cell = inv[wsel].astype(np.int32)   # inv[i] == 점 i 가 속한 cells 리스트 인덱스

    P_other = P_r[~mask_wall]
    no = len(P_other)
    osel = rng.choice(no, max_other_sample, replace=False) if no > max_other_sample else np.arange(no)
    other_xyz = np.round(P_other[osel], 3).astype(np.float32)

    bmin = P_r.min(axis=0)
    bmax = P_r.max(axis=0)

    return {
        "dataset": dataset,
        "reference": ref_stem,
        "target": target_stem,
        "plumb_axis": [float(v) for v in axis],
        "plumb_origin": [float(v) for v in origin],
        "facing_dir": [float(v) for v in facing_dir],
        "cell_size_m": float(cell),
        "min_pts": int(min_pts),
        "planar_ratio": float(planar_ratio),
        "mask_nz": float(mask_nz),
        "median_tilt_deg": median_tilt,
        "iqr_deg": iqr,
        "sections": sections,
        "n_cells": len(cells),
        "n_cells_included": int(len(tilts_included)),
        "bbox_min": [float(v) for v in bmin],
        "bbox_max": [float(v) for v in bmax],
        "cells": cells,
        "wall_xyz": wall_xyz.reshape(-1).tolist(),
        "wall_cell": wall_cell.tolist(),
        "other_xyz": other_xyz.reshape(-1).tolist(),
        "wall_point_total": int(nw),
        "other_point_total": int(no),
    }


def save_result(result: dict, out_path: Path) -> None:
    out_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
