"""Point cloud preprocessing pipeline for slope displacement monitoring.

Steps performed for a non-reference snapshot:
  1. Load .pts (ASCII: first line=count, then 'x y z intensity r g b').
  2. Voxel-downsample to ~TARGET_POINTS, save as binary PLY (*_simple.ply).
  3. FGR global registration between current_simple and reference_simple -> T0.
  4. Multi-scale point-to-plane ICP on the originals at voxels (1, 0.4, 0.1, 0.05, 0.01)
     with init=T0 -> final transform T.
  5. Apply T to the simple cloud, compute per-point displacement vs reference,
     save *_simple.ply (transformed XYZ) and *_disp.bin (n x 4 float32:
     [signed_normal, magnitude, horizontal_xy, vertical_z]).
  6. Save *_meta.json (transform, fitness history, counts, timing).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import open3d as o3d
import pandas as pd
from scipy.spatial import cKDTree

DATA_ROOT = Path(os.environ.get("SLOPE_DATA_ROOT", Path(__file__).resolve().parent.parent / "data"))
TARGET_POINTS = 500_000
ICP_VOXELS = (1.0, 0.4, 0.1, 0.05, 0.01)


# ---------- I/O ----------

def list_scan_files(dataset: str) -> list[dict]:
    """List raw scan files (.ply, excluding *_simple.ply) in a dataset folder."""
    folder = DATA_ROOT / dataset
    if not folder.is_dir():
        return []
    rows = []
    for p in sorted(folder.glob("*.ply")):
        if p.stem.endswith("_simple"):
            continue  # skip downsampled files
        stem = p.stem
        simple = folder / f"{stem}_simple.ply"
        meta = folder / f"{stem}_meta.json"
        disp = folder / f"{stem}_disp.bin"
        rgb = folder / f"{stem}_rgb.bin"
        rows.append({
            "name": p.name,
            "stem": stem,
            "size_mb": round(p.stat().st_size / 1024 / 1024, 1),
            "has_simple": simple.exists(),
            "has_disp": disp.exists(),
            "has_meta": meta.exists(),
            "has_rgb": rgb.exists(),
        })
    rows.sort(key=lambda r: r["stem"])
    return rows


def load_scan(path: Path, with_rgb: bool = False) -> o3d.geometry.PointCloud:
    """Load a raw scan PLY file via Open3D."""
    pcd = o3d.io.read_point_cloud(str(path))
    if not with_rgb and pcd.has_colors():
        pcd.colors = o3d.utility.Vector3dVector()  # discard to save memory
    return pcd


def load_scan_xyz_rgb_arrays(path: Path):
    """Load PLY and return (xyz float64, rgb uint8) arrays."""
    pcd = o3d.io.read_point_cloud(str(path))
    xyz = np.asarray(pcd.points, dtype=np.float64)
    if pcd.has_colors():
        rgb = (np.asarray(pcd.colors) * 255.0).clip(0, 255).astype(np.uint8)
    else:
        rgb = np.zeros((len(xyz), 3), dtype=np.uint8)
    return xyz, rgb


def write_simple_ply(pcd: o3d.geometry.PointCloud, path: Path) -> None:
    o3d.io.write_point_cloud(str(path), pcd, write_ascii=False, compressed=False)


def read_simple_ply(path: Path) -> o3d.geometry.PointCloud:
    return o3d.io.read_point_cloud(str(path))


# ---------- Downsampling ----------

def voxel_downsample_to_target(pcd: o3d.geometry.PointCloud, target: int = TARGET_POINTS):
    """Voxel-downsample, iteratively converging on target point count."""
    bb = pcd.get_axis_aligned_bounding_box()
    extent = bb.get_extent()
    volume = max(float(extent[0] * extent[1] * extent[2]), 1e-6)
    voxel = (volume / target) ** (1.0 / 3.0)
    ds = pcd
    for _ in range(8):
        ds = pcd.voxel_down_sample(voxel_size=float(voxel))
        c = len(ds.points)
        if 0.8 * target <= c <= 1.2 * target:
            return ds, float(voxel)
        if c == 0:
            voxel *= 0.5
            continue
        voxel *= (c / target) ** (1.0 / 3.0)
    return ds, float(voxel)


# ---------- Registration ----------

COLOCATED_THRESHOLD = 5.0     # clouds with centers < this (m) are considered same-frame
MAX_FGR_ROTATION_DEG = 30.0   # for co-located scans, FGR rotation > this falls back to identity
MIN_ICP_FITNESS = 0.3         # warn if final-scale fitness is below this


def _rotation_angle_deg(T: np.ndarray) -> float:
    """Extract rotation angle in degrees from a 4x4 transform matrix."""
    R = T[:3, :3]
    trace = np.clip(np.trace(R), -1.0, 3.0)
    angle_rad = np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(angle_rad))


def _features(pcd: o3d.geometry.PointCloud, voxel: float):
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2.0, max_nn=30)
    )
    return o3d.pipelines.registration.compute_fpfh_feature(
        pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 5.0, max_nn=100)
    )


def fgr_global(src_simple: o3d.geometry.PointCloud,
               tgt_simple: o3d.geometry.PointCloud,
               voxel: float = 0.2) -> np.ndarray:
    s = src_simple.voxel_down_sample(voxel)
    t = tgt_simple.voxel_down_sample(voxel)
    fs = _features(s, voxel)
    ft = _features(t, voxel)
    res = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        s, t, fs, ft,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=voxel * 1.5
        ),
    )
    return np.asarray(res.transformation)


def multiscale_icp(src: o3d.geometry.PointCloud,
                   tgt: o3d.geometry.PointCloud,
                   init_T: np.ndarray,
                   voxels=ICP_VOXELS,
                   progress=None):
    T = np.asarray(init_T, dtype=np.float64).copy()
    history = []
    n_scales = len(voxels)
    for i, v in enumerate(voxels):
        s = src.voxel_down_sample(v)
        t = tgt.voxel_down_sample(v)
        s.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=v * 2.0, max_nn=30))
        t.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=v * 2.0, max_nn=30))
        # Coarse scales get wider correspondence search and more iterations
        # to allow ICP to refine a rough FGR alignment.
        if i < n_scales // 2:
            cd = v * 4.0
            max_iter = 200
        else:
            cd = v * 2.0
            max_iter = 100
        if progress:
            progress("icp", f"scale {i+1}/{n_scales}, voxel={v}m, cd={cd:.2f}m, pts={len(s.points):,}")
        res = o3d.pipelines.registration.registration_icp(
            s, t,
            max_correspondence_distance=cd,
            init=T,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=max_iter, relative_fitness=1e-7, relative_rmse=1e-7
            ),
        )
        T = np.asarray(res.transformation)
        entry = {
            "voxel": float(v),
            "max_corr_dist": float(cd),
            "fitness": float(res.fitness),
            "inlier_rmse": float(res.inlier_rmse),
            "src_count": int(len(s.points)),
            "tgt_count": int(len(t.points)),
        }
        history.append(entry)
        if progress:
            progress("icp_result", f"scale {i+1}/{n_scales}: fitness={res.fitness:.4f}, RMSE={res.inlier_rmse:.6f}m")
    return T, history


# ---------- Displacement ----------

def compute_displacement(current_simple: o3d.geometry.PointCloud,
                         ref_simple: o3d.geometry.PointCloud) -> np.ndarray:
    """Return n x 4 float32 array per current point:
    [signed_normal, magnitude, horizontal_xy_signed, vertical_z_signed].

    `current_simple` must already be in the reference frame.
    Signed normal: displacement projected onto the reference surface normal
    at the nearest reference point. Positive = outward (away from wall body).
    Horizontal: dot of XY displacement with the XY component of the reference
    normal (positive = displaced in the outward XY direction). Falls back to
    raw XY magnitude when the normal has no horizontal component.
    Vertical: raw dz.
    """
    if not ref_simple.has_normals():
        ref_simple.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30)
        )
        ref_simple.orient_normals_consistent_tangent_plane(k=20)

    cur = np.asarray(current_simple.points, dtype=np.float64)
    ref = np.asarray(ref_simple.points, dtype=np.float64)
    n = np.asarray(ref_simple.normals, dtype=np.float64)

    tree = cKDTree(ref)
    dist, idx = tree.query(cur, k=1, workers=-1)

    delta = cur - ref[idx]                                 # (N,3) displacement vectors
    normals = n[idx]                                       # (N,3)

    signed_normal = np.einsum("ij,ij->i", delta, normals)  # signed along normal
    magnitude = dist.astype(np.float32)                    # unsigned NN distance

    # Horizontal: project XY displacement onto outward XY direction of normal
    nxy = normals[:, :2]
    nxy_norm = np.linalg.norm(nxy, axis=1)
    has_h = nxy_norm > 1e-6
    horiz = np.zeros(len(cur), dtype=np.float64)
    if has_h.any():
        nxy_dir = np.zeros_like(nxy)
        nxy_dir[has_h] = nxy[has_h] / nxy_norm[has_h, None]
        horiz[has_h] = np.einsum("ij,ij->i", delta[has_h, :2], nxy_dir[has_h])
    # For points where the normal is nearly vertical (floor-like), use raw XY mag
    horiz[~has_h] = np.linalg.norm(delta[~has_h, :2], axis=1)

    # Vertical: Z component of the normal-projected displacement.
    # signed_normal * normal_z removes along-surface sliding artifacts.
    vert = signed_normal * normals[:, 2]

    out = np.column_stack([signed_normal, magnitude, horiz, vert]).astype(np.float32)
    return out


# ---------- Top-level orchestration ----------

def preprocess(dataset: str, target_file: str, *, progress=None) -> dict:
    """Run the full preprocess pipeline for `target_file` against the oldest
    scan .ply in the dataset folder.

    `progress(stage:str, detail:str)` is called periodically for UI updates.
    """
    def _p(stage, detail=""):
        if progress is not None:
            progress(stage, detail)

    folder = DATA_ROOT / dataset
    target_path = folder / target_file
    if not target_path.is_file():
        raise FileNotFoundError(target_path)

    listing = list_scan_files(dataset)
    if not listing:
        raise RuntimeError(f"No scan .ply files in {folder}")
    ref_entry = listing[0]
    ref_path = folder / ref_entry["name"]
    is_reference = (target_path.name == ref_path.name)

    timing: dict[str, float] = {}
    t0 = time.time()

    # --- Reference simple: ensure it exists ---
    ref_stem = ref_path.stem
    ref_simple_path = folder / f"{ref_stem}_simple.ply"
    if not ref_simple_path.exists():
        _p("load_ref", str(ref_path.name))
        s = time.time()
        ref_pcd_full = load_scan(ref_path, with_rgb=True)
        timing["load_ref"] = time.time() - s

        _p("downsample_ref", f"{len(ref_pcd_full.points):,} pts")
        s = time.time()
        ref_simple, ref_voxel = voxel_downsample_to_target(ref_pcd_full, TARGET_POINTS)
        timing["downsample_ref"] = time.time() - s

        write_simple_ply(ref_simple, ref_simple_path)
        if ref_simple.has_colors():
            rgb_u8 = (np.asarray(ref_simple.colors) * 255.0).clip(0, 255).astype(np.uint8)
            (folder / f"{ref_stem}_rgb.bin").write_bytes(rgb_u8.tobytes())
        # Reference has identity transform and zero displacement.
        ref_meta = folder / f"{ref_stem}_meta.json"
        ref_meta.write_text(json.dumps({
            "is_reference": True,
            "source": ref_path.name,
            "simple_voxel": ref_voxel,
            "simple_count": len(ref_simple.points),
            "transform": np.eye(4).tolist(),
            "icp_history": [],
            "global_voxel": None,
            "timing": {"downsample": timing.get("downsample_ref", 0.0)},
        }, indent=2))
    else:
        ref_pcd_full = None  # loaded lazily if needed

    if is_reference:
        # Nothing further to do; reference itself has no displacement.
        return {
            "ok": True,
            "is_reference": True,
            "reference": ref_path.name,
            "target": target_path.name,
            "timing": timing,
        }

    # --- Target simple ---
    tgt_stem = target_path.stem
    tgt_simple_path = folder / f"{tgt_stem}_simple.ply"
    _p("load_target", str(target_path.name))
    s = time.time()
    tgt_pcd_full = load_scan(target_path, with_rgb=True)
    timing["load_target"] = time.time() - s

    _p("downsample_target", f"{len(tgt_pcd_full.points):,} pts")
    s = time.time()
    tgt_simple, tgt_voxel = voxel_downsample_to_target(tgt_pcd_full, TARGET_POINTS)
    timing["downsample_target"] = time.time() - s

    # Save the *untransformed* simple first (will be overwritten with transformed at end).
    write_simple_ply(tgt_simple, tgt_simple_path)
    if tgt_simple.has_colors():
        rgb_u8 = (np.asarray(tgt_simple.colors) * 255.0).clip(0, 255).astype(np.uint8)
        (folder / f"{tgt_stem}_rgb.bin").write_bytes(rgb_u8.tobytes())

    # --- Ensure reference originals/simples loaded ---
    ref_simple = read_simple_ply(ref_simple_path)
    if ref_pcd_full is None:
        _p("load_ref_full", str(ref_path.name))
        s = time.time()
        ref_pcd_full = load_scan(ref_path)
        timing["load_ref_full"] = time.time() - s

    # --- Global registration (FGR) on the *_simple clouds ---
    _p("fgr", f"voxel=0.2 on simple clouds")
    s = time.time()
    T0 = fgr_global(tgt_simple, ref_simple, voxel=0.2)
    timing["fgr"] = time.time() - s

    fgr_angle = _rotation_angle_deg(T0)
    ref_center = np.asarray(ref_simple.points).mean(axis=0)
    tgt_center = np.asarray(tgt_simple.points).mean(axis=0)
    center_dist = float(np.linalg.norm(tgt_center - ref_center))
    colocated = center_dist < COLOCATED_THRESHOLD

    # For co-located scans (same scanner frame), large rotations indicate FGR failure.
    # For distant scans (different frames), large rotations are expected and valid.
    if colocated and fgr_angle > MAX_FGR_ROTATION_DEG:
        fgr_accepted = False
        _p("fgr_rejected",
           f"Co-located scans (center dist {center_dist:.1f}m) but FGR rotation "
           f"{fgr_angle:.1f}° > {MAX_FGR_ROTATION_DEG}° -- falling back to identity init")
        T0_used = np.eye(4, dtype=np.float64)
    else:
        fgr_accepted = True
        _p("fgr_ok", f"FGR rotation {fgr_angle:.1f}°, center dist {center_dist:.1f}m")
        T0_used = T0

    # --- Multi-scale ICP on the originals, init=T0 ---
    _p("icp", f"voxels={ICP_VOXELS}")
    s = time.time()
    T_final, history = multiscale_icp(tgt_pcd_full, ref_pcd_full, T0_used, voxels=ICP_VOXELS, progress=_p)
    timing["icp"] = time.time() - s

    # --- If ICP fitness is poor and we used FGR, retry with identity init ---
    final_fitness = history[-1]["fitness"] if history else 0.0
    if final_fitness < MIN_ICP_FITNESS and fgr_accepted:
        _p("icp_retry",
           f"ICP fitness {final_fitness:.4f} < {MIN_ICP_FITNESS}, "
           f"retrying with identity init")
        s = time.time()
        T_retry, history_retry = multiscale_icp(
            tgt_pcd_full, ref_pcd_full, np.eye(4, dtype=np.float64),
            voxels=ICP_VOXELS, progress=_p)
        timing["icp_retry"] = time.time() - s
        retry_fitness = history_retry[-1]["fitness"] if history_retry else 0.0
        if retry_fitness > final_fitness:
            _p("icp_retry_accepted",
               f"Identity-init fitness {retry_fitness:.4f} > FGR-init {final_fitness:.4f}, using retry")
            T_final = T_retry
            history = history_retry
            T0_used = np.eye(4, dtype=np.float64)
            final_fitness = retry_fitness
        else:
            _p("icp_retry_rejected",
               f"Identity-init fitness {retry_fitness:.4f} <= FGR-init {final_fitness:.4f}, keeping original")

    # --- Warn if final fitness is still low ---
    if final_fitness < MIN_ICP_FITNESS:
        _p("warning",
           f"Final ICP fitness {final_fitness:.4f} is below {MIN_ICP_FITNESS} -"
           f"displacement results may be unreliable")

    # --- Apply T_final to both simple and full target ---
    tgt_simple_t = o3d.geometry.PointCloud(tgt_simple)
    tgt_simple_t.transform(T_final)
    write_simple_ply(tgt_simple_t, tgt_simple_path)

    _p("transform_full", f"applying transform to full target ({len(tgt_pcd_full.points):,} pts)")
    s = time.time()
    tgt_full_t = o3d.geometry.PointCloud(tgt_pcd_full)
    tgt_full_t.transform(T_final)
    timing["transform_full"] = time.time() - s

    # --- Displacement on full clouds, then map to simple for display ---
    n_full_tgt = len(tgt_full_t.points)
    n_full_ref = len(ref_pcd_full.points)
    _p("displacement", f"full vs full ({n_full_tgt:,} vs {n_full_ref:,} pts)")
    s = time.time()
    disp_full = compute_displacement(tgt_full_t, ref_pcd_full)
    timing["displacement_full"] = time.time() - s

    # Map full displacement to simple cloud points via NN
    _p("displacement_map", f"mapping to simple ({len(tgt_simple_t.points):,} pts)")
    s = time.time()
    full_t_xyz = np.asarray(tgt_full_t.points, dtype=np.float64)
    simple_t_xyz = np.asarray(tgt_simple_t.points, dtype=np.float64)
    tree_full_t = cKDTree(full_t_xyz)
    _, nn_idx = tree_full_t.query(simple_t_xyz, k=1, workers=-1)
    disp = disp_full[nn_idx]
    timing["displacement_map"] = time.time() - s

    disp_path = folder / f"{tgt_stem}_disp.bin"
    disp.tofile(disp_path)

    meta_path = folder / f"{tgt_stem}_meta.json"
    meta = {
        "is_reference": False,
        "reference": ref_path.name,
        "source": target_path.name,
        "simple_voxel": tgt_voxel,
        "simple_count": int(len(tgt_simple_t.points)),
        "global_voxel": 0.2,
        "icp_voxels": list(ICP_VOXELS),
        "icp_history": history,
        "transform_global": T0.tolist(),
        "transform_global_used": T0_used.tolist(),
        "fgr_rotation_deg": fgr_angle,
        "fgr_accepted": fgr_accepted,
        "center_distance": center_dist,
        "final_icp_fitness": final_fitness,
        "registration_reliable": final_fitness >= MIN_ICP_FITNESS,
        "transform": T_final.tolist(),
        "displacement_stats": {
            "signed_normal": _stats(disp[:, 0]),
            "magnitude": _stats(disp[:, 1]),
            "horizontal": _stats(disp[:, 2]),
            "vertical": _stats(disp[:, 3]),
        },
        "timing": timing,
        "total_seconds": time.time() - t0,
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    _p("done", f"{meta['total_seconds']:.1f}s")
    return {"ok": True, "is_reference": False, **meta}


def extract_rgb_for_simple(dataset: str, stem: str) -> Path:
    """Back-fill per-point RGB for an existing *_simple.ply by nearest-neighbor
    lookup in the original scan .ply. Writes *_rgb.bin (N x 3 uint8) and returns
    the path. No-op if the file already exists.

    Handles the case where the simple cloud has been transformed by ICP: we
    inverse-transform it back to source frame before the NN query.
    """
    folder = DATA_ROOT / dataset
    scan_path = folder / f"{stem}.ply"
    simple_path = folder / f"{stem}_simple.ply"
    rgb_path = folder / f"{stem}_rgb.bin"
    meta_path = folder / f"{stem}_meta.json"

    if rgb_path.exists():
        return rgb_path
    if not scan_path.is_file():
        raise FileNotFoundError(scan_path)
    if not simple_path.is_file():
        raise FileNotFoundError(simple_path)

    simple = read_simple_ply(simple_path)
    sxyz = np.asarray(simple.points, dtype=np.float64)

    # Inverse-transform simple back to source frame if a non-identity transform exists.
    T = np.eye(4)
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            T = np.asarray(meta.get("transform", np.eye(4).tolist()), dtype=np.float64)
        except Exception:
            pass
    if not np.allclose(T, np.eye(4)):
        Tinv = np.linalg.inv(T)
        ones = np.ones((len(sxyz), 1))
        sxyz_h = np.hstack([sxyz, ones]) @ Tinv.T
        sxyz = sxyz_h[:, :3]

    xyz, rgb = load_scan_xyz_rgb_arrays(scan_path)
    tree = cKDTree(xyz)
    _, idx = tree.query(sxyz, k=1, workers=-1)
    out = rgb[idx]  # (N, 3) uint8
    out.tofile(rgb_path)
    return rgb_path


def _stats(arr: np.ndarray) -> dict:
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "abs_mean": float(np.mean(np.abs(arr))),
        "rms": float(np.sqrt(np.mean(arr.astype(np.float64) ** 2))),
        "p95_abs": float(np.percentile(np.abs(arr), 95)),
    }
