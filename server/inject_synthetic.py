"""DEV-ONLY: Inject a synthetic displacement field into 20260511_simple.ply
so we can validate the viewer/colormap end-to-end when the two source .pts
files happen to be identical.

It does NOT alter the original .pts. It overwrites *_simple.ply with a
perturbed copy and rewrites *_disp.bin + the displacement stats in
*_meta.json so the frontend sees a realistic field.

Run:  python -m server.inject_synthetic
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import open3d as o3d

from . import pipeline

DATASET = "CH2_RETAINWALL"
TARGET_STEM = "20260511"


def main():
    folder = pipeline.DATA_ROOT / DATASET
    ref_ply = folder / "20260508_simple.ply"
    tgt_ply = folder / f"{TARGET_STEM}_simple.ply"
    meta_path = folder / f"{TARGET_STEM}_meta.json"
    disp_path = folder / f"{TARGET_STEM}_disp.bin"

    ref = pipeline.read_simple_ply(ref_ply)
    tgt = pipeline.read_simple_ply(tgt_ply)

    pts = np.asarray(tgt.points)
    bb_min = pts.min(0); bb_max = pts.max(0)
    extent = bb_max - bb_min
    center = (bb_max + bb_min) / 2

    # Need a normal direction for the wall; estimate from a global PCA.
    cov = np.cov(pts.T)
    w, v = np.linalg.eigh(cov)
    wall_normal = v[:, 0]                # smallest variance dir = surface normal
    if wall_normal[1] < 0:               # orient roughly +Y (outward)
        wall_normal = -wall_normal

    # Synthetic bulge: gaussian blob centered on wall, pushed along normal.
    # Magnitude up to ~50 mm at the peak, smaller broad settlement elsewhere.
    blob_center = center + wall_normal * 0.0
    sigma_xy = max(extent[0], extent[2]) * 0.18
    sigma_z = extent[2] * 0.25
    dx = pts - blob_center
    rad2 = (dx[:, 0]**2) / (sigma_xy**2) + (dx[:, 2]**2) / (sigma_z**2)
    bulge_mag = 0.050 * np.exp(-rad2)              # +50 mm outward at peak

    # Add a mild settling at the top (negative Z translation), and a tilt.
    z_rel = (pts[:, 2] - bb_min[2]) / max(extent[2], 1e-6)
    settle = -0.008 * z_rel                        # up to -8 mm at top in Z
    tilt_x = (pts[:, 0] - center[0]) / max(extent[0], 1e-6) * 0.010   # +/-10 mm in X tied to X position

    # Compose displacement vector field.
    disp_vec = (bulge_mag[:, None] * wall_normal[None, :]
                + np.column_stack([tilt_x, np.zeros_like(tilt_x), settle]))

    new_pts = pts + disp_vec
    tgt_perturbed = o3d.geometry.PointCloud()
    tgt_perturbed.points = o3d.utility.Vector3dVector(new_pts)
    pipeline.write_simple_ply(tgt_perturbed, tgt_ply)
    print(f"wrote perturbed {tgt_ply.name}  ({len(new_pts):,} pts)")
    print(f"wall_normal = {wall_normal}")

    disp = pipeline.compute_displacement(tgt_perturbed, ref)
    disp.tofile(disp_path)
    print(f"wrote {disp_path.name}  ({disp.shape[0]:,} x 4)")
    print(f"  signed range = [{disp[:,0].min():.4f}, {disp[:,0].max():.4f}]")
    print(f"  mag    range = [{disp[:,1].min():.4f}, {disp[:,1].max():.4f}]")
    print(f"  horiz  range = [{disp[:,2].min():.4f}, {disp[:,2].max():.4f}]")
    print(f"  vert   range = [{disp[:,3].min():.4f}, {disp[:,3].max():.4f}]")

    meta = json.loads(meta_path.read_text())
    meta["synthetic_perturbation"] = True
    meta["displacement_stats"] = {
        "signed_normal": pipeline._stats(disp[:, 0]),
        "magnitude":     pipeline._stats(disp[:, 1]),
        "horizontal":    pipeline._stats(disp[:, 2]),
        "vertical":      pipeline._stats(disp[:, 3]),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print("updated meta.")


if __name__ == "__main__":
    main()
