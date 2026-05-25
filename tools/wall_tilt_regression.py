"""옹벽 기울기 회귀 CLI — 실제 계산은 server.wall_tilt 에 위임.

사용:
    python tools/wall_tilt_regression.py <dataset> <target_stem> [--save] [--no-plot]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 프로젝트 루트를 sys.path 에 추가
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from server.wall_tilt import compute_wall_tilt, save_result  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("target_stem")
    ap.add_argument("--data-root", default=str(ROOT / "data"))
    ap.add_argument("--plumb", default=None)
    ap.add_argument("--mask-nz", type=float, default=0.3)
    ap.add_argument("--mag-min", type=float, default=0.001)
    ap.add_argument("--mag-max", type=float, default=0.50)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    root = Path(args.data_root)
    try:
        result = compute_wall_tilt(
            dataset=args.dataset,
            target_stem=args.target_stem,
            data_root=root,
            mask_nz=args.mask_nz,
            mag_min_m=args.mag_min,
            mag_max_m=args.mag_max,
            n_bootstrap=args.n_boot,
            plumb_path=Path(args.plumb) if args.plumb else None,
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    a = result["alpha_mm_per_m"]
    lo, hi = result["alpha_ci95_mm_per_m"]
    print(f"기준 → 대상              : {result['reference']} → {result['target']}")
    print(f"옹벽 면 점 개수           : {result['wall_face_point_count']:,}"
          f"  / 전체 {result['total_point_count']:,}")
    print(f"높이 범위 h              : {result['height_range_m'][0]:.3f} "
          f"~ {result['height_range_m'][1]:.3f} m")
    print(f"  α (회귀 기울기)         : {a:+.4f} mm/m")
    print(f"  α 95% CI               : [{lo:+.4f}, {hi:+.4f}] mm/m")
    print(f"  β                       : {result['beta_mm']:+.3f} mm")
    print(f"  잔차 robust σ           : {result['residual_robust_sigma_mm']:.3f} mm")
    print(f"  R²                      : {result['r_squared']:.4f}")
    print(f">>> 추가 기울기 변화      : {result['tilt_change_deg']:+.5f}°  "
          f"({result['tilt_direction']})")
    print(f"    95% CI               : [{result['tilt_change_ci95_deg'][0]:+.5f}, "
          f"{result['tilt_change_ci95_deg'][1]:+.5f}]°")
    print(f"    판정                  : {result['verdict']}")

    if args.save:
        out_path = root / args.dataset / f"wall_tilt_{args.target_stem}.json"
        save_result(result, out_path)
        print(f"\n[saved] {out_path}")

    if not args.no_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np
            h = np.asarray(result["scatter_h_m"])
            d = np.asarray(result["scatter_d_mm"])
            h_ref = result["h_ref_m"]
            alpha = result["alpha_mm_per_m"]
            beta = result["beta_mm"]
            h_lo, h_hi = result["height_range_m"]
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(h, d, s=0.8, alpha=0.25, c="C0")
            hh = np.linspace(h_lo, h_hi, 100)
            ax.plot(hh, alpha * (hh - h_ref) + beta, "r-", lw=2,
                    label=f"α={alpha:+.3f} mm/m  ({result['tilt_change_deg']:+.4f}°)")
            ax.axhline(0, color="k", lw=0.5)
            ax.set_xlabel("height along plumb h (m)")
            ax.set_ylabel("signed normal disp (mm) — outward +")
            ax.set_title(f"{args.dataset}  {result['reference']} → {args.target_stem}"
                         f"  wall tilt regression")
            ax.legend()
            ax.grid(alpha=0.3)
            plot_path = root / args.dataset / f"wall_tilt_{args.target_stem}.png"
            fig.savefig(str(plot_path), dpi=110, bbox_inches="tight")
            print(f"[saved] {plot_path}")
        except ImportError:
            print("[skip] matplotlib not installed, no plot")


if __name__ == "__main__":
    main()
