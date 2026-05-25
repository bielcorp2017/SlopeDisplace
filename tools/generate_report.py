"""옹벽 모니터링 분석 리포트(PDF) 생성 스크립트.

사용:
    python tools/generate_report.py <dataset> [--site "현장명"] [--out report.pdf]

생성 내용:
    1. 표지            — 현장명, 데이터셋, 측정 요약
    2. 수직봉 측정      — plumb.json 결과·검증
    3. 시계열 요약 표   — 날짜별 α, tilt, fitness, σ
    4. 시계열 그래프    — α / tilt 막대 + 산점도
    5. 각 비교쌍 상세    — 회귀 차트 + 메트릭 (날짜별)
    6. 종합 결론
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import font_manager

# --- Korean font ---
_FONT_PATH = r"C:\Windows\Fonts\malgun.ttf"
if Path(_FONT_PATH).exists():
    font_manager.fontManager.addfont(_FONT_PATH)
    plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent.parent


# ---------- helpers ----------
def fmt_date(stem: str) -> str:
    if len(stem) == 8 and stem.isdigit():
        return f"{stem[:4]}-{stem[4:6]}-{stem[6:8]}"
    return stem


def fmt_signed(v, d=4):
    if v is None:
        return "—"
    return f"{v:+.{d}f}"


def text_page(pdf: PdfPages, title: str, lines: list[str], subtitle: str = ""):
    fig = plt.figure(figsize=(8.27, 11.69))  # A4
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.text(0.5, 0.92, title, ha="center", va="top", fontsize=22, fontweight="bold")
    if subtitle:
        ax.text(0.5, 0.86, subtitle, ha="center", va="top", fontsize=12, color="#555")
    y = 0.78
    for line in lines:
        ax.text(0.1, y, line, ha="left", va="top", fontsize=11)
        y -= 0.035
    pdf.savefig(fig)
    plt.close(fig)


def section_header(ax, text):
    ax.text(0.05, 0.96, text, transform=ax.transAxes,
            ha="left", va="top", fontsize=16, fontweight="bold")
    # 가로선: Line2D 를 직접 추가 (axhline 은 transform 인수 미허용)
    from matplotlib.lines import Line2D
    line = Line2D([0.05, 0.95], [0.93, 0.93],
                  transform=ax.transAxes, color="#888", lw=0.8)
    ax.add_line(line)


def kv_table(ax, items, x=0.08, y0=0.88, lh=0.045, col_split=0.42):
    """items: list of (key, value, [color]) tuples — value rendered to the right."""
    y = y0
    for it in items:
        if len(it) >= 3:
            k, v, c = it[0], it[1], it[2]
        else:
            k, v, c = it[0], it[1], "#222"
        ax.text(x, y, k, transform=ax.transAxes,
                ha="left", va="top", fontsize=11, color="#666")
        ax.text(x + col_split, y, v, transform=ax.transAxes,
                ha="left", va="top", fontsize=11, color=c,
                family="Malgun Gothic")
        y -= lh


# ---------- pages ----------
def page_cover(pdf, site, dataset, scans, summary):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")

    ax.text(0.5, 0.85, "옹벽 변위 모니터링 분석 리포트",
            ha="center", va="top", fontsize=24, fontweight="bold")
    ax.text(0.5, 0.79, site, ha="center", va="top",
            fontsize=18, color="#1a4d80")
    ax.text(0.5, 0.74, f"데이터셋  ·  {dataset}", ha="center", va="top",
            fontsize=12, color="#666")

    # divider
    ax.axhline(y=0.70, xmin=0.20, xmax=0.80, color="#888", lw=0.8)

    # summary box
    box_y = 0.62
    ax.text(0.1, box_y, "요약", ha="left", va="top",
            fontsize=14, fontweight="bold")
    items = [
        ("기준일", summary["reference_date"]),
        ("최신일", summary["latest_date"]),
        ("관측기간", f"{summary['days_span']} 일"),
        ("스캔 개수", f"{len(scans)} 개"),
        ("측정 수직 벡터", f"({summary['plumb_axis'][0]:+.4f}, "
                       f"{summary['plumb_axis'][1]:+.4f}, "
                       f"{summary['plumb_axis'][2]:+.4f})"),
        ("Z축 대비 기울기", f"{summary['plumb_tilt']:.4f}°"),
        ("", ""),
        ("최신 옹벽 기울기 변화", f"{summary['latest_tilt']:+.5f}° "
                              f"({summary['latest_direction']})"),
        ("95% CI", f"[{summary['latest_ci'][0]:+.5f}, "
                   f"{summary['latest_ci'][1]:+.5f}]°"),
        ("판정", summary["latest_verdict"]),
    ]
    y = box_y - 0.05
    for k, v in items:
        if not k and not v:
            y -= 0.02; continue
        ax.text(0.1, y, k, ha="left", va="top", fontsize=11, color="#666")
        color = "#222"
        if "OUTWARD" in str(v): color = "#c00"
        elif "INWARD" in str(v): color = "#06a"
        elif "NO SIGNIFICANT" in str(v): color = "#080"
        ax.text(0.45, y, v, ha="left", va="top",
                fontsize=11, color=color, family="Malgun Gothic")
        y -= 0.035

    ax.text(0.5, 0.08,
            f"생성일자 {datetime.now():%Y-%m-%d %H:%M}   ·   "
            f"SlopeDisplace v1",
            ha="center", va="top", fontsize=9, color="#888")

    pdf.savefig(fig); plt.close(fig)


def page_plumb(pdf, plumb):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    section_header(ax, "1. 수직봉(plumb) 측정 결과")
    ax.text(0.05, 0.89,
            "현장에 설치한 수직봉을 별도로 스캔하고, PCA·슬라이스 원피팅·"
            "원기둥 LSQ 세 가지 독립 방법으로 축 벡터를 추출. 세 방법의 일치도가 "
            "곧 측정 신뢰의 증거.",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=10, color="#444", wrap=True)

    val = plumb.get("validation", {})
    items = [
        ("원본 PLY",                Path(plumb["source_ply"]).name),
        ("축 벡터",                 f"({plumb['axis'][0]:+.6f}, "
                                  f"{plumb['axis'][1]:+.6f}, "
                                  f"{plumb['axis'][2]:+.6f})"),
        ("중심",                    f"({plumb['center'][0]:.3f}, "
                                  f"{plumb['center'][1]:.3f}, "
                                  f"{plumb['center'][2]:.3f}) m"),
        ("반경",                    f"{plumb['radius_m']*1000:.2f} mm"),
        ("봉 길이(인라이어)",         f"{plumb['rod_length_m']:.3f} m"),
        ("Z축 대비 기울기",          f"{plumb['tilt_from_z_deg']:.4f}°"),
        ("인라이어 점 개수",          f"{plumb['n_inliers']:,}"),
        ("표면 잡음 σ",              f"{plumb['surface_sigma_m']*1000:.2f} mm"),
        ("", ""),
        ("[검증] Cyl ↔ PCA",         f"{val.get('cylinder_vs_pca_deg', 0):.4f}°"),
        ("[검증] Cyl ↔ Slice",       f"{val.get('cylinder_vs_slice_deg', 0):.4f}°"),
        ("[검증] PCA ↔ Slice",       f"{val.get('pca_vs_slice_deg', 0):.4f}°"),
        ("선형성 L",                 f"{val.get('linearity', 0):.4f}"),
        ("planarity P",             f"{val.get('planarity', 0):.4f}"),
        ("", ""),
        ("축 일치 PASS",             "YES" if val.get("axis_match_pass") else "NO"),
        ("막대 형태 PASS",            "YES" if val.get("shape_pass") else "NO"),
        ("원기둥 표면 PASS",          "YES" if val.get("surface_pass") else "NO"),
        ("종합 PROVEN",              "YES" if val.get("overall_proven") else "NO"),
    ]
    y = 0.82
    for k, v in items:
        if not k and not v:
            y -= 0.012; continue
        ax.text(0.08, y, k, transform=ax.transAxes,
                ha="left", va="top", fontsize=10, color="#666")
        color = "#222"
        if "YES" in str(v): color = "#080"
        if "NO" == str(v): color = "#c00"
        ax.text(0.48, y, v, transform=ax.transAxes,
                ha="left", va="top", fontsize=10, color=color, family="Malgun Gothic")
        y -= 0.028

    ax.text(0.05, 0.08,
            "→ 세 방법 모두 1° 이내 일치, 선형성·평면성 모두 통과, 표면 잡음이 "
            "스캐너 노이즈 수준이면 측정된 수직 벡터는 신뢰할 수 있음.",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=10, color="#444", style="italic")
    pdf.savefig(fig); plt.close(fig)


def page_timeseries(pdf, tilts):
    """막대그래프 + 표로 시계열 요약."""
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    title_ax = fig.add_axes([0, 0.92, 1, 0.06]); title_ax.axis("off")
    section_header(title_ax, "2. 시계열 비교")

    # 표
    table_ax = fig.add_axes([0.05, 0.62, 0.9, 0.27]); table_ax.axis("off")
    headers = ["기준 → 대상", "경과(일)", "α [mm/m]",
               "기울기 [°]", "방향", "σ [mm]", "ICP fit"]
    rows = []
    for t in tilts:
        a = t["alpha_mm_per_m"]
        deg = t["tilt_change_deg"]
        rows.append([
            f"{fmt_date(t['reference'])} → {fmt_date(t['target'])}",
            str(t["days_span"]),
            f"{a:+.4f}",
            f"{deg:+.5f}",
            t["tilt_direction"] if t["significant"] else "n/a",
            f"{t['residual_robust_sigma_mm']:.2f}",
            f"{t['icp_fitness']*100:.1f}%" if t.get("icp_fitness") else "—",
        ])
    table = table_ax.table(cellText=rows, colLabels=headers, loc="center",
                          cellLoc="center")
    table.auto_set_font_size(False); table.set_fontsize(9)
    table.scale(1.0, 1.5)
    for i, h in enumerate(headers):
        c = table[(0, i)]; c.set_facecolor("#2c3e50"); c.set_text_props(color="white", weight="bold")
    for r in range(1, len(rows) + 1):
        for c in range(len(headers)):
            cell = table[(r, c)]
            if c == 4:  # direction
                v = rows[r-1][4]
                if "OUTWARD" in v: cell.set_text_props(color="#c00")
                elif "INWARD" in v: cell.set_text_props(color="#06a")
                else: cell.set_text_props(color="#080")

    # bar charts
    bar_ax = fig.add_axes([0.10, 0.30, 0.80, 0.25])
    labels = [f"{fmt_date(t['target'])}\n(+{t['days_span']}일)" for t in tilts]
    vals_deg = [t["tilt_change_deg"] for t in tilts]
    ci_low = [t["tilt_change_ci95_deg"][0] for t in tilts]
    ci_hi = [t["tilt_change_ci95_deg"][1] for t in tilts]
    err_low = [vals_deg[i] - ci_low[i] for i in range(len(tilts))]
    err_hi = [ci_hi[i] - vals_deg[i] for i in range(len(tilts))]
    colors = ["#c0392b" if v > 0 else "#2980b9" for v in vals_deg]
    bar_ax.bar(labels, vals_deg, yerr=[err_low, err_hi], color=colors,
               capsize=8, alpha=0.85, edgecolor="#222")
    bar_ax.axhline(0, color="#333", lw=0.8)
    bar_ax.set_ylabel("추가 기울기 변화 (°)")
    bar_ax.set_title("기준일 대비 옹벽 기울기 변화 (95% CI bar)")
    bar_ax.grid(axis="y", alpha=0.3)

    # interpretation box
    ax2 = fig.add_axes([0, 0, 1, 0.28]); ax2.axis("off")
    interp_lines = []
    if len(tilts) >= 2:
        a0, a1 = tilts[0]["tilt_change_deg"], tilts[-1]["tilt_change_deg"]
        d0, d1 = tilts[0]["days_span"], tilts[-1]["days_span"]
        rate0 = a0 / max(d0, 1)
        rate1 = a1 / max(d1, 1)
        interp_lines.append(f"  · 첫 비교({d0}일): {a0:+.5f}° → 일평균 {rate0*1000:.3f} x 1e-3 °/일")
        interp_lines.append(f"  · 최근 비교({d1}일): {a1:+.5f}° → 일평균 {rate1*1000:.3f} x 1e-3 °/일")
        if abs(a1) > abs(a0):
            interp_lines.append(f"  · 누적 lean 이 {abs(a1)/max(abs(a0),1e-9):.1f}× 증가했고 방향 일관됨"
                                if (a0*a1 > 0) else "  · 방향이 바뀌었음 — 잡음 또는 비정상 변동 의심")
    ax2.text(0.05, 0.95, "[해석]", transform=ax2.transAxes,
             fontsize=12, fontweight="bold", va="top")
    for i, line in enumerate(interp_lines):
        ax2.text(0.05, 0.85 - i*0.10, line, transform=ax2.transAxes,
                 fontsize=10, va="top")

    pdf.savefig(fig); plt.close(fig)


def page_regression_detail(pdf, t, idx, total):
    """페이지: 한 비교쌍의 회귀 상세."""
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    title_ax = fig.add_axes([0, 0.92, 1, 0.06]); title_ax.axis("off")
    section_header(title_ax,
        f"3.{idx} 회귀 상세 — {fmt_date(t['reference'])} → {fmt_date(t['target'])} ({t['days_span']}일)")

    # scatter + regression line
    sc_ax = fig.add_axes([0.10, 0.42, 0.80, 0.40])
    h = np.asarray(t["scatter_h_m"]); d = np.asarray(t["scatter_d_mm"])
    sc_ax.scatter(h, d, s=1.2, alpha=0.25, c="#3a7bd5")
    h_lo, h_hi = t["height_range_m"]
    hh = np.linspace(h_lo, h_hi, 100)
    sc_ax.plot(hh, t["alpha_mm_per_m"] * (hh - t["h_ref_m"]) + t["beta_mm"],
               "r-", lw=2, label=f"α={t['alpha_mm_per_m']:+.3f} mm/m  "
                                  f"({t['tilt_change_deg']:+.4f}°)")
    sc_ax.axhline(0, color="#333", lw=0.4)
    sc_ax.set_xlabel("높이 h (m, plumb 축)")
    sc_ax.set_ylabel("signed normal disp (mm) — outward +")
    sc_ax.legend(loc="upper left", fontsize=9)
    sc_ax.grid(alpha=0.3)

    # metrics table
    mt_ax = fig.add_axes([0.10, 0.08, 0.80, 0.28]); mt_ax.axis("off")
    items = [
        ("α (회귀 기울기)",       f"{t['alpha_mm_per_m']:+.4f} mm/m"),
        ("α 95% CI",            f"[{t['alpha_ci95_mm_per_m'][0]:+.4f}, "
                                f"{t['alpha_ci95_mm_per_m'][1]:+.4f}]"),
        ("β (h_ref 변위)",        f"{t['beta_mm']:+.3f} mm"),
        ("h 범위",               f"{t['height_range_m'][0]:.2f} ~ {t['height_range_m'][1]:.2f} m"),
        ("h_ref",                f"{t['h_ref_m']:.3f} m"),
        ("잔차 robust σ",         f"{t['residual_robust_sigma_mm']:.2f} mm"),
        ("R²",                  f"{t['r_squared']:.4f}"),
        ("옹벽 점수",             f"{t['wall_face_point_count']:,} / "
                                f"{t['total_point_count']:,}"),
        ("ICP fitness",         f"{t['icp_fitness']*100:.1f}%"
                                if t.get("icp_fitness") else "—"),
        ("종합 판정",             t["verdict"]),
    ]
    y = 0.95
    for k, v in items:
        mt_ax.text(0.0, y, k, transform=mt_ax.transAxes,
                   ha="left", va="top", fontsize=10, color="#666")
        color = "#222"
        if "OUTWARD" in str(v): color = "#c00"
        elif "INWARD" in str(v): color = "#06a"
        elif "NO SIGNIFICANT" in str(v): color = "#080"
        mt_ax.text(0.40, y, v, transform=mt_ax.transAxes,
                   ha="left", va="top", fontsize=10, color=color,
                   family="Malgun Gothic")
        y -= 0.075

    pdf.savefig(fig); plt.close(fig)


def page_conclusion(pdf, summary, tilts):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    title_ax = fig.add_axes([0, 0.92, 1, 0.06]); title_ax.axis("off")
    section_header(title_ax, "4. 종합 결론 및 권고")

    ax = fig.add_axes([0, 0, 1, 0.92]); ax.axis("off")

    latest = tilts[-1] if tilts else None

    lines = []
    lines.append("【주요 발견】")
    lines.append("")
    if latest:
        s = "OUTWARD" if latest["significant"] and latest["alpha_mm_per_m"] > 0 \
            else ("INWARD" if latest["significant"] else "유의미한 변화 없음")
        lines.append(f"• 기준일({fmt_date(latest['reference'])})로부터 "
                     f"{fmt_date(latest['target'])} ({latest['days_span']}일) 사이,"
                     f" 옹벽이 {abs(latest['tilt_change_deg']):.4f}° 만큼 "
                     f"{s} 방향으로 변화함.")
        lines.append(f"• 통계적으로 유의 (95% CI 가 0 을 포함하지 "
                     f"{'안 함' if latest['significant'] else '함'}).")
        if latest.get("icp_fitness", 0) >= 0.5:
            lines.append(f"• ICP 정합 품질 {latest['icp_fitness']*100:.1f}%, "
                         f"잔차 σ={latest['residual_robust_sigma_mm']:.2f}mm — "
                         f"스캐너 잡음 수준으로 신뢰 가능.")
        else:
            lines.append(f"• ⚠ ICP 정합 품질 {latest['icp_fitness']*100:.1f}% 로 낮음 — "
                         f"결과 해석에 주의 필요.")
    lines.append("")
    lines.append("【측정 방법론 요약】")
    lines.append("")
    lines.append("1. 현장 수직봉을 스캔, PCA·슬라이스·원기둥LSQ 세 방법 교차검증으로 "
                 "정밀 수직 벡터(plumb axis) 추출.")
    lines.append("2. 각 날짜 스캔을 기준일과 ICP 정합. 모든 점에 대해 변위 벡터를 계산.")
    lines.append("3. 옹벽 표면 패치(|n·plumb|<0.3)만 마스킹.")
    lines.append("4. 그 점들의 signed_normal 변위를 plumb 축 높이에 대해 robust 회귀(Huber).")
    lines.append("5. 회귀 기울기 α [mm/m] = 단위 높이당 outward 변위 증가율, "
                 "atan(α/1000) ~= 추가 기울기 [rad].")
    lines.append("6. 부트스트랩 95% CI 로 통계 유의성 판단.")
    lines.append("")
    lines.append("【한계와 권고】")
    lines.append("")
    lines.append("• 본 분석은 두 시점 간 *추가* 기울기 변화를 측정함 — 기준일 자체의 "
                 "절대 수직도가 아님.")
    lines.append("• 옹벽 디자인의 굴곡은 두 스캔에 모두 존재하므로 차분에서 자동 상쇄됨.")
    lines.append("• ICP 정합 품질이 낮으면 (<30%) 변위 결과는 옹벽 변화가 아닌 정합 "
                 "오차를 측정할 수 있음.")
    lines.append("• 시계열 누적이 더 길어지면 lean 진행률(°/일)을 정확히 추정 가능.")
    lines.append("• 권고 모니터링 주기: 주 1회 ~ 월 1회, lean 변화 0.05°/주 초과 시 "
                 "현장 점검.")

    y = 0.95
    for line in lines:
        weight = "bold" if line.startswith("【") else "normal"
        size = 11.5 if line.startswith("【") else 10
        color = "#1a4d80" if line.startswith("【") else "#222"
        ax.text(0.06, y, line, transform=ax.transAxes,
                ha="left", va="top", fontsize=size, fontweight=weight,
                color=color, wrap=True)
        y -= 0.034
    pdf.savefig(fig); plt.close(fig)


# ---------- main ----------
def days_between(stem_a: str, stem_b: str) -> int:
    """8자리 YYYYMMDD stem 사이의 일수."""
    try:
        a = datetime.strptime(stem_a, "%Y%m%d")
        b = datetime.strptime(stem_b, "%Y%m%d")
        return (b - a).days
    except Exception:
        return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", help="예: CH2_RETAINWALL")
    ap.add_argument("--site", default="현장명 미지정",
                    help="현장명 (예: 대산당진 3공구 천의2터널)")
    ap.add_argument("--out", default=None,
                    help="출력 PDF 경로 (기본: data/<dataset>/report_<YYYYMMDD>.pdf)")
    ap.add_argument("--data-root", default=str(ROOT / "data"))
    ap.add_argument("--all", action="store_true",
                    help="모든 시점의 비교를 포함 (기본은 최신 1개만)")
    args = ap.parse_args()

    root = Path(args.data_root) / args.dataset
    plumb_path = root / "plumb.json"
    if not plumb_path.exists():
        print(f"[ERROR] plumb.json not found: {plumb_path}", file=sys.stderr)
        sys.exit(1)
    plumb = json.loads(plumb_path.read_text(encoding="utf-8"))

    # 모든 wall_tilt_*.json 수집
    tilt_files = sorted(root.glob("wall_tilt_*.json"))
    if not tilt_files:
        print(f"[ERROR] no wall_tilt_*.json in {root}", file=sys.stderr)
        sys.exit(1)
    tilts = []
    for f in tilt_files:
        t = json.loads(f.read_text(encoding="utf-8"))
        t["days_span"] = days_between(t["reference"], t["target"])
        tilts.append(t)
    tilts.sort(key=lambda x: x["target"])

    # 기본은 최신 1개만 — --all 로 모든 시점 포함
    if not args.all and len(tilts) > 1:
        tilts = tilts[-1:]

    scans = sorted(set([t["reference"] for t in tilts] + [t["target"] for t in tilts]))

    latest = tilts[-1]
    summary = {
        "reference_date": fmt_date(latest["reference"]),
        "latest_date": fmt_date(latest["target"]),
        "days_span": latest["days_span"],
        "plumb_axis": plumb["axis"],
        "plumb_tilt": plumb["tilt_from_z_deg"],
        "latest_tilt": latest["tilt_change_deg"],
        "latest_direction": latest["tilt_direction"] if latest["significant"] else "n/a",
        "latest_ci": latest["tilt_change_ci95_deg"],
        "latest_verdict": latest["verdict"],
    }

    out = Path(args.out) if args.out else \
          root / f"report_{datetime.now():%Y%m%d}.pdf"

    print(f"[generate] {out}")
    with PdfPages(out) as pdf:
        page_cover(pdf, args.site, args.dataset, scans, summary)
        page_plumb(pdf, plumb)
        # 시계열 비교 페이지는 2개 이상 비교쌍이 있을 때만 의미 있음
        if len(tilts) >= 2:
            page_timeseries(pdf, tilts)
        for i, t in enumerate(tilts, start=1):
            page_regression_detail(pdf, t, i, len(tilts))
        page_conclusion(pdf, summary, tilts)
    n_pages = 3 + len(tilts) + (1 if len(tilts) >= 2 else 0)
    print(f"[done] {n_pages} pages → {out}")


if __name__ == "__main__":
    main()
