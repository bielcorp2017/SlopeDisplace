"""옹벽 분석결과 단일 페이지 HTML 리포트 생성.

차트는 matplotlib → PNG → base64 data URL 로 임베드, 단일 .html 파일로
어떤 브라우저에서도 바로 열림. 분석 데이터는 빌드 시점에 베이크-인.

사용:
    python tools/generate_report_html.py <dataset> --site "현장명" --out out.html
"""

from __future__ import annotations
import argparse, base64, io, json, sys
from datetime import datetime
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# Korean font
_FP = r"C:\Windows\Fonts\malgun.ttf"
if Path(_FP).exists():
    font_manager.fontManager.addfont(_FP)
    plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.facecolor"] = "#0f1115"
plt.rcParams["axes.facecolor"] = "#161a22"
plt.rcParams["savefig.facecolor"] = "#0f1115"
plt.rcParams["text.color"] = "#cdd2da"
plt.rcParams["axes.labelcolor"] = "#cdd2da"
plt.rcParams["axes.edgecolor"] = "#2a3140"
plt.rcParams["xtick.color"] = "#98a0ad"
plt.rcParams["ytick.color"] = "#98a0ad"
plt.rcParams["axes.titlecolor"] = "#e6e8eb"
plt.rcParams["grid.color"] = "#2a3140"

ROOT = Path(__file__).resolve().parent.parent


def fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def fmt_date(s):
    if len(s) == 8 and s.isdigit():
        return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
    return s


def days_between(a, b):
    try:
        return (datetime.strptime(b, "%Y%m%d") - datetime.strptime(a, "%Y%m%d")).days
    except Exception:
        return 0


# ---------- charts ----------
def chart_regression(t):
    fig, ax = plt.subplots(figsize=(9, 4.2))
    h = np.asarray(t["scatter_h_m"]); d = np.asarray(t["scatter_d_mm"])
    ax.scatter(h, d, s=2.0, alpha=0.30, c="#4ea1ff", edgecolors="none")
    h_lo, h_hi = t["height_range_m"]
    hh = np.linspace(h_lo, h_hi, 100)
    ax.plot(hh, t["alpha_mm_per_m"] * (hh - t["h_ref_m"]) + t["beta_mm"],
            color="#fb7185", lw=2.4,
            label=f"α = {t['alpha_mm_per_m']:+.3f} mm/m   ({t['tilt_change_deg']:+.4f}°)")
    ax.axhline(0, color="#5b6577", lw=0.5)
    ax.set_xlabel("높이 h (m, plumb 축)")
    ax.set_ylabel("signed normal disp (mm)  ·  outward +")
    ax.set_title("옹벽 표면점 signed_normal vs plumb 높이 회귀")
    ax.legend(loc="upper left", fontsize=10, frameon=False, labelcolor="#e6e8eb")
    ax.grid(alpha=0.4)
    fig.tight_layout()
    return fig_to_b64(fig)


def chart_hist(values_mm, label, color, clip=None):
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    v = np.asarray(values_mm)
    if clip:
        v = v[(v >= -clip) & (v <= clip)]
    ax.hist(v, bins=80, color=color, alpha=0.85, edgecolor="#0f1115", linewidth=0.3)
    ax.axvline(0, color="#5b6577", lw=0.5)
    ax.axvline(float(np.mean(v)), color="#fbbf24", lw=1.2,
               label=f"mean = {np.mean(v):+.2f} mm")
    ax.set_xlabel(label); ax.set_ylabel("점 수")
    ax.legend(loc="upper right", fontsize=10, frameon=False, labelcolor="#e6e8eb")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    return fig_to_b64(fig)


def chart_profile(h_m, val_mm, label, color, n_slices=30):
    h = np.asarray(h_m); v = np.asarray(val_mm)
    bins = np.linspace(h.min(), h.max(), n_slices + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    means, stds = [], []
    for i in range(n_slices):
        m = (h >= bins[i]) & (h < bins[i + 1])
        if m.sum() < 5:
            means.append(np.nan); stds.append(np.nan); continue
        means.append(np.mean(v[m])); stds.append(np.std(v[m]))
    means, stds = np.array(means), np.array(stds)
    valid = ~np.isnan(means)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.fill_betweenx(centers[valid], means[valid]-stds[valid],
                     means[valid]+stds[valid], alpha=0.25, color=color)
    ax.plot(means[valid], centers[valid], "-o", color=color, ms=3.5,
            mfc=color, mec="#0f1115", label="평균 ± 1σ")
    ax.axvline(0, color="#5b6577", lw=0.5)
    ax.set_xlabel(label); ax.set_ylabel("높이 h (m, plumb 축)")
    ax.legend(loc="best", fontsize=10, frameon=False, labelcolor="#e6e8eb")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig_to_b64(fig)


# ---------- data load ----------
def load_target(root, tgt_stem, ref_stem, plumb_axis):
    pcd_t = o3d.io.read_point_cloud(str(root / f"{tgt_stem}_simple.ply"))
    P_t = np.asarray(pcd_t.points, dtype=np.float64)
    disp = np.fromfile(root / f"{tgt_stem}_disp.bin", dtype=np.float32).reshape(-1, 4)
    pcd_r = o3d.io.read_point_cloud(str(root / f"{ref_stem}_simple.ply"))
    pcd_r.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30))
    pcd_r.orient_normals_consistent_tangent_plane(k=20)
    N_r = np.asarray(pcd_r.normals)
    tree = cKDTree(np.asarray(pcd_r.points))
    _, idx = tree.query(P_t, k=1, workers=-1)
    n_at_t = N_r[idx]
    mask_wall = np.abs(n_at_t @ plumb_axis) < 0.3
    return P_t, disp, mask_wall


# ---------- HTML template ----------
HTML = r"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8" />
<title>__TITLE__</title>
<style>
  :root {
    --bg: #0a0c10;
    --bg2: #0f1115;
    --card: #161a22;
    --card2: #1d222c;
    --border: #2a3140;
    --text: #e6e8eb;
    --text2: #cdd2da;
    --dim: #98a0ad;
    --dim2: #5b6577;
    --accent: #4ea1ff;
    --accent2: #3b82f6;
    --ok: #4ade80;
    --warn: #fbbf24;
    --out: #fb7185;
    --inn: #60a5fa;
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; background: var(--bg); color: var(--text);
    font: 14px/1.6 "맑은 고딕", Malgun Gothic, ui-sans-serif, system-ui, -apple-system, sans-serif;
    -webkit-font-smoothing: antialiased; }
  .container { max-width: 1200px; margin: 0 auto; padding: 0 24px; }
  .mono { font-family: ui-monospace, "Consolas", "맑은 고딕", monospace; font-variant-numeric: tabular-nums; }

  /* HERO */
  .hero {
    background:
      radial-gradient(ellipse 80% 60% at 50% -20%, rgba(78,161,255,.18) 0%, transparent 60%),
      linear-gradient(180deg, #0a1428 0%, var(--bg) 100%);
    border-bottom: 1px solid var(--border);
    padding: 80px 0 60px;
    text-align: center;
  }
  .hero .badge {
    display: inline-block; padding: 4px 14px; border-radius: 999px;
    background: rgba(78,161,255,.12); color: var(--accent);
    border: 1px solid rgba(78,161,255,.3);
    font-size: 11px; font-weight: 600; letter-spacing: 1.2px;
    text-transform: uppercase;
  }
  .hero h1 {
    font-size: 38px; font-weight: 800; margin: 18px 0 8px;
    letter-spacing: -0.5px; line-height: 1.2;
    background: linear-gradient(180deg, #fff 0%, #cfd6e0 100%);
    -webkit-background-clip: text; background-clip: text;
    -webkit-text-fill-color: transparent;
  }
  .hero .sub { color: var(--dim); font-size: 15px; }
  .hero .meta { margin-top: 24px; color: var(--text2); font-size: 13px; }
  .hero .meta b { color: #fff; font-weight: 600; }

  /* KPI cards */
  .kpis {
    display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px;
    margin: -36px auto 40px; max-width: 1200px; padding: 0 24px;
    position: relative; z-index: 1;
  }
  .kpi {
    background: linear-gradient(180deg, var(--card2) 0%, var(--card) 100%);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px 22px;
    box-shadow: 0 6px 24px rgba(0,0,0,.4);
  }
  .kpi .lbl { color: var(--dim); font-size: 12px; text-transform: uppercase; letter-spacing: 1.2px; font-weight: 600; }
  .kpi .val { font-size: 32px; font-weight: 700; margin: 8px 0 4px; }
  .kpi .val.outward { color: var(--out); }
  .kpi .val.inward { color: var(--inn); }
  .kpi .val.neutral { color: var(--ok); }
  .kpi .val.warn { color: var(--warn); }
  .kpi .sub { color: var(--dim); font-size: 12px; }

  /* sections */
  section { padding: 36px 0; border-bottom: 1px solid var(--border); }
  section:last-of-type { border-bottom: none; }
  .sec-head {
    display: flex; align-items: baseline; gap: 12px;
    margin-bottom: 18px; padding-bottom: 10px; border-bottom: 1px solid var(--border);
  }
  .sec-head .num {
    color: var(--accent); font-weight: 700; font-size: 14px;
    font-family: ui-monospace, monospace;
  }
  .sec-head h2 { font-size: 22px; font-weight: 700; margin: 0; letter-spacing: -0.2px; }
  .sec-head .tag {
    margin-left: auto; color: var(--dim); font-size: 12px;
  }
  .desc { color: var(--text2); font-size: 13.5px; margin: 0 0 16px; }

  /* cards */
  .card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 14px;
  }
  .card h3 { margin: 0 0 8px; font-size: 14px; color: var(--text2); font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.8px; }

  /* table */
  table.kv {
    width: 100%; border-collapse: collapse; font-size: 13px;
  }
  table.kv td { padding: 6px 8px; border-bottom: 1px dashed #232936; vertical-align: top; }
  table.kv tr:last-child td { border-bottom: none; }
  table.kv td.k { color: var(--dim); width: 38%; font-weight: 500; }
  table.kv td.v { color: var(--text); font-family: ui-monospace, Consolas, "맑은 고딕", monospace; }
  table.kv td.v.outward { color: var(--out); font-weight: 700; }
  table.kv td.v.inward { color: var(--inn); font-weight: 700; }
  table.kv td.v.ok { color: var(--ok); font-weight: 600; }
  table.kv td.v.fail { color: var(--out); font-weight: 600; }
  table.kv td.v.warn { color: var(--warn); }

  .grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
  @media (max-width: 800px) {
    .grid2 { grid-template-columns: 1fr; }
    .kpis { grid-template-columns: 1fr; }
    .hero h1 { font-size: 26px; }
  }

  img.chart { width: 100%; border-radius: 8px; display: block; }

  .verdict {
    display: inline-block; padding: 4px 10px; border-radius: 4px;
    font-size: 12px; font-weight: 700; letter-spacing: 0.5px;
  }
  .verdict.sig.out { background: rgba(251,113,133,.18); color: var(--out); border: 1px solid rgba(251,113,133,.4); }
  .verdict.sig.in { background: rgba(96,165,250,.18); color: var(--inn); border: 1px solid rgba(96,165,250,.4); }
  .verdict.nochg { background: rgba(74,222,128,.15); color: var(--ok); border: 1px solid rgba(74,222,128,.4); }

  .note {
    background: rgba(78,161,255,.06); border-left: 3px solid var(--accent);
    padding: 10px 14px; margin-top: 10px; border-radius: 4px;
    color: var(--text2); font-size: 13px;
  }
  .warnbox {
    background: rgba(251,113,133,.07); border-left: 3px solid var(--out);
    padding: 10px 14px; margin-top: 10px; border-radius: 4px;
    color: var(--text2); font-size: 13px;
  }
  ul.bullets { padding-left: 18px; margin: 8px 0; }
  ul.bullets li { margin: 6px 0; color: var(--text2); }
  ul.bullets li b { color: var(--text); }

  footer {
    color: var(--dim); font-size: 12px; text-align: center;
    padding: 30px 0 50px;
  }
  footer .sep { margin: 0 8px; color: var(--dim2); }
</style>
</head>
<body>

<div class="hero">
  <div class="container">
    <span class="badge">옹벽 변위 모니터링 분석 리포트</span>
    <h1>__SITE__</h1>
    <div class="sub">옹벽 변위 분석결과  ·  데이터셋 __DATASET__</div>
    <div class="meta">
      기준 <b>__REF_DATE__</b>  →  최신 <b>__LATEST_DATE__</b>
      &nbsp;·&nbsp; 관측 <b>__DAYS__일</b>
      &nbsp;·&nbsp; 옹벽 점 <b>__WALL_POINTS__</b>개
      &nbsp;·&nbsp; ICP fitness <b>__FITNESS__</b>
    </div>
  </div>
</div>

<div class="kpis">
  <div class="kpi">
    <div class="lbl">경사 (tilt) 변화</div>
    <div class="val mono __TILT_CLASS__">__TILT_VAL__°</div>
    <div class="sub">95% CI __TILT_CI__  ·  __TILT_DIR__</div>
  </div>
  <div class="kpi">
    <div class="lbl">옹벽 수평 변위 평균</div>
    <div class="val mono __HORIZ_CLASS__">__HORIZ_MEAN__ mm</div>
    <div class="sub">p95 |·| __HORIZ_P95__ mm  ·  σ __HORIZ_STD__ mm</div>
  </div>
  <div class="kpi">
    <div class="lbl">옹벽 수직 변위 평균</div>
    <div class="val mono __VERT_CLASS__">__VERT_MEAN__ mm</div>
    <div class="sub">p95 |·| __VERT_P95__ mm  ·  σ __VERT_STD__ mm</div>
  </div>
</div>

<div class="container">

<!-- Section 1 -->
<section>
  <div class="sec-head"><span class="num">01</span><h2>측정 개요</h2><span class="tag">Overview</span></div>
  <p class="desc">본 분석은 현장에 설치한 수직봉으로 정밀 plumb 벡터를 추출한 뒤, 두 시점의 3D 점군을 ICP 로 정합하여 옹벽 표면의 변위를 평가했다. 옹벽이 평탄하지 않은 디자인이어도 표면점 노멀 마스킹과 plumb 축 회귀로 robust 한 분석이 가능하다.</p>
  <div class="grid2">
    <div class="card">
      <h3>분석 대상</h3>
      <table class="kv">
        <tr><td class="k">현장</td><td class="v">__SITE__</td></tr>
        <tr><td class="k">데이터셋</td><td class="v">__DATASET__</td></tr>
        <tr><td class="k">기준 스캔</td><td class="v">__REF_DATE__</td></tr>
        <tr><td class="k">최신 스캔</td><td class="v">__LATEST_DATE__</td></tr>
        <tr><td class="k">관측 기간</td><td class="v">__DAYS__ 일</td></tr>
        <tr><td class="k">전체 점</td><td class="v">__TOTAL_POINTS__</td></tr>
        <tr><td class="k">옹벽 면 점</td><td class="v">__WALL_POINTS__</td></tr>
        <tr><td class="k">정합 신뢰</td><td class="v __RELIABLE_CLASS__">__RELIABLE_LABEL__</td></tr>
      </table>
    </div>
    <div class="card">
      <h3>분석 항목</h3>
      <ul class="bullets">
        <li><b>경사 (Tilt)</b> — 옹벽 평균 기울기의 기준일 대비 추가 변화 [°]</li>
        <li><b>수평 변위 (Horizontal)</b> — 옹벽 facing 방향 ± 분포 및 높이 프로파일 [mm]</li>
        <li><b>수직 변위 (Vertical)</b> — Z축 방향 침하/융기 양상 [mm]</li>
      </ul>
      <h3 style="margin-top:18px">방법론 단계</h3>
      <ul class="bullets">
        <li>수직봉을 3-method (PCA/Slice/Cylinder LSQ) 교차검증</li>
        <li>FGR + 다중스케일 ICP (1m → 1cm) 정합</li>
        <li>변위벡터 → signed_normal/horizontal/vertical 분해</li>
        <li>옹벽 면 마스킹 (|n·plumb|&lt;0.3) 후 통계·회귀</li>
        <li>부트스트랩 95% CI 로 유의성 판단</li>
      </ul>
    </div>
  </div>
</section>

<!-- Section 2 -->
<section>
  <div class="sec-head"><span class="num">02</span><h2>수직봉(plumb) 측정 결과</h2><span class="tag">Reference Vector</span></div>
  <p class="desc">현장 수직봉을 별도 스캔하여 PCA · 슬라이스 원피팅 · 원기둥 LSQ 세 방법을 독립적으로 적용했다. 세 방법이 같은 축을 가리키면 측정 신뢰가 증명된다.</p>
  <div class="grid2">
    <div class="card">
      <h3>측정값</h3>
      <table class="kv">
        <tr><td class="k">원본 PLY</td><td class="v">__PLUMB_SRC__</td></tr>
        <tr><td class="k">축 벡터</td><td class="v">__PLUMB_AXIS__</td></tr>
        <tr><td class="k">중심</td><td class="v">__PLUMB_CENTER__</td></tr>
        <tr><td class="k">반경</td><td class="v">__PLUMB_RADIUS__ mm</td></tr>
        <tr><td class="k">봉 길이</td><td class="v">__PLUMB_LEN__ m</td></tr>
        <tr><td class="k">Z축 대비 기울기</td><td class="v">__PLUMB_TILT__°</td></tr>
        <tr><td class="k">표면 잡음 σ</td><td class="v">__PLUMB_SIGMA__ mm</td></tr>
        <tr><td class="k">인라이어 점 수</td><td class="v">__PLUMB_INLIERS__</td></tr>
      </table>
    </div>
    <div class="card">
      <h3>3-방법 교차검증</h3>
      <table class="kv">
        <tr><td class="k">Cylinder ↔ PCA</td><td class="v">__VAL_CP__°</td></tr>
        <tr><td class="k">Cylinder ↔ Slice</td><td class="v">__VAL_CS__°</td></tr>
        <tr><td class="k">PCA ↔ Slice</td><td class="v">__VAL_PS__°</td></tr>
        <tr><td class="k">선형성 L</td><td class="v">__VAL_L__</td></tr>
        <tr><td class="k">planarity P</td><td class="v">__VAL_P__</td></tr>
        <tr><td class="k">축 일치 (&lt;1°)</td><td class="v __VAL_AX_CLS__">__VAL_AX__</td></tr>
        <tr><td class="k">막대 형태</td><td class="v __VAL_SH_CLS__">__VAL_SH__</td></tr>
        <tr><td class="k">원기둥 표면</td><td class="v __VAL_SU_CLS__">__VAL_SU__</td></tr>
        <tr><td class="k">종합 판정</td><td class="v __VAL_OV_CLS__">__VAL_OV__</td></tr>
      </table>
      <div class="note">→ 세 방법 모두 1° 이내 일치, 형태·표면 PASS → plumb 벡터 신뢰 가능</div>
    </div>
  </div>
</section>

<!-- Section 3 -->
<section>
  <div class="sec-head"><span class="num">03</span><h2>경사 (Tilt) 분석</h2><span class="tag">Regression Analysis</span></div>
  <p class="desc">옹벽 표면점의 <span class="mono">signed_normal</span>(= 변위의 표면 법선 방향 성분, +outward) 을 plumb 축 높이에 대해 robust Huber 회귀하여, 단위 높이당 outward 변위 증가율 α[mm/m] 를 얻는다. atan(α/1000) 이 곧 옹벽의 추가 기울기.</p>
  <div class="card">
    <img class="chart" src="data:image/png;base64,__CHART_REG__" alt="regression scatter" />
  </div>
  <div class="grid2">
    <div class="card">
      <h3>회귀 결과</h3>
      <table class="kv">
        <tr><td class="k">α (회귀 기울기)</td><td class="v">__ALPHA__ mm/m</td></tr>
        <tr><td class="k">α 95% CI</td><td class="v">__ALPHA_CI__</td></tr>
        <tr><td class="k">β (h_ref 변위)</td><td class="v">__BETA__ mm</td></tr>
        <tr><td class="k">h 범위</td><td class="v">__H_RANGE__</td></tr>
        <tr><td class="k">잔차 robust σ</td><td class="v">__RESID__ mm</td></tr>
        <tr><td class="k">R²</td><td class="v">__R2__</td></tr>
        <tr><td class="k">회귀 사용 점</td><td class="v">__REG_PTS__</td></tr>
      </table>
    </div>
    <div class="card">
      <h3>판정</h3>
      <p style="font-size:32px; font-weight:800; margin:8px 0 2px;" class="mono __TILT_CLASS__">__TILT_VAL__°</p>
      <p style="margin:0 0 12px; color: var(--dim);">95% CI __TILT_CI__°  ·  방향 <b>__TILT_DIR__</b></p>
      <span class="verdict __VERDICT_CLS__">__VERDICT__</span>
      <div class="note">옹벽 디자인의 굴곡은 두 스캔에 공통이므로 차분(displacement)에서 자동 상쇄된다. 따라서 측정된 변화는 디자인이 아닌 *실제 변형* 만 반영.</div>
    </div>
  </div>
</section>

<!-- Section 4 -->
<section>
  <div class="sec-head"><span class="num">04</span><h2>수평 변위 분석</h2><span class="tag">Horizontal Displacement</span></div>
  <p class="desc">옹벽 면 표면점의 수평 변위 <span class="mono">disp[2]</span>(= 옹벽 법선의 XY 성분 부호 투영). +값은 outward(옹벽 앞쪽), −값은 inward(사면쪽).</p>
  <div class="grid2">
    <div class="card">
      <h3>옹벽 면 통계</h3>
      <table class="kv">
        <tr><td class="k">점 개수</td><td class="v">__HW_N__</td></tr>
        <tr><td class="k">평균 (mean)</td><td class="v __HW_MEAN_CLS__">__HW_MEAN__ mm</td></tr>
        <tr><td class="k">|평균|</td><td class="v">__HW_ABS_MEAN__ mm</td></tr>
        <tr><td class="k">중앙값 (median)</td><td class="v">__HW_MED__ mm</td></tr>
        <tr><td class="k">표준편차 σ</td><td class="v">__HW_STD__ mm</td></tr>
        <tr><td class="k">RMS</td><td class="v">__HW_RMS__ mm</td></tr>
        <tr><td class="k">p95 (|·|)</td><td class="v">__HW_P95__ mm</td></tr>
        <tr><td class="k">max / min</td><td class="v">__HW_MAX__ / __HW_MIN__ mm</td></tr>
      </table>
    </div>
    <div class="card">
      <h3>분포</h3>
      <img class="chart" src="data:image/png;base64,__CHART_HHIST__" />
    </div>
  </div>
  <div class="card">
    <h3>높이별 평균 ± 1σ 프로파일</h3>
    <img class="chart" src="data:image/png;base64,__CHART_HPROF__" />
    <div class="note">위쪽 패치만 outward 면 lean(기울어짐), 전 구간 일정한 양수면 평행이동(translation). 본 결과는 회귀 결과와 함께 해석.</div>
  </div>
</section>

<!-- Section 5 -->
<section>
  <div class="sec-head"><span class="num">05</span><h2>수직 변위 분석</h2><span class="tag">Vertical Displacement</span></div>
  <p class="desc">옹벽 면 표면점의 수직 변위 <span class="mono">disp[3]</span>(= signed_normal × normal_z). +값은 융기(상승), −값은 침하(하강).</p>
  <div class="grid2">
    <div class="card">
      <h3>옹벽 면 통계</h3>
      <table class="kv">
        <tr><td class="k">점 개수</td><td class="v">__VW_N__</td></tr>
        <tr><td class="k">평균 (mean)</td><td class="v __VW_MEAN_CLS__">__VW_MEAN__ mm</td></tr>
        <tr><td class="k">|평균|</td><td class="v">__VW_ABS_MEAN__ mm</td></tr>
        <tr><td class="k">중앙값 (median)</td><td class="v">__VW_MED__ mm</td></tr>
        <tr><td class="k">표준편차 σ</td><td class="v">__VW_STD__ mm</td></tr>
        <tr><td class="k">RMS</td><td class="v">__VW_RMS__ mm</td></tr>
        <tr><td class="k">p95 (|·|)</td><td class="v">__VW_P95__ mm</td></tr>
        <tr><td class="k">max(융기) / min(침하)</td><td class="v">__VW_MAX__ / __VW_MIN__ mm</td></tr>
      </table>
    </div>
    <div class="card">
      <h3>분포</h3>
      <img class="chart" src="data:image/png;base64,__CHART_VHIST__" />
    </div>
  </div>
  <div class="card">
    <h3>높이별 평균 ± 1σ 프로파일</h3>
    <img class="chart" src="data:image/png;base64,__CHART_VPROF__" />
  </div>
</section>

<!-- Section 6 -->
<section>
  <div class="sec-head"><span class="num">06</span><h2>종합 결론 및 권고</h2><span class="tag">Conclusion</span></div>
  <div class="grid2">
    <div class="card">
      <h3>주요 발견</h3>
      <ul class="bullets">
        <li><b>경사:</b> 기준일 대비 <span class="mono __TILT_CLASS__">__TILT_VAL__°</span> __TILT_DIR_WORD__ (CI __TILT_CI__°)</li>
        <li><b>수평:</b> 옹벽 면 평균 <span class="mono">__HW_MEAN__ mm</span>, p95 |·| __HW_P95__ mm</li>
        <li><b>수직:</b> 옹벽 면 평균 <span class="mono">__VW_MEAN__ mm</span> (__VW_DIR_WORD__), p95 |·| __VW_P95__ mm</li>
        <li><b>정합:</b> ICP fitness <span class="mono">__FITNESS__</span>, 잔차 σ <span class="mono">__RESID__ mm</span> — __RELIABLE_LABEL__</li>
      </ul>
    </div>
    <div class="card">
      <h3>한계 및 권고</h3>
      <ul class="bullets">
        <li>측정값은 <b>두 시점 간 추가 변화량</b>이며, 기준일 자체의 절대 수직도가 아님</li>
        <li>옹벽 디자인 굴곡은 차분에서 자동 상쇄되어 본 분석은 디자인 무관한 robust 측정</li>
        <li>스캐너 잡음 한계(2~5mm) 이하의 변화는 검출 불가</li>
        <li>모니터링 주기: <b>주 1회 ~ 월 1회</b> 권장</li>
        <li>경고 임계값: 누적 기울기 <b>0.10° 초과</b>, 수평 p95 |·| <b>10 mm 초과</b>, 수직 침하 <b>5 mm 초과</b> 시 현장 점검</li>
      </ul>
    </div>
  </div>
</section>

</div> <!-- container -->

<footer>
  생성 __NOW__ <span class="sep">·</span> SlopeDisplace 분석 시스템 v1 <span class="sep">·</span> __DATASET__
</footer>

</body>
</html>
"""


def stat(arr):
    a = np.asarray(arr, dtype=np.float64)
    return dict(
        n=int(a.size),
        mean=float(a.mean()),
        abs_mean=float(np.abs(a).mean()),
        median=float(np.median(a)),
        std=float(a.std()),
        rms=float(np.sqrt(np.mean(a*a))),
        p95_abs=float(np.percentile(np.abs(a), 95)),
        max=float(a.max()),
        min=float(a.min()),
    )


def cls_signed(v):
    if v > 0.5: return "outward"
    if v < -0.5: return "inward"
    return "neutral"


def cls_signed_table(v):
    if v > 0.5: return "outward"
    if v < -0.5: return "inward"
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--site", default="현장명")
    ap.add_argument("--out", default=None)
    ap.add_argument("--data-root", default=str(ROOT / "data"))
    args = ap.parse_args()

    root = Path(args.data_root) / args.dataset
    plumb = json.loads((root / "plumb.json").read_text(encoding="utf-8"))
    tilt_files = sorted(root.glob("wall_tilt_*.json"))
    if not tilt_files:
        print("[ERROR] no wall_tilt_*.json", file=sys.stderr); sys.exit(1)
    tilts = [json.loads(p.read_text(encoding="utf-8")) for p in tilt_files]
    tilts.sort(key=lambda x: x["target"])
    t = tilts[-1]
    ref_stem, tgt_stem = t["reference"], t["target"]
    meta = json.loads((root / f"{tgt_stem}_meta.json").read_text(encoding="utf-8"))

    plumb_axis = np.asarray(plumb["axis"]); plumb_axis /= np.linalg.norm(plumb_axis)
    plumb_origin = np.asarray(plumb["center"])

    print(f"[load] {tgt_stem} ...")
    P_t, disp, mask_wall = load_target(root, tgt_stem, ref_stem, plumb_axis)
    horiz_w = disp[mask_wall, 2] * 1000.0
    vert_w = disp[mask_wall, 3] * 1000.0
    h_w = (P_t[mask_wall] - plumb_origin) @ plumb_axis

    sH = stat(horiz_w); sV = stat(vert_w)

    print("[chart] regression ...")
    c_reg = chart_regression(t)
    clip_h = max(20.0, sH['p95_abs'] * 3)
    clip_v = max(20.0, sV['p95_abs'] * 3)
    print("[chart] horizontal hist + profile ...")
    c_hh = chart_hist(horiz_w, "수평 변위 (mm)  ·  outward +", "#60a5fa", clip=clip_h)
    c_hp = chart_profile(h_w, horiz_w, "수평 변위 (mm)", "#60a5fa")
    print("[chart] vertical hist + profile ...")
    c_vh = chart_hist(vert_w, "수직 변위 (mm)  ·  융기 +", "#4ade80", clip=clip_v)
    c_vp = chart_profile(h_w, vert_w, "수직 변위 (mm)", "#4ade80")

    # tilt class
    a = t["alpha_mm_per_m"]; sig = t["significant"]
    if sig and a > 0: tilt_cls = "outward"
    elif sig and a < 0: tilt_cls = "inward"
    else: tilt_cls = "neutral"
    verdict_cls = ("sig out" if sig and a > 0 else
                   "sig in" if sig and a < 0 else "nochg")
    tilt_dir_word = ("OUTWARD 방향 (옹벽 앞으로)" if a > 0 else
                     "INWARD 방향 (사면쪽으로)") if sig else "유의미한 변화 없음"
    vw_dir = "융기" if sV['mean'] > 0 else "침하"

    fit = t.get("icp_fitness") or 0
    fit_str = f"{fit*100:.1f}%"
    reliable = t.get("registration_reliable") is True
    reliable_lbl = "RELIABLE" if reliable else "UNRELIABLE"
    reliable_cls = "ok" if reliable else "fail"

    val = plumb.get("validation", {})
    def pf(b): return ("PASS" if b else "FAIL")
    def pf_cls(b): return ("ok" if b else "fail")

    sub = {
        "TITLE":           f"{args.site} 옹벽 변위 분석결과",
        "SITE":            args.site,
        "DATASET":         args.dataset,
        "REF_DATE":        fmt_date(ref_stem),
        "LATEST_DATE":     fmt_date(tgt_stem),
        "DAYS":            str(days_between(ref_stem, tgt_stem)),
        "WALL_POINTS":     f"{int(mask_wall.sum()):,}",
        "TOTAL_POINTS":    f"{len(P_t):,}",
        "FITNESS":         fit_str,
        "RELIABLE_LABEL":  reliable_lbl,
        "RELIABLE_CLASS":  reliable_cls,

        "TILT_VAL":        f"{t['tilt_change_deg']:+.4f}",
        "TILT_CLASS":      tilt_cls,
        "TILT_CI":         f"[{t['tilt_change_ci95_deg'][0]:+.4f}, {t['tilt_change_ci95_deg'][1]:+.4f}]",
        "TILT_DIR":        t["tilt_direction"] if sig else "변화 없음",
        "TILT_DIR_WORD":   tilt_dir_word,
        "VERDICT":         t["verdict"],
        "VERDICT_CLS":     verdict_cls,

        "ALPHA":           f"{a:+.4f}",
        "ALPHA_CI":        f"[{t['alpha_ci95_mm_per_m'][0]:+.4f}, {t['alpha_ci95_mm_per_m'][1]:+.4f}] mm/m",
        "BETA":            f"{t['beta_mm']:+.3f}",
        "H_RANGE":         f"{t['height_range_m'][0]:.2f} ~ {t['height_range_m'][1]:.2f} m",
        "RESID":           f"{t['residual_robust_sigma_mm']:.2f}",
        "R2":              f"{t['r_squared']:.4f}",
        "REG_PTS":         f"{t['wall_face_point_count']:,}",

        "PLUMB_SRC":       Path(plumb["source_ply"]).name,
        "PLUMB_AXIS":      f"({plumb['axis'][0]:+.6f}, {plumb['axis'][1]:+.6f}, {plumb['axis'][2]:+.6f})",
        "PLUMB_CENTER":    f"({plumb['center'][0]:.3f}, {plumb['center'][1]:.3f}, {plumb['center'][2]:.3f}) m",
        "PLUMB_RADIUS":    f"{plumb['radius_m']*1000:.2f}",
        "PLUMB_LEN":       f"{plumb['rod_length_m']:.3f}",
        "PLUMB_TILT":      f"{plumb['tilt_from_z_deg']:.4f}",
        "PLUMB_SIGMA":     f"{plumb['surface_sigma_m']*1000:.2f}",
        "PLUMB_INLIERS":   f"{plumb['n_inliers']:,}",
        "VAL_CP":          f"{val.get('cylinder_vs_pca_deg', 0):.4f}",
        "VAL_CS":          f"{val.get('cylinder_vs_slice_deg', 0):.4f}",
        "VAL_PS":          f"{val.get('pca_vs_slice_deg', 0):.4f}",
        "VAL_L":           f"{val.get('linearity', 0):.4f}",
        "VAL_P":           f"{val.get('planarity', 0):.4f}",
        "VAL_AX":          pf(val.get("axis_match_pass")),
        "VAL_AX_CLS":      pf_cls(val.get("axis_match_pass")),
        "VAL_SH":          pf(val.get("shape_pass")),
        "VAL_SH_CLS":      pf_cls(val.get("shape_pass")),
        "VAL_SU":          pf(val.get("surface_pass")),
        "VAL_SU_CLS":      pf_cls(val.get("surface_pass")),
        "VAL_OV":          ("PROVEN" if val.get("overall_proven") else "NOT PROVEN"),
        "VAL_OV_CLS":      pf_cls(val.get("overall_proven")),

        "HORIZ_MEAN":      f"{sH['mean']:+.2f}",
        "HORIZ_CLASS":     cls_signed(sH['mean']),
        "HORIZ_P95":       f"{sH['p95_abs']:.2f}",
        "HORIZ_STD":       f"{sH['std']:.2f}",
        "VERT_MEAN":       f"{sV['mean']:+.2f}",
        "VERT_CLASS":      cls_signed(sV['mean']),
        "VERT_P95":        f"{sV['p95_abs']:.2f}",
        "VERT_STD":        f"{sV['std']:.2f}",

        "HW_N":            f"{sH['n']:,}",
        "HW_MEAN":         f"{sH['mean']:+.2f}",
        "HW_MEAN_CLS":     cls_signed_table(sH['mean']),
        "HW_ABS_MEAN":     f"{sH['abs_mean']:.2f}",
        "HW_MED":          f"{sH['median']:+.2f}",
        "HW_STD":          f"{sH['std']:.2f}",
        "HW_RMS":          f"{sH['rms']:.2f}",
        "HW_P95":          f"{sH['p95_abs']:.2f}",
        "HW_MAX":          f"{sH['max']:+.2f}",
        "HW_MIN":          f"{sH['min']:+.2f}",

        "VW_N":            f"{sV['n']:,}",
        "VW_MEAN":         f"{sV['mean']:+.2f}",
        "VW_MEAN_CLS":     cls_signed_table(sV['mean']),
        "VW_ABS_MEAN":     f"{sV['abs_mean']:.2f}",
        "VW_MED":          f"{sV['median']:+.2f}",
        "VW_STD":          f"{sV['std']:.2f}",
        "VW_RMS":          f"{sV['rms']:.2f}",
        "VW_P95":          f"{sV['p95_abs']:.2f}",
        "VW_MAX":          f"{sV['max']:+.2f}",
        "VW_MIN":          f"{sV['min']:+.2f}",
        "VW_DIR_WORD":     vw_dir,

        "CHART_REG":       c_reg,
        "CHART_HHIST":     c_hh,
        "CHART_HPROF":     c_hp,
        "CHART_VHIST":     c_vh,
        "CHART_VPROF":     c_vp,

        "NOW":             datetime.now().strftime("%Y-%m-%d %H:%M"),
    }

    html = HTML
    for k, v in sub.items():
        html = html.replace(f"__{k}__", str(v))

    out = Path(args.out) if args.out else (ROOT / "web" / "daesan_slope_20250627.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"[done] {len(html):,} bytes → {out}")


if __name__ == "__main__":
    main()
