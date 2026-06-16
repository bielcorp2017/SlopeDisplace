import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const $ = (id) => document.getElementById(id);

// ---- query params ----
const qs = new URLSearchParams(location.search);
const dataset = qs.get("dataset");
const stem = qs.get("stem");

// ---- three.js scene ----
const canvas = $("canvas");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0b0d12);

const camera = new THREE.PerspectiveCamera(50, 1, 0.01, 8000);
camera.up.set(0, 0, 1); // Z-up (scan data)

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = false;
controls.rotateSpeed = 0.4;

scene.add(new THREE.AmbientLight(0xffffff, 0.9));

function resize() {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (canvas.width !== w || canvas.height !== h) {
    renderer.setSize(w, h, false);
    camera.aspect = w / Math.max(1, h);
    camera.updateProjectionMatrix();
  }
}
function animate() {
  resize();
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(animate);
}
animate();

// ---- color helpers ----
// divergent blue→white→red around a center value
function divergent(v, center, half) {
  let t = (v - center) / (half || 1e-9); // -1..+1
  t = Math.max(-1, Math.min(1, t));
  if (t < 0) { const a = 1 + t; return [a, a, 1]; }   // blue side
  const a = 1 - t; return [1, a, a];                  // red side
}
// categorical hue from integer
function categorical(i) {
  const h = (i * 0.61803398875) % 1.0; // golden-ratio hashing
  const c = new THREE.Color().setHSL(h, 0.6, 0.6);
  return [c.r, c.g, c.b];
}

// ---- globals built from data ----
let D = null;
let layers = {};     // named THREE.Object3D groups/points
let wallGeom = null; // BufferGeometry for wall points
let wallColorAttr = null;
let cellOfWallPoint = null; // Int32Array

const STEP_COUNT = 6;
let step = 0;

// ============================================================
// build scene from debug data
// ============================================================
function build(d) {
  D = d;
  $("pair").textContent = `${d.dataset} · ${d.reference} → ${d.target}`;

  const cell = d.cell_size_m;
  const axis = new THREE.Vector3(...d.plumb_axis).normalize();

  // ---- wall points ----
  const wxyz = Float32Array.from(d.wall_xyz);
  cellOfWallPoint = Int32Array.from(d.wall_cell);
  const nWall = wxyz.length / 3;
  // 벽면 점 범위 — 카메라 fit 은 전체 씬이 아닌 옹벽에 맞춘다(길고 얇아 멀면 안 보임)
  const wmin = [Infinity, Infinity, Infinity], wmax = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < nWall; i++) {
    for (let a = 0; a < 3; a++) {
      const v = wxyz[i * 3 + a];
      if (v < wmin[a]) wmin[a] = v;
      if (v > wmax[a]) wmax[a] = v;
    }
  }
  D._wallMin = wmin; D._wallMax = wmax;
  wallGeom = new THREE.BufferGeometry();
  wallGeom.setAttribute("position", new THREE.BufferAttribute(wxyz, 3));
  const wcol = new Float32Array(nWall * 3);
  wallColorAttr = new THREE.BufferAttribute(wcol, 3);
  wallGeom.setAttribute("color", wallColorAttr);
  const wallMat = new THREE.PointsMaterial({ size: 2.2, sizeAttenuation: false, vertexColors: true });
  layers.wall = new THREE.Points(wallGeom, wallMat);
  scene.add(layers.wall);

  // ---- other (non-wall) points ----
  const oxyz = Float32Array.from(d.other_xyz);
  const ogeom = new THREE.BufferGeometry();
  ogeom.setAttribute("position", new THREE.BufferAttribute(oxyz, 3));
  const oMat = new THREE.PointsMaterial({ size: 1.6, sizeAttenuation: false, color: 0x4a5260 });
  layers.other = new THREE.Points(ogeom, oMat);
  scene.add(layers.other);

  // ---- voxel boxes + normal arrows (per cell) ----
  layers.boxes = new THREE.Group();
  layers.boxesStatus = new THREE.Group(); // colored by include/exclude
  layers.normals = new THREE.Group();
  scene.add(layers.boxes);
  scene.add(layers.boxesStatus);
  scene.add(layers.normals);

  const unitBox = new THREE.BoxGeometry(cell, cell, cell);
  const unitEdges = new THREE.EdgesGeometry(unitBox);
  const neutralMat = new THREE.LineBasicMaterial({ color: 0x3a4456, transparent: true, opacity: 0.5 });
  const matInc = new THREE.LineBasicMaterial({ color: 0x4ade80 });           // 포함=초록
  const matNonPlanar = new THREE.LineBasicMaterial({ color: 0xf87171 });     // 비평면=빨강
  const matFew = new THREE.LineBasicMaterial({ color: 0x5b6577, transparent: true, opacity: 0.6 }); // 점부족=회색

  const arrowLen = cell * 0.9;
  for (const c of d.cells) {
    const cx = (c.key[0] + 0.5) * cell;
    const cy = (c.key[1] + 0.5) * cell;
    const cz = (c.key[2] + 0.5) * cell;

    // neutral wireframe (step 3: 복셀 분할)
    const ln = new THREE.LineSegments(unitEdges, neutralMat);
    ln.position.set(cx, cy, cz);
    layers.boxes.add(ln);

    // status-colored wireframe (step 5)
    const sMat = c.included ? matInc : (c.reason === "non_planar" ? matNonPlanar : matFew);
    const sln = new THREE.LineSegments(unitEdges, sMat);
    sln.position.set(cx, cy, cz);
    layers.boxesStatus.add(sln);

    // normal arrow (step 4) — only cells that got a plane fit
    if (c.normal && c.centroid) {
      const origin = new THREE.Vector3(...c.centroid);
      const dir = new THREE.Vector3(...c.normal).normalize();
      const col = c.included ? 0x9ad0ff : 0xf8a0a0;
      const arr = new THREE.ArrowHelper(dir, origin, arrowLen, col, arrowLen * 0.32, arrowLen * 0.2);
      layers.normals.add(arr);
    }
  }

  // ---- plumb axis line through wall centroid ----
  layers.plumb = new THREE.Group();
  {
    // centroid of included cells (fallback: bbox center)
    const inc = d.cells.filter((c) => c.included && c.centroid);
    let cen;
    if (inc.length) {
      cen = new THREE.Vector3();
      for (const c of inc) cen.add(new THREE.Vector3(...c.centroid));
      cen.multiplyScalar(1 / inc.length);
    } else {
      cen = new THREE.Vector3(
        (d.bbox_min[0] + d.bbox_max[0]) / 2,
        (d.bbox_min[1] + d.bbox_max[1]) / 2,
        (d.bbox_min[2] + d.bbox_max[2]) / 2
      );
    }
    const hSpan = d.bbox_max[2] - d.bbox_min[2];
    const L = Math.max(8, hSpan * 0.7);
    const p0 = cen.clone().addScaledVector(axis, -L * 0.5);
    const p1 = cen.clone().addScaledVector(axis, L * 0.6);
    const g = new THREE.BufferGeometry().setFromPoints([p0, p1]);
    const m = new THREE.LineBasicMaterial({ color: 0xffd54a, depthTest: false, transparent: true, opacity: 0.95 });
    const ln = new THREE.Line(g, m); ln.renderOrder = 999;
    layers.plumb.add(ln);
    const sg = new THREE.SphereGeometry(Math.max(L * 0.012, 0.05), 16, 12);
    const sph = new THREE.Mesh(sg, new THREE.MeshBasicMaterial({ color: 0xffd54a, depthTest: false }));
    sph.position.copy(cen); sph.renderOrder = 999;
    layers.plumb.add(sph);
  }
  scene.add(layers.plumb);

  // ---- 상/하부 경계 평면 (주황) ----
  // 크기는 '포함된 셀'의 범위 기준 — 전체 벽 bbox 는 이상치 때문에 과대해
  // 평면이 화면을 압도한다.
  layers.split = new THREE.Group();
  if (d.sections && typeof d.sections.split_h_m === "number") {
    const o = new THREE.Vector3(...d.plumb_origin);
    const incCells = d.cells.filter((c) => c.included && c.centroid);
    let imin = new THREE.Vector3(Infinity, Infinity, Infinity);
    let imax = new THREE.Vector3(-Infinity, -Infinity, -Infinity);
    for (const c of incCells) {
      imin.min(new THREE.Vector3(...c.centroid));
      imax.max(new THREE.Vector3(...c.centroid));
    }
    if (!incCells.length) { imin = new THREE.Vector3(...D._wallMin); imax = new THREE.Vector3(...D._wallMax); }
    const cen = imin.clone().add(imax).multiplyScalar(0.5);
    // 평면 위 한 점 p: (p - origin)·axis = split_h
    const hAtCen = cen.clone().sub(o).dot(axis);
    const p = cen.clone().addScaledVector(axis, d.sections.split_h_m - hAtCen);
    const w = Math.max(imax.x - imin.x, 4) * 1.08;
    const dep = Math.max(imax.y - imin.y, 4) * 1.08;
    const g = new THREE.PlaneGeometry(w, dep);
    const mesh = new THREE.Mesh(g, new THREE.MeshBasicMaterial({
      color: 0xffa040, transparent: true, opacity: 0.10, side: THREE.DoubleSide, depthWrite: false,
    }));
    mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), axis);
    mesh.position.copy(p);
    layers.split.add(mesh);
    const edge = new THREE.LineSegments(
      new THREE.EdgesGeometry(g),
      new THREE.LineBasicMaterial({ color: 0xffa040, transparent: true, opacity: 0.9 })
    );
    edge.quaternion.copy(mesh.quaternion);
    edge.position.copy(p);
    layers.split.add(edge);
  }
  scene.add(layers.split);

  fitCamera(D._wallMin, D._wallMax);
  renderSteps();
  goStep(0);
}

function fitCamera(bmin, bmax) {
  const min = new THREE.Vector3(...bmin), max = new THREE.Vector3(...bmax);
  const center = min.clone().add(max).multiplyScalar(0.5);
  const size = max.clone().sub(min);
  const radius = Math.max(size.x, size.y, size.z) * 0.5 + 1;
  controls.target.copy(center);
  // view from an oblique angle, slightly above
  camera.position.set(center.x + radius * 1.15, center.y - radius * 1.15, center.z + radius * 0.85);
  camera.near = radius / 200; camera.far = radius * 50;
  camera.updateProjectionMatrix();
  controls.update();
}

// ============================================================
// wall point coloring per step
// ============================================================
function colorWall(mode) {
  if (!wallGeom) return;
  const col = wallColorAttr.array;
  const n = col.length / 3;
  const cells = D.cells;
  const med = D.median_tilt_deg ?? 0;
  const half = Math.max((D.iqr_deg ?? 1) * 1.5, 1.5); // ± window for divergent
  for (let i = 0; i < n; i++) {
    const ci = cellOfWallPoint[i];
    const c = cells[ci];
    let r = 0.55, g = 0.6, b = 0.7; // neutral
    if (mode === "wall") { r = 0.35; g = 0.55; b = 0.95; }
    else if (mode === "cell") { const t = categorical(ci); r = t[0]; g = t[1]; b = t[2]; }
    else if (mode === "status") {
      if (c.included) { r = 0.30; g = 0.85; b = 0.45; }
      else if (c.reason === "non_planar") { r = 0.97; g = 0.40; b = 0.40; }
      else { r = 0.38; g = 0.42; b = 0.50; }
    } else if (mode === "tilt") {
      if (c.included && typeof c.tilt_deg === "number") {
        const t = divergent(c.tilt_deg, med, half); r = t[0]; g = t[1]; b = t[2];
      } else { r = 0.28; g = 0.30; b = 0.36; } // 제외 셀은 어둡게
    }
    col[i * 3] = r; col[i * 3 + 1] = g; col[i * 3 + 2] = b;
  }
  wallColorAttr.needsUpdate = true;
}

// ============================================================
// steps
// ============================================================
const STEPS = [
  {
    title: "원본 점군",
    sub: "기준(reference) 스캔",
    body: (d) => `
      <h2>1. 기준 스캔 점군</h2>
      <p>절대 기울기는 <b>기준 스캔 한 장의 형상</b>만으로 구합니다(정합·변위와 무관).
      먼저 스캔 전체에서 옹벽 면에 해당하는 점만 골라내야 합니다.</p>
      <div class="nums">
        <span class="k">전체 벽면 점</span><span>${d.wall_point_total.toLocaleString()}</span>
        <span class="k">비벽면 점</span><span>${d.other_point_total.toLocaleString()}</span>
      </div>`,
  },
  {
    title: "벽면 마스킹",
    sub: "노멀로 바닥·천장 제외",
    body: (d) => `
      <h2>2. 벽면 점 선별</h2>
      <p>각 점의 표면 노멀과 연직축(plumb)의 내적을 봅니다. 노멀이 <b>거의 수평</b>인
      점만 벽면으로 채택하고, 노멀이 수직인 바닥·천장(회색)은 버립니다.</p>
      <div class="formula">|n · plumb| &lt; mask_nz (${d.mask_nz})  →  벽면</div>
      <p>파란 점이 남은 벽면 점입니다.</p>`,
  },
  {
    title: "복셀 분할",
    sub: `${0} m 격자`,
    body: (d) => `
      <h2>3. 복셀(voxel) 격자 분할</h2>
      <p>벽면을 한 변 <span class="hl">${d.cell_size_m} m</span>짜리 정육면체 셀로 나눕니다.
      평면도상 휘어진 벽이라도 작은 셀 하나는 <b>국소적으로 평면</b>에 가깝기 때문에,
      단일 전역 평면보다 정확하게 다룰 수 있습니다.</p>
      <div class="nums">
        <span class="k">셀 크기</span><span>${d.cell_size_m} m</span>
        <span class="k">총 셀 수</span><span>${d.n_cells}</span>
        <span class="k">셀당 최소 점</span><span>${d.min_pts}</span>
      </div>`,
  },
  {
    title: "국소 평면 피팅",
    sub: "PCA 노멀 추정",
    body: (d) => `
      <h2>4. 셀별 국소 평면 피팅</h2>
      <p>각 셀의 점들을 <b>PCA(공분산 고유분해)</b>로 분석합니다. 가장 작은 고유값에
      대응하는 고유벡터가 평면의 노멀(화살표)입니다. 노멀은 옹벽 바깥(outward)
      방향으로 부호를 맞춥니다.</p>
      <div class="formula">tilt = arcsin( n · plumb )</div>
      <p>노멀과 연직축의 각도가 곧 그 셀의 수직 대비 기울기입니다.
      이 성분은 <b>수평 방위에 불변</b>이라, 벽이 휘어 방위가 달라지는 셀들도 그대로
      모을 수 있습니다.</p>`,
  },
  {
    title: "평면성 검사",
    sub: "비평면 셀 제외",
    body: (d) => `
      <h2>5. 평면성 검사 — 곡선/변곡부 제외</h2>
      <p>최소 고유값(평면 두께)이 중간 고유값의 ${(d.planar_ratio * 100).toFixed(0)}%를 넘으면
      그 셀은 <b>평평하지 않다</b>고 보고 제외합니다. 사행 변곡부처럼 한 셀에 두 방위가
      섞인 곳이 자동으로 걸러집니다.</p>
      <div class="formula">w₀ &gt; ${d.planar_ratio} · w₁  →  제외(비평면)</div>
      <div class="nums">
        <span class="k"><span style="color:#4ade80">■</span> 포함</span><span>${d.n_cells_included}개</span>
        <span class="k"><span style="color:#f87171">■</span> 비평면 제외</span><span>${d.cells.filter(c=>c.reason==="non_planar").length}개</span>
        <span class="k"><span style="color:#5b6577">■</span> 점부족 제외</span><span>${d.cells.filter(c=>c.reason==="too_few").length}개</span>
      </div>`,
  },
  {
    title: "기울기 집계",
    sub: "셀 median",
    body: (d) => `
      <h2>6. 강건 집계 → 절대 기울기</h2>
      <p>포함된 셀들의 기울기를 <b>중앙값(median)</b>으로 모읍니다. 평균이 아닌 중앙값을
      쓰므로 일부 이상 셀(요철·잡음)에 둔감합니다. 색은 중앙값 기준
      파랑(안쪽)↔빨강(바깥).</p>
      <div class="formula">절대 기울기 = median( 셀별 tilt )</div>
      <div class="nums">
        <span class="k">절대 기울기</span><span class="hl">${d.median_tilt_deg==null?"—":d.median_tilt_deg.toFixed(3)+"°"} ${d.median_tilt_deg==null?"":(d.median_tilt_deg>=0?"(OUTWARD)":"(INWARD)")}</span>
        <span class="k">셀간 IQR</span><span>${d.iqr_deg==null?"—":d.iqr_deg.toFixed(3)+"°"}</span>
        <span class="k">사용 셀</span><span>${d.n_cells_included} / ${d.n_cells}</span>
      </div>
      ${d.sections ? `
      <p style="margin-top:4px"><b>상/하부 분리:</b> 높이별 셀 기울기의 계단 변화(changepoint)를
      탐지해 경계 <span class="hl">h=${d.sections.split_h_m.toFixed(2)} m</span>(주황 평면)에서
      나눴습니다. 하부는 설계상 경사, 상부는 수직 설계이므로 <b>상부 값이 실제 변형 판단 기준</b>입니다.</p>
      <div class="nums">
        <span class="k">상부(수직부)</span><span class="hl">${d.sections.upper.tilt_deg.toFixed(3)}° ${d.sections.upper.tilt_deg>=0?"(OUTWARD)":"(INWARD)"} · 셀 ${d.sections.upper.cell_count}</span>
        <span class="k">하부(설계 경사부)</span><span>${d.sections.lower.tilt_deg.toFixed(3)}° ${d.sections.lower.tilt_deg>=0?"(OUTWARD)":"(INWARD)"} · 셀 ${d.sections.lower.cell_count}</span>
      </div>` : ""}
      <p style="color:var(--text-dim);font-size:12px">노란 선 = 연직 기준(plumb). 이 절대 기울기에
      두 스캔 사이의 <b>변화량(α)</b>을 더하면 최신 시점의 절대 기울기가 됩니다.</p>`,
  },
];

// visibility config per step
function applyStep(s) {
  const show = (name, v) => { if (layers[name]) layers[name].visible = v; };
  show("wall", true);
  show("other", s === 0);            // 비벽면은 1단계에서만
  show("boxes", s === 2);            // 중립 복셀은 분할 단계
  show("boxesStatus", s === 4);      // 상태색 복셀은 평면성 검사
  show("normals", s === 3 || s === 5);
  show("plumb", s === 5);
  show("split", s === 5);            // 상/하부 경계 평면은 집계 단계에서

  if (s === 0) colorWall("neutral");
  else if (s === 1) colorWall("wall");
  else if (s === 2) colorWall("wall");
  else if (s === 3) colorWall("cell");
  else if (s === 4) colorWall("status");
  else if (s === 5) colorWall("tilt");

  updateLegend(s);
}

function updateLegend(s) {
  const el = $("legend");
  if (s === 4) {
    el.className = "legend show";
    el.innerHTML = `<div class="title">셀 분류</div><div class="swatches">
      <div class="sw"><i style="background:#4ade80"></i>포함 (평면)</div>
      <div class="sw"><i style="background:#f87171"></i>비평면 제외</div>
      <div class="sw"><i style="background:#5b6577"></i>점부족 제외</div></div>`;
  } else if (s === 5) {
    const med = D.median_tilt_deg ?? 0;
    const half = Math.max((D.iqr_deg ?? 1) * 1.5, 1.5);
    el.className = "legend show";
    el.innerHTML = `<div class="title">셀 기울기 (수직 대비)</div>
      <div class="bar" style="background:linear-gradient(90deg,#5b9bff,#ffffff,#ff6b6b)"></div>
      <div class="ticks"><span>${(med-half).toFixed(2)}°</span><span>${med.toFixed(2)}°</span><span>${(med+half).toFixed(2)}°</span></div>`;
  } else {
    el.className = "legend";
    el.innerHTML = "";
  }
}

// ============================================================
// step UI
// ============================================================
function renderSteps() {
  const wrap = $("steps");
  wrap.innerHTML = "";
  STEPS.forEach((st, i) => {
    const div = document.createElement("div");
    div.className = "step" + (i === step ? " active" : "");
    div.innerHTML = `<div class="num">${i + 1}</div>
      <div class="lbl">${st.title}<small>${st.sub.replace("0 m", D.cell_size_m + " m")}</small></div>`;
    div.addEventListener("click", () => goStep(i));
    wrap.appendChild(div);
  });
}

function goStep(i) {
  step = Math.max(0, Math.min(STEP_COUNT - 1, i));
  // update step list active state
  [...$("steps").children].forEach((c, idx) => c.classList.toggle("active", idx === step));
  $("narration").innerHTML = STEPS[step].body(D);
  $("prev").disabled = step === 0;
  $("next").disabled = step === STEP_COUNT - 1;
  applyStep(step);
}

$("prev").addEventListener("click", () => goStep(step - 1));
$("next").addEventListener("click", () => goStep(step + 1));
window.addEventListener("keydown", (e) => {
  if (e.key === "ArrowRight" || e.key === "ArrowDown") goStep(step + 1);
  else if (e.key === "ArrowLeft" || e.key === "ArrowUp") goStep(step - 1);
});

// ============================================================
// load
// ============================================================
function showOverlay(msg) {
  const o = $("overlay");
  o.style.display = "flex";
  o.innerHTML = msg;
}

async function load() {
  if (!dataset || !stem) {
    showOverlay("dataset / stem 파라미터가 없습니다.<br>분석 화면의 기울기 패널에서 '계산 과정' 버튼으로 여세요.");
    $("narration").innerHTML = "<p>파라미터 없음.</p>";
    return;
  }
  try {
    const url = `/api/wall-tilt-debug/${encodeURIComponent(dataset)}/${encodeURIComponent(stem)}`;
    const r = await fetch(url);
    if (!r.ok) {
      const txt = await r.text();
      showOverlay(`데이터를 불러오지 못했습니다 (${r.status}).<br><span style="font-size:12px">${txt}</span>`);
      $("narration").innerHTML = "<p style='color:var(--err)'>로드 실패.</p>";
      return;
    }
    const d = await r.json();
    build(d);
  } catch (e) {
    showOverlay("오류: " + e.message);
  }
}

load();
