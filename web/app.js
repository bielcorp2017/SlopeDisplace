import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { PLYLoader } from "three/addons/loaders/PLYLoader.js";

const $ = (id) => document.getElementById(id);

// ---- State ----
const state = {
  dataset: null,
  files: [],
  referenceName: null,
  selectedStem: null,
  mode: "horizontal",  // signed_normal | magnitude | horizontal | vertical | rgb
  clamp: 0.005,           // half-range (e.g. ±0.005 m for signed; 0..0.005 for magnitude)
  pointSize: 3.2,
  // Loaded cloud:
  geometry: null,
  disp: null,             // Float32Array (N x 4)
  rgb: null,              // Uint8Array (N x 3), original scan colors
  rgbStem: null,          // stem for which rgb is currently loaded
  meta: null,
  pointCount: 0,
  refStem: null,
};

// ---- Three.js scene ----
const canvas = $("canvas");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0b0d12);

const camera = new THREE.PerspectiveCamera(50, 1, 0.01, 5000);
camera.position.set(2, 2, 2);
camera.up.set(0, 0, 1);  // Z-up; common for scan data

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = false;
controls.zoomSpeed = 1.0;       // OrbitControls 의 기본 zoom 은 사용 안 함 (아래에서 직접 처리)
controls.rotateSpeed = 0.3;
controls.enableZoom = false;    // 휠 줌은 직접 처리 (디바이스별 deltaY 변동 무시)

// 휠 이벤트마다 고정 배수로 줌 — 트랙패드/마우스 무관하게 일정한 한 번당 줌량
const ZOOM_FACTOR = 1.10;       // 한 번당 10%
canvas.addEventListener("wheel", (e) => {
  e.preventDefault();
  const sign = e.deltaY > 0 ? 1 : -1; // wheel down(deltaY>0) = zoom out
  const factor = (sign > 0) ? ZOOM_FACTOR : (1 / ZOOM_FACTOR);
  const dir = new THREE.Vector3().subVectors(camera.position, controls.target);
  let dist = dir.length() * factor;
  dist = Math.max(controls.minDistance, Math.min(controls.maxDistance, dist));
  dir.setLength(dist);
  camera.position.copy(controls.target).add(dir);
  controls.update();
}, { passive: false });

scene.add(new THREE.AmbientLight(0xffffff, 0.6));

// ---- Axes (50m, positioned at cloud center after loading) ----
const axesLen = 50;
let axesGroup = null;

function updateAxes(center) {
  if (axesGroup) scene.remove(axesGroup);
  axesGroup = new THREE.Group();
  const o = center || new THREE.Vector3(0, 0, 0);
  const axMat = (c) => new THREE.LineBasicMaterial({ color: c });
  const axLine = (pts, c) => { const g = new THREE.BufferGeometry().setFromPoints(pts); return new THREE.Line(g, axMat(c)); };
  axesGroup.add(axLine([o.clone(), new THREE.Vector3(o.x + axesLen, o.y, o.z)], 0xff0000)); // X red
  axesGroup.add(axLine([o.clone(), new THREE.Vector3(o.x, o.y + axesLen, o.z)], 0x00ff00)); // Y green
  axesGroup.add(axLine([o.clone(), new THREE.Vector3(o.x, o.y, o.z + axesLen)], 0x0000ff)); // Z blue
  scene.add(axesGroup);
}

let pointsObj = null;
let pointsMaterial = null;

// ---- Point picking (raycaster) ----
const raycaster = new THREE.Raycaster();
const _mouse = new THREE.Vector2();
let _pickedMarker = null;
let _pointerDownPos = null;

canvas.addEventListener("pointerdown", (e) => {
  _pointerDownPos = { x: e.clientX, y: e.clientY };
});

canvas.addEventListener("pointerup", (e) => {
  if (!_pointerDownPos) return;
  const dx = e.clientX - _pointerDownPos.x;
  const dy = e.clientY - _pointerDownPos.y;
  _pointerDownPos = null;
  // Ignore drags (orbit/pan)
  if (dx * dx + dy * dy > 9) return;

  if (!pointsObj || !state.disp) return;
  const rect = canvas.getBoundingClientRect();
  _mouse.x =  ((e.clientX - rect.left) / rect.width)  * 2 - 1;
  _mouse.y = -((e.clientY - rect.top)  / rect.height) * 2 + 1;

  // Compute threshold from camera distance to target
  const camDist = camera.position.distanceTo(controls.target);
  raycaster.params.Points.threshold = camDist * 0.005;
  raycaster.setFromCamera(_mouse, camera);
  const hits = raycaster.intersectObject(pointsObj);
  const tooltip = $("point-tooltip");

  if (!hits.length) {
    tooltip.style.display = "none";
    if (_pickedMarker) { scene.remove(_pickedMarker); _pickedMarker = null; }
    return;
  }

  const hit = hits[0];
  const idx = hit.index;
  const d = state.disp;
  const sn = d[idx * 4 + 0];
  const mg = d[idx * 4 + 1];
  const hz = d[idx * 4 + 2];
  const vt = d[idx * 4 + 3];
  const pos = hit.point;

  const fmt = (v) => (v >= 0 ? "+" : "") + (v * 1000).toFixed(2) + " mm";
  tooltip.innerHTML =
    `<b>Point #${idx.toLocaleString()}</b><br>` +
    `법선: ${fmt(sn)}<br>` +
    `크기: ${(mg * 1000).toFixed(2)} mm<br>` +
    `수평: ${fmt(hz)}<br>` +
    `수직: ${fmt(vt)}<br>` +
    `<span style="color:#888">좌표: (${pos.x.toFixed(3)}, ${pos.y.toFixed(3)}, ${pos.z.toFixed(3)})</span>`;
  tooltip.style.display = "block";
  tooltip.style.left = (e.clientX - canvas.getBoundingClientRect().left + 12) + "px";
  tooltip.style.top  = (e.clientY - canvas.getBoundingClientRect().top  - 10) + "px";

  // Highlight marker
  if (_pickedMarker) scene.remove(_pickedMarker);
  const markerGeo = new THREE.SphereGeometry(raycaster.params.Points.threshold * 2, 8, 8);
  const markerMat = new THREE.MeshBasicMaterial({ color: 0xffff00, transparent: true, opacity: 0.6 });
  _pickedMarker = new THREE.Mesh(markerGeo, markerMat);
  _pickedMarker.position.copy(pos);
  scene.add(_pickedMarker);
});

function resize() {
  const parent = canvas.parentElement;
  const w = Math.max(1, parent.clientWidth);
  const h = Math.max(1, parent.clientHeight);
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(document.querySelector(".right"));
window.addEventListener("resize", resize);
requestAnimationFrame(resize);  // initial size after first layout

function animate() {
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(animate);
}
animate();

// ---- Colormaps (mirrored from matplotlib) ----
// Divergent: cool-blue -> white -> warm-red (RdBu_r-like, simplified linear blend).
function colorSigned(t, out, off) {
  // t in [-1, 1]
  const a = Math.min(1, Math.max(-1, t));
  if (a >= 0) {
    // 0 -> white (1,1,1), 1 -> red (0.7, 0.05, 0.1)
    out[off]   = 1.0 - a * (1.0 - 0.70);
    out[off+1] = 1.0 - a * (1.0 - 0.05);
    out[off+2] = 1.0 - a * (1.0 - 0.10);
  } else {
    // 0 -> white, -1 -> blue (0.07, 0.30, 0.85)
    const v = -a;
    out[off]   = 1.0 - v * (1.0 - 0.07);
    out[off+1] = 1.0 - v * (1.0 - 0.30);
    out[off+2] = 1.0 - v * (1.0 - 0.85);
  }
}

// Sequential JET-like (blue -> cyan -> yellow -> red)
function colorJet(t, out, off) {
  const a = Math.min(1, Math.max(0, t));
  // Piecewise linear approximation of jet
  let r, g, b;
  if (a < 0.125)      { r = 0;                  g = 0;                  b = 0.5 + 4*a; }
  else if (a < 0.375) { r = 0;                  g = 4*(a-0.125);        b = 1; }
  else if (a < 0.625) { r = 4*(a-0.375);        g = 1;                  b = 1 - 4*(a-0.375); }
  else if (a < 0.875) { r = 1;                  g = 1 - 4*(a-0.625);    b = 0; }
  else                { r = 1 - 4*(a-0.875)*0.5; g = 0;                  b = 0; }
  out[off] = r; out[off+1] = g; out[off+2] = b;
}

// Color reference (no displacement data): solid gray.
function colorReference(n) {
  const c = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) { c[i*3] = 0.55; c[i*3+1] = 0.58; c[i*3+2] = 0.62; }
  return c;
}

function recolor() {
  if (!state.geometry) return;
  const n = state.pointCount;
  const colors = state.geometry.getAttribute("color");
  const arr = colors.array;

  if (state.mode === "rgb") {
    if (!state.rgb || state.rgbStem !== state.selectedStem) {
      // Load lazily, then recolor.
      const stemAtRequest = state.selectedStem;
      loadRgb(stemAtRequest).then(() => {
        if (state.selectedStem === stemAtRequest) recolor();
      }).catch(e => {
        if (state.selectedStem === stemAtRequest) logJob("RGB 로드 실패: " + e.message, "err");
      });
      // While loading, keep current colors but switch legend.
      updateLegend("rgb");
      return;
    }
    const rgb = state.rgb;
    const inv = 1 / 255;
    for (let i = 0; i < n; i++) {
      arr[i*3]   = rgb[i*3]   * inv;
      arr[i*3+1] = rgb[i*3+1] * inv;
      arr[i*3+2] = rgb[i*3+2] * inv;
    }
    colors.needsUpdate = true;
    updateLegend("rgb");
    updateStats();
    return;
  }

  if (!state.disp) {
    // Reference cloud: solid color.
    for (let i = 0; i < n; i++) { arr[i*3]=0.55; arr[i*3+1]=0.58; arr[i*3+2]=0.62; }
    colors.needsUpdate = true;
    updateLegend("reference");
    updateStats();
    return;
  }

  const channel = { signed_normal: 0, magnitude: 1, horizontal: 2, vertical: 3 }[state.mode];
  const signed = state.mode !== "magnitude";
  const clamp = state.clamp;
  const disp = state.disp;

  for (let i = 0; i < n; i++) {
    const v = disp[i * 4 + channel];
    if (signed) {
      colorSigned(v / clamp, arr, i * 3);
    } else {
      colorJet(v / clamp, arr, i * 3);
    }
  }
  colors.needsUpdate = true;
  updateLegend("disp");
  updateStats();
}

async function loadRgb(stem) {
  const url = `/api/rgb/${encodeURIComponent(state.dataset)}/${encodeURIComponent(stem)}`;
  logJob(`원본 RGB 로딩...`);
  const buf = await fetch(url).then(r => {
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return r.arrayBuffer();
  });
  state.rgb = new Uint8Array(buf);
  state.rgbStem = stem;
  if (state.rgb.length !== state.pointCount * 3) {
    logJob(`경고: rgb 크기 불일치 (${state.rgb.length} vs ${state.pointCount*3})`, "warn");
  } else {
    logJob(`원본 RGB 로드 완료`, "ok");
  }
}

function updateLegend(kind) {
  const title = $("legend-title");
  const bar = $("legend-bar");
  const ticks = $("legend-ticks");
  const clampBar = document.querySelector(".clamp");
  if (clampBar) clampBar.style.display = (state.mode === "rgb") ? "none" : "";
  if (kind === "reference") {
    title.textContent = "기준 점군 (변위 없음)";
    bar.style.background = "linear-gradient(90deg, #8c919b, #8c919b)";
    ticks.innerHTML = "<span></span><span>reference</span><span></span>";
    return;
  }
  if (kind === "rgb") {
    title.textContent = "원본 스캔 RGB";
    bar.style.background =
      "linear-gradient(90deg, rgb(15,15,15), rgb(180,140,90), rgb(245,235,220))";
    ticks.innerHTML = "<span>scan</span><span>true color</span><span></span>";
    return;
  }
  const c = state.clamp;
  const label = {
    signed_normal: "법선 변위 (m)",
    magnitude:     "최소거리 (m)",
    horizontal:    "수평 변위 (m)",
    vertical:      "수직 변위 (m)",
  }[state.mode];
  title.textContent = label;
  if (state.mode === "magnitude") {
    // JET 0..clamp
    bar.style.background =
      "linear-gradient(90deg, #000080 0%, #0000ff 12%, #00ffff 38%, #ffff00 62%, #ff0000 88%, #800000 100%)";
    ticks.innerHTML = `<span>0</span><span>${(c/2).toFixed(3)}</span><span>${c.toFixed(3)}</span>`;
  } else {
    // RdBu_r -clamp..+clamp
    bar.style.background =
      "linear-gradient(90deg, rgb(18,77,217) 0%, white 50%, rgb(179,13,26) 100%)";
    ticks.innerHTML = `<span>-${c.toFixed(3)}</span><span>0</span><span>+${c.toFixed(3)}</span>`;
  }
}

function updateStats() {
  const el = $("stats");
  if (!state.meta || state.meta.is_reference || !state.meta.displacement_stats) {
    el.style.display = "none";
    return;
  }
  const channelKey = { signed_normal: "signed_normal", magnitude: "magnitude",
                       horizontal: "horizontal", vertical: "vertical" }[state.mode];
  if (!channelKey) {
    el.style.display = "none";
    return;
  }
  el.style.display = "block";
  const s = state.meta.displacement_stats[channelKey];
  const m = state.meta;
  const fit = (m.icp_history && m.icp_history.length) ? m.icp_history[m.icp_history.length-1] : null;
  const rows = [
    ["min",      s.min.toFixed(4) + " m"],
    ["max",      s.max.toFixed(4) + " m"],
    ["mean",     s.mean.toFixed(4) + " m"],
    ["abs mean", s.abs_mean.toFixed(4) + " m"],
    ["rms",      s.rms.toFixed(4) + " m"],
    ["p95 |·|",  s.p95_abs.toFixed(4) + " m"],
    ["points",   m.simple_count.toLocaleString()],
    ["ICP fit",  fit ? `${(fit.fitness*100).toFixed(1)}% / rmse ${fit.inlier_rmse.toFixed(4)}` : "-"],
  ];
  $("stats-tbl").innerHTML =
    rows.map(([k,v]) => `<tr><td class=k>${k}</td><td>${v}</td></tr>`).join("");
}

// ---- File list / dataset ----
async function loadDatasets() {
  const r = await fetch("/api/datasets").then(r => r.json());
  const sel = $("dataset");
  sel.innerHTML = "";
  for (const name of r.datasets) {
    const opt = document.createElement("option");
    opt.value = name; opt.textContent = name;
    sel.appendChild(opt);
  }
  if (r.datasets.length) {
    state.dataset = r.datasets[0];
    sel.value = state.dataset;
    await loadFiles();
  }
}

async function loadFiles() {
  const tbody = $("files");
  tbody.innerHTML = "";
  const r = await fetch(`/api/files?dataset=${encodeURIComponent(state.dataset)}`).then(r=>r.json());
  state.files = r.files;
  state.referenceName = r.reference;
  if (!r.files.length) { $("empty").style.display = "block"; return; }
  $("empty").style.display = "none";

  for (const f of r.files) {
    const tr = document.createElement("tr");
    tr.className = "row" + (f.name === r.reference ? " ref" : "");
    tr.dataset.stem = f.stem;

    const status = f.has_disp ? "정합완료" :
                   (f.has_simple ? "단순화됨" :
                    (f.name === r.reference ? "기준(원본)" : "원본만"));
    const cls = f.has_disp ? "ok" : (f.has_simple || f.name === r.reference ? "warn" : "");

    tr.innerHTML = `
      <td>${f.name}</td>
      <td class="status">${f.size_mb} MB</td>
      <td class="status ${cls}">${status}</td>
      <td></td>`;

    // Show preprocess button for files without _disp (or for reference without _simple).
    const needsPre = (!f.has_disp && f.name !== r.reference) ||
                     (f.name === r.reference && !f.has_simple);
    if (needsPre) {
      const btn = document.createElement("button");
      btn.className = "pre";
      btn.textContent = "전처리";
      btn.onclick = (e) => { e.stopPropagation(); runPreprocess(f.name, btn); };
      tr.lastElementChild.appendChild(btn);
    }
    tr.onclick = () => selectFile(f);
    tbody.appendChild(tr);
  }

  // Auto-select reference (original) file first, then fall back to others.
  // 가장 최근 (파일명 정렬 최후) 의 변위 데이터 보유 스캔을 우선 선택,
  // 없으면 최근의 simple 보유 스캔, 없으면 reference, 그래도 없으면 첫 파일
  const reversed = r.files.slice().reverse();
  const auto =
    reversed.find(f => f.name !== r.reference && f.has_disp) ||
    reversed.find(f => f.name !== r.reference && f.has_simple) ||
    r.files.find(f => f.name === r.reference && f.has_simple) ||
    r.files[0];
  if (auto && (auto.has_disp || auto.has_simple)) selectFile(auto);
}

function selectFile(f) {
  state.selectedStem = f.stem;
  for (const row of document.querySelectorAll("tr.row")) {
    row.classList.toggle("selected", row.dataset.stem === f.stem);
  }
  if (f.has_disp) {
    loadCloud(f.stem, true);
  } else if (f.has_simple) {
    loadCloud(f.stem, false);
  } else {
    logJob(`'${f.name}'은 전처리가 필요합니다 — 옆 "전처리" 버튼을 누르세요.`, "warn");
  }
  // tilt 패널은 항상 최신 스캔 기반으로 유지 — 파일 선택과 무관하게 갱신만 (또는 그대로)
  const tiltPanel = document.getElementById("tilt-panel");
  if (tiltPanel && tiltPanel.classList.contains("open")) {
    const latest = getLatestStemWithDisp();
    if (latest && (!tiltLastData || tiltLastData.target !== latest)) {
      loadTilt(latest);
    }
  }
}

// ---- Cloud loading ----
async function loadCloud(stem, hasDisp) {
  logJob(`로딩: ${stem}_simple.ply ${hasDisp ? "+ disp" : ""}`);
  const plyUrl  = `/api/ply/${encodeURIComponent(state.dataset)}/${encodeURIComponent(stem)}`;
  const metaUrl = `/api/meta/${encodeURIComponent(state.dataset)}/${encodeURIComponent(stem)}`;
  const dispUrl = `/api/disp/${encodeURIComponent(state.dataset)}/${encodeURIComponent(stem)}`;

  const [plyBuf, metaResp] = await Promise.all([
    fetch(plyUrl).then(r => r.arrayBuffer()),
    fetch(metaUrl).then(r => r.ok ? r.json() : null),
  ]);

  const loader = new PLYLoader();
  let geometry;
  try {
    geometry = loader.parse(plyBuf);
  } catch (e) {
    logJob(`PLY 파싱 실패: ${e.message}`, "err");
    return;
  }
  const pos = geometry.getAttribute("position");
  const n = pos.count;

  let disp = null;
  if (hasDisp) {
    const dispBuf = await fetch(dispUrl).then(r => r.arrayBuffer());
    disp = new Float32Array(dispBuf);
    if (disp.length !== n * 4) {
      logJob(`경고: disp 크기 불일치 (${disp.length} vs ${n*4})`, "warn");
    }
  }

  // Color attribute.
  const colors = new Float32Array(n * 3);
  geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));

  // Replace existing points.
  if (pointsObj) { scene.remove(pointsObj); pointsObj.geometry.dispose(); }
  pointsMaterial = new THREE.PointsMaterial({
    size: state.pointSize, vertexColors: true, sizeAttenuation: false,
  });
  pointsObj = new THREE.Points(geometry, pointsMaterial);
  scene.add(pointsObj);

  state.geometry = geometry;
  state.disp = disp;
  state.rgb = null;        // reset; lazy-loaded if user picks rgb mode
  state.rgbStem = null;
  state.meta = metaResp;
  state.pointCount = n;
  state.refStem = metaResp && metaResp.reference ? metaResp.reference.replace(/\.pts$/, "") : null;

  // Fit camera to bounds.
  geometry.computeBoundingBox();
  geometry.computeBoundingSphere();
  fitCameraTo(geometry.boundingSphere);
  updateAxes(geometry.boundingBox.min.clone());

  recolor();
  logJob(`로딩 완료: ${n.toLocaleString()} pts`, "ok");

  // Warn if registration was unreliable
  if (metaResp && metaResp.registration_reliable === false) {
    const fit = (metaResp.final_icp_fitness * 100).toFixed(1);
    const angle = metaResp.fgr_rotation_deg?.toFixed(1) ?? "?";
    logJob(`⚠ 정합 품질 낮음 (fitness ${fit}%, FGR 회전 ${angle}°) — 변위 결과가 부정확할 수 있습니다. 재전처리를 권장합니다.`, "warn");
  }
}

function fitCameraTo(sphere) {
  if (!sphere) return;
  const c = sphere.center.clone();
  const r = Math.max(sphere.radius, 0.5);

  // fov 에 맞춰 점군 전체가 화면에 들어가는 거리 계산 (+20% 여유)
  const fov = camera.fov * Math.PI / 180;
  const d = (r / Math.tan(fov / 2)) * 1.2;

  // 카메라를 -X 방향 d 만큼 떨어뜨려 점군 중심 바라보기
  camera.position.set(c.x - d, c.y, c.z);
  controls.target.copy(c);

  // near 는 작게(점 안까지 가도 안 잘리게), far 는 충분히 크게
  camera.near = Math.max(d * 0.001, 0.01);
  camera.far  = d * 200;
  camera.updateProjectionMatrix();

  // 줌 범위 제한 — 너무 가까이 들어가거나 너무 멀어지지 않게
  // 이전 r×0.05 / r×30 은 너무 넓어 limit 에 닿으면 점군이 사라지는 듯 보였음
  controls.minDistance = r * 0.2;   // 옹벽 표면 가까이까지만
  controls.maxDistance = r * 6;     // 점군이 항상 화면에 의미있게 보이는 거리
  controls.update();
}

// ---- Preprocess job ----
async function runPreprocess(filename, btn) {
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span>';
  logJob(`전처리 시작: ${filename}`);
  try {
    const r = await fetch(
      `/api/preprocess?dataset=${encodeURIComponent(state.dataset)}&filename=${encodeURIComponent(filename)}`,
      { method: "POST" }
    ).then(r => r.json());
    if (!r.job_id) throw new Error("no job_id");
    await pollJob(r.job_id);
  } catch (e) {
    logJob(`전처리 실패: ${e.message}`, "err");
  } finally {
    btn.disabled = false;
    btn.textContent = "전처리";
    await loadFiles();  // refresh status / buttons
  }
}

async function pollJob(jid) {
  let lastMsg = "";
  while (true) {
    const j = await fetch(`/api/job/${jid}`).then(r => r.json());
    const msg = `${j.stage}${j.detail ? " — " + j.detail : ""}`;
    if (msg !== lastMsg) {
      logJob(`  ${msg}`);
      lastMsg = msg;
    }
    if (j.status === "done") {
      const dt = (j.finished - j.started).toFixed(1);
      logJob(`완료 (${dt}s)`, "ok");
      return j;
    }
    if (j.status === "error") {
      logJob(`오류: ${j.error}`, "err");
      throw new Error(j.error);
    }
    await new Promise(r => setTimeout(r, 500));
  }
}

// ---- Job log ----
function logJob(msg, kind="") {
  const div = $("joblog");
  const line = document.createElement("div");
  line.className = "line " + kind;
  const ts = new Date().toLocaleTimeString();
  line.innerHTML = `<span class="ts">${ts}</span>${msg}`;
  div.appendChild(line);
  div.scrollTop = div.scrollHeight;
  while (div.children.length > 200) div.removeChild(div.firstChild);
}

// ---- Tilt analysis panel ----
let tiltChart = null;
let tiltLinesGroup = null;
let tiltLastData = null;
let tiltExagg = 1000;
let tiltLinesVisible = true;

// 가장 최근 스캔(파일명 정렬 최후) 중 변위 데이터가 있는 것을 선택.
// reference 스캔은 자기 자신과 비교가 없으므로 제외.
function getLatestStemWithDisp() {
  if (!state.files || state.files.length === 0) return null;
  for (let i = state.files.length - 1; i >= 0; i--) {
    const f = state.files[i];
    if (f.name === state.referenceName) continue;
    if (f.has_disp) return f.stem;
  }
  return null;
}

function clearTiltLines() {
  if (tiltLinesGroup) {
    scene.remove(tiltLinesGroup);
    tiltLinesGroup.traverse(o => {
      if (o.geometry) o.geometry.dispose();
      if (o.material) o.material.dispose();
    });
    tiltLinesGroup = null;
  }
}

function buildTiltLines(t, exagg) {
  clearTiltLines();
  if (!t || !t.wall_centroid_xyz || !t.wall_facing_xy) return;
  if (!tiltLinesVisible) return;

  const c = new THREE.Vector3(...t.wall_centroid_xyz);
  const plumb = new THREE.Vector3(...t.plumb_axis).normalize();
  const facing = new THREE.Vector3(...t.wall_facing_xy);
  facing.z = 0;
  if (facing.lengthSq() < 1e-6) facing.set(1, 0, 0); else facing.normalize();
  const alpha = t.alpha_mm_per_m;

  // rotation axis (perpendicular to plumb and facing) - around which we tilt
  const rotAxis = new THREE.Vector3().crossVectors(plumb, facing).normalize();
  const effTilt = (alpha / 1000) * exagg;        // radians, exaggerated
  const lean = plumb.clone().applyAxisAngle(rotAxis, effTilt).normalize();

  const hRange = t.height_range_m;
  const wallH = (hRange ? (hRange[1] - hRange[0]) : 10);
  const L = Math.max(20, wallH * 1.4);

  const group = new THREE.Group();

  // 1) Plumb reference line (yellow)
  {
    const p0 = c.clone().addScaledVector(plumb, -L * 0.5);
    const p1 = c.clone().addScaledVector(plumb, L * 0.6);
    const g = new THREE.BufferGeometry().setFromPoints([p0, p1]);
    const m = new THREE.LineBasicMaterial({ color: 0xffd54a, depthTest: true });
    group.add(new THREE.Line(g, m));
  }
  // 2) Wall lean line (red) — exaggerated
  {
    const p0 = c.clone().addScaledVector(lean, -L * 0.5);
    const p1 = c.clone().addScaledVector(lean, L * 0.6);
    const g = new THREE.BufferGeometry().setFromPoints([p0, p1]);
    const m = new THREE.LineBasicMaterial({ color: 0xff5050, depthTest: true });
    group.add(new THREE.Line(g, m));
  }
  // 3) Centroid marker (small yellow sphere)
  {
    const r = Math.max(L * 0.012, 0.05);
    const sg = new THREE.SphereGeometry(r, 20, 14);
    const sm = new THREE.MeshBasicMaterial({ color: 0xffd54a });
    const sph = new THREE.Mesh(sg, sm);
    sph.position.copy(c);
    group.add(sph);
  }
  // 4) Top markers — endpoints near top so user sees divergence
  {
    const r = Math.max(L * 0.008, 0.03);
    const plumbTop = c.clone().addScaledVector(plumb, L * 0.6);
    const leanTop = c.clone().addScaledVector(lean, L * 0.6);
    const sg = new THREE.SphereGeometry(r, 16, 10);
    const m1 = new THREE.MeshBasicMaterial({ color: 0xffd54a });
    const m2 = new THREE.MeshBasicMaterial({ color: 0xff5050 });
    const a = new THREE.Mesh(sg, m1); a.position.copy(plumbTop); group.add(a);
    const b = new THREE.Mesh(sg.clone(), m2); b.position.copy(leanTop); group.add(b);
  }

  scene.add(group);
  tiltLinesGroup = group;
}

function fmt(n, d = 4, sign = false) {
  if (n === null || n === undefined || !isFinite(n)) return "—";
  const s = n.toFixed(d);
  return (sign && n >= 0 ? "+" : "") + s;
}

function renderTilt(t) {
  const body = $("tilt-body");
  tiltLastData = t;
  if (!t) {
    body.innerHTML = '<div class="empty-msg">변위 데이터가 있는 스캔을 선택하세요.</div>';
    if (tiltChart) { tiltChart.destroy(); tiltChart = null; }
    clearTiltLines();
    return;
  }
  const isNoChange = !t.significant;
  const dirClass = isNoChange ? "nochange" : (t.tilt_direction === "OUTWARD" ? "outward" : "inward");
  const arrow = isNoChange ? "≈" : (t.tilt_direction === "OUTWARD" ? "→ OUT" : "← IN");

  // registration 품질 경고
  let warnHtml = "";
  const fit = t.icp_fitness;
  const reliable = t.registration_reliable;
  const sigma = t.residual_robust_sigma_mm;
  if (fit !== null && fit !== undefined && fit < 0.30) {
    warnHtml += `<div class="tilt-warn">⚠ <b>정합 품질 낮음</b><br>
      ICP fitness ${(fit*100).toFixed(1)}% (권장 ≥30%) — 두 점군이 미세하게 안 맞춰져
      회귀 결과가 옹벽 변화가 아닌 정합 오차일 수 있음. 재전처리 권장.</div>`;
  } else if (reliable === false) {
    warnHtml += `<div class="tilt-warn">⚠ 정합 메타가 'unreliable' 로 마킹됨.</div>`;
  }
  if (sigma > 30) {
    warnHtml += `<div class="tilt-warn">⚠ <b>잔차 σ=${sigma.toFixed(0)} mm</b> 가 큼.
      스캐너 잡음(2-5mm) 대비 과도 — 옹벽 외 큰 장면 변화 가능성.</div>`;
  }

  body.innerHTML = warnHtml + `
    <div class="tilt-card">
      <div class="pair" style="display:flex;align-items:center;gap:6px">
        <span style="background:rgba(78,161,255,.2);color:var(--accent);padding:1px 6px;border-radius:3px;font-size:10px;font-weight:600">최신</span>
        <b>${t.reference}</b> → <b>${t.target}</b>
      </div>
      <div class="tilt-headline ${dirClass}">${fmt(t.tilt_change_deg, 4, true)}°  ${arrow}</div>
      <div class="tilt-ci">95% CI [${fmt(t.tilt_change_ci95_deg[0], 4, true)}, ${fmt(t.tilt_change_ci95_deg[1], 4, true)}]°</div>
      <div class="tilt-verdict ${isNoChange ? "nosig" : "sig"}">${t.verdict}</div>
      <div class="tilt-metrics">
        <span class="k">α</span><span>${fmt(t.alpha_mm_per_m, 3, true)} mm/m</span>
        <span class="k">α 95% CI</span><span>[${fmt(t.alpha_ci95_mm_per_m[0], 3, true)}, ${fmt(t.alpha_ci95_mm_per_m[1], 3, true)}]</span>
        <span class="k">β (h_ref)</span><span>${fmt(t.beta_mm, 2, true)} mm</span>
        <span class="k">잔차 σ</span><span>${fmt(t.residual_robust_sigma_mm, 2)} mm</span>
        <span class="k">옹벽 점수</span><span>${t.wall_face_point_count.toLocaleString()} / ${t.total_point_count.toLocaleString()}</span>
        <span class="k">높이 범위</span><span>${fmt(t.height_range_m[0], 2)} ~ ${fmt(t.height_range_m[1], 2)} m</span>
        <span class="k">R²</span><span>${fmt(t.r_squared, 4)}</span>
        <span class="k">ICP fitness</span><span>${fit !== null && fit !== undefined ? (fit*100).toFixed(1)+"%" : "—"}</span>
      </div>
    </div>
    <div class="tilt-chart-wrap"><canvas id="tilt-chart"></canvas></div>
    <div class="tilt-3d">
      <div class="tilt-3d-row">
        <label>3D 라인</label>
        <input type="checkbox" id="tilt-lines-vis" ${tiltLinesVisible ? "checked" : ""} />
        <span style="flex:1"></span>
        <span class="swatch" style="background:#ffd54a"></span><span style="font-size:10px;color:#98a0ad">plumb</span>
        <span class="swatch" style="background:#ff5050;margin-left:6px"></span><span style="font-size:10px;color:#98a0ad">lean</span>
      </div>
      <div class="tilt-3d-row">
        <label>과장</label>
        <input type="range" id="tilt-exagg" min="0" max="5000" step="50" value="${tiltExagg}" />
        <span class="val" id="tilt-exagg-val">×${tiltExagg}</span>
      </div>
      <div class="tilt-3d-legend">실제 ${fmt(t.tilt_change_deg, 4, true)}° → 화면 ${fmt(t.tilt_change_deg * tiltExagg, 2, true)}°</div>
    </div>
    <div class="tilt-actions">
      <button id="tilt-recompute">재계산</button>
    </div>
  `;

  // chart
  const ctx = $("tilt-chart").getContext("2d");
  if (tiltChart) tiltChart.destroy();
  const h = t.scatter_h_m, d = t.scatter_d_mm;
  const scatterData = h.map((hi, i) => ({ x: hi, y: d[i] }));
  // regression line endpoints
  const h_ref = t.h_ref_m, alpha = t.alpha_mm_per_m, beta = t.beta_mm;
  const h0 = t.height_range_m[0], h1 = t.height_range_m[1];
  const lineData = [
    { x: h0, y: alpha * (h0 - h_ref) + beta },
    { x: h1, y: alpha * (h1 - h_ref) + beta },
  ];
  tiltChart = new Chart(ctx, {
    type: "scatter",
    data: {
      datasets: [
        {
          label: "wall point",
          data: scatterData,
          backgroundColor: "rgba(78,161,255,0.35)",
          pointRadius: 1.2,
          pointHoverRadius: 3,
        },
        {
          label: `α=${fmt(alpha, 3, true)} mm/m`,
          data: lineData,
          type: "line",
          borderColor: "rgba(248,113,113,0.9)",
          borderWidth: 2,
          pointRadius: 0,
          fill: false,
          tension: 0,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      parsing: false,
      plugins: {
        legend: { display: true, position: "top",
                  labels: { color: "#98a0ad", font: { size: 10 }, boxWidth: 14 } },
        tooltip: {
          callbacks: {
            label: (ctx) => `h=${ctx.parsed.x.toFixed(2)}m, d=${ctx.parsed.y.toFixed(2)}mm`,
          },
        },
      },
      scales: {
        x: { type: "linear",
             title: { display: true, text: "h along plumb (m)", color: "#98a0ad", font: { size: 10 } },
             ticks: { color: "#98a0ad", font: { size: 10 } },
             grid: { color: "rgba(255,255,255,0.06)" } },
        y: { type: "linear",
             title: { display: true, text: "signed disp (mm) — out +", color: "#98a0ad", font: { size: 10 } },
             ticks: { color: "#98a0ad", font: { size: 10 } },
             grid: { color: "rgba(255,255,255,0.06)" } },
      },
    },
  });

  $("tilt-recompute").addEventListener("click", async () => {
    if (!state.dataset) return;
    const latest = getLatestStemWithDisp();
    if (!latest) return;
    await loadTilt(latest, /*recompute=*/true);
  });

  // 3D lines controls
  $("tilt-lines-vis").addEventListener("change", (e) => {
    tiltLinesVisible = e.target.checked;
    if (tiltLinesVisible) buildTiltLines(tiltLastData, tiltExagg);
    else clearTiltLines();
  });
  $("tilt-exagg").addEventListener("input", (e) => {
    tiltExagg = parseInt(e.target.value, 10);
    $("tilt-exagg-val").textContent = "×" + tiltExagg;
    const legend = document.querySelector(".tilt-3d-legend");
    if (legend && tiltLastData) {
      legend.textContent = `실제 ${fmt(tiltLastData.tilt_change_deg, 4, true)}° → 화면 ${fmt(tiltLastData.tilt_change_deg * tiltExagg, 2, true)}°`;
    }
    buildTiltLines(tiltLastData, tiltExagg);
  });

  // 3D 라인 즉시 그리기
  buildTiltLines(t, tiltExagg);
}

async function loadTilt(stem, recompute = false) {
  if (!stem) { renderTilt(null); return; }
  const url = `/api/wall-tilt/${encodeURIComponent(state.dataset)}/${encodeURIComponent(stem)}` +
              (recompute ? "?recompute=true" : "");
  try {
    renderTilt(null);
    const body = $("tilt-body");
    body.innerHTML = '<div class="empty-msg"><span class="spinner"></span> 계산 중...</div>';
    const r = await fetch(url);
    if (r.status === 404) {
      body.innerHTML = '<div class="empty-msg">plumb.json 또는 변위 데이터가 없습니다.</div>';
      return;
    }
    if (!r.ok) {
      const msg = await r.text();
      body.innerHTML = `<div class="empty-msg" style="color:var(--err)">에러: ${msg}</div>`;
      return;
    }
    const t = await r.json();
    renderTilt(t);
  } catch (e) {
    $("tilt-body").innerHTML = `<div class="empty-msg" style="color:var(--err)">실패: ${e.message}</div>`;
  }
}

$("tilt-toggle").addEventListener("click", () => {
  const panel = $("tilt-panel");
  const btn = $("tilt-toggle");
  const opened = panel.classList.toggle("open");
  btn.classList.toggle("active", opened);
  if (opened) {
    // 패널은 항상 "최신 스캔" 의 기울기를 보여준다 (3D 뷰의 선택 파일과 무관)
    const latest = getLatestStemWithDisp();
    if (latest) loadTilt(latest);
    else renderTilt(null);
  } else {
    // 패널이 닫히면 3D 라인도 제거
    clearTiltLines();
  }
});

// ---- UI events ----
$("dataset").addEventListener("change", (e) => {
  state.dataset = e.target.value;
  loadFiles();
});
$("refresh").addEventListener("click", () => {
  logJob("파일 목록 새로고침...");
  loadFiles();
});
document.querySelectorAll(".modes button").forEach(b => {
  b.addEventListener("click", () => {
    for (const x of document.querySelectorAll(".modes button")) x.classList.remove("active");
    b.classList.add("active");
    state.mode = b.dataset.mode;
    recolor();
  });
});
$("clamp").addEventListener("input", (e) => {
  state.clamp = parseFloat(e.target.value);
  $("clamp-val").textContent =
    (state.mode === "magnitude" ? " " : "±") +
    state.clamp.toFixed(3) + " m";
  recolor();
});
$("psize").addEventListener("input", (e) => {
  state.pointSize = parseFloat(e.target.value);
  if (pointsMaterial) pointsMaterial.size = state.pointSize;
});
$("cam-reset").addEventListener("click", () => {
  if (state.geometry && state.geometry.boundingSphere) {
    fitCameraTo(state.geometry.boundingSphere);
  }
});

// ---- Boot ----
loadDatasets().catch(e => logJob("초기화 실패: " + e.message, "err"));
