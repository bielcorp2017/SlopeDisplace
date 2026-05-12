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
  mode: "rgb",  // signed_normal | magnitude | horizontal | vertical | rgb
  clamp: 0.05,            // half-range (e.g. ±0.05 m for signed; 0..0.05 for magnitude)
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
controls.enableDamping = true;
controls.dampingFactor = 0.08;

scene.add(new THREE.AmbientLight(0xffffff, 0.6));

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
  const auto = r.files.find(f => f.name === r.reference && f.has_simple) || r.files.find(f => f.has_disp) || r.files.find(f => f.has_simple) || r.files[0];
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

  recolor();
  logJob(`로딩 완료: ${n.toLocaleString()} pts`, "ok");
}

function fitCameraTo(sphere) {
  if (!sphere) return;
  const c = sphere.center, r = Math.max(sphere.radius, 0.5);
  const dir = new THREE.Vector3(-1, 0, 0.6).normalize();
  camera.position.copy(c).addScaledVector(dir, r * 0.2);
  controls.target.copy(c);
  camera.near = Math.max(r / 1000, 0.001);
  camera.far  = r * 100;
  camera.updateProjectionMatrix();
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

// ---- Boot ----
loadDatasets().catch(e => logJob("초기화 실패: " + e.message, "err"));
