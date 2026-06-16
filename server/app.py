"""FastAPI server for slope displacement viewer.

Run from the project root with:
    python -m uvicorn server.app:app --reload --port 8000

Browse to http://localhost:8000/  (serves web/index.html).
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import pipeline
from . import wall_tilt as wt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = PROJECT_ROOT / "web"
DATA_ROOT = pipeline.DATA_ROOT

# C++ preprocessor: look for slope_preprocess executable
_CPP_EXE = os.environ.get("SLOPE_CPP_EXE", "")
if not _CPP_EXE:
    # Auto-detect common build locations
    for candidate in [
        PROJECT_ROOT / "cpp" / "build" / "Release" / "slope_preprocess.exe",
        PROJECT_ROOT / "cpp" / "build" / "slope_preprocess.exe",
        PROJECT_ROOT / "cpp" / "build" / "Debug" / "slope_preprocess.exe",
        PROJECT_ROOT / "cpp" / "build" / "slope_preprocess",
    ]:
        if candidate.is_file():
            _CPP_EXE = str(candidate)
            break

app = FastAPI(title="Slope Displacement Viewer")

# In-memory job tracker for preprocess runs.
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


def _new_job() -> str:
    jid = uuid.uuid4().hex[:12]
    with _jobs_lock:
        _jobs[jid] = {
            "id": jid,
            "status": "running",
            "stage": "queued",
            "detail": "",
            "started": time.time(),
            "finished": None,
            "error": None,
            "result": None,
        }
    return jid


def _update_job(jid: str, **kw):
    with _jobs_lock:
        if jid in _jobs:
            _jobs[jid].update(kw)


@app.get("/api/datasets")
def list_datasets():
    if not DATA_ROOT.is_dir():
        return {"datasets": []}
    names = sorted([p.name for p in DATA_ROOT.iterdir() if p.is_dir()])
    return {"datasets": names}


@app.get("/api/files")
def list_files(dataset: str, group: str = ""):
    rows = pipeline.list_scan_files(dataset, group=group)
    if not rows:
        return {"dataset": dataset, "group": group, "files": [], "reference": None}
    # The first (oldest by filename) is the reference.
    return {"dataset": dataset, "group": group, "files": rows, "reference": rows[0]["name"]}


@app.get("/api/groups")
def list_groups(dataset: str):
    groups = pipeline.list_groups(dataset)
    return {"dataset": dataset, "groups": groups}


@app.get("/api/meta/{dataset}/{stem}")
def get_meta(dataset: str, stem: str):
    p = DATA_ROOT / dataset / f"{stem}_meta.json"
    if not p.is_file():
        raise HTTPException(404, f"meta not found: {stem}")
    return JSONResponse(json.loads(p.read_text()))


@app.get("/api/ply/{dataset}/{stem}")
def get_ply(dataset: str, stem: str):
    p = DATA_ROOT / dataset / f"{stem}_simple.ply"
    if not p.is_file():
        raise HTTPException(404, f"ply not found: {stem}")
    # no-cache: browser must revalidate (ETag/Last-Modified) before reusing, so a
    # re-preprocessed cloud is never served stale from the browser disk cache.
    return FileResponse(p, media_type="application/octet-stream",
                        headers={"Cache-Control": "no-cache"})


@app.get("/api/disp/{dataset}/{stem}")
def get_disp(dataset: str, stem: str):
    p = DATA_ROOT / dataset / f"{stem}_disp.bin"
    if not p.is_file():
        raise HTTPException(404, f"disp not found: {stem}")
    return FileResponse(p, media_type="application/octet-stream",
                        headers={"Cache-Control": "no-cache"})


@app.get("/api/rgb/{dataset}/{stem}")
def get_rgb(dataset: str, stem: str):
    """Original per-point RGB for the simple cloud. Generated lazily from the
    source .ply via nearest-neighbor lookup if the cache doesn't exist."""
    folder = DATA_ROOT / dataset
    rgb_path = folder / f"{stem}_rgb.bin"
    if not rgb_path.exists():
        try:
            pipeline.extract_rgb_for_simple(dataset, stem)
        except FileNotFoundError as e:
            raise HTTPException(404, f"source not found: {e}")
    return FileResponse(rgb_path, media_type="application/octet-stream",
                        headers={"Cache-Control": "no-cache"})


@app.post("/api/preprocess")
def start_preprocess(dataset: str, filename: str, group: str = ""):
    folder = DATA_ROOT / dataset
    if not (folder / filename).is_file():
        raise HTTPException(404, f"file not found: {filename}")

    jid = _new_job()

    def progress(stage: str, detail: str = ""):
        _update_job(jid, stage=stage, detail=detail)

    MIN_FITNESS = 0.30  # 같은 임계값을 pipeline 들과 공유

    def _check_meta_quality() -> tuple[bool, dict | None]:
        """전처리 결과 meta.json 을 읽어 정합 품질을 평가.
        반환 (reliable, meta_dict_or_None)."""
        stem = Path(filename).stem
        meta_path = folder / f"{stem}_meta.json"
        if not meta_path.is_file():
            return (False, None)
        try:
            m = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            return (False, None)
        if m.get("is_reference"):
            return (True, m)
        # registration_reliable 필드 우선
        if m.get("registration_reliable") is True:
            return (True, m)
        if m.get("registration_reliable") is False:
            return (False, m)
        # 둘다 없으면 fitness 로 추론
        fit = m.get("final_icp_fitness")
        if fit is None and m.get("icp_history"):
            fit = m["icp_history"][-1].get("fitness")
        if fit is None:
            return (False, m)
        return (fit >= MIN_FITNESS, m)

    def worker_python(use_existing_jid: bool = True):
        """Python pipeline 으로 전처리 (fallback 으로도 호출됨)."""
        try:
            result = pipeline.preprocess(dataset, filename, group=group,
                                         progress=progress)
            _update_job(
                jid, status="done", stage="done", detail="",
                finished=time.time(), result=result,
            )
        except Exception as e:  # noqa: BLE001
            _update_job(
                jid, status="error", stage="error",
                detail=str(e), finished=time.time(), error=str(e),
            )

    def worker_cpp():
        """Run C++ preprocessor. 끝나면 정합 품질 체크 → fitness 낮으면 Python 으로 재시도."""
        try:
            proc = subprocess.Popen(
                [_CPP_EXE, "--dataset", dataset, "--target", filename,
                 "--data-root", str(DATA_ROOT)],
                stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, text=True,
            )
            for line in proc.stderr:
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                    _update_job(jid, stage=msg.get("stage", ""),
                                detail=msg.get("detail", ""))
                except json.JSONDecodeError:
                    _update_job(jid, detail=line)
            rc = proc.wait()
            if rc != 0:
                _update_job(jid, status="error", stage="error",
                            detail=f"exit code {rc}", finished=time.time(),
                            error=f"C++ preprocessor exited with code {rc}")
                return

            # C++ 성공 → 품질 체크
            reliable, m = _check_meta_quality()
            if reliable:
                _update_job(jid, status="done", stage="done", detail="",
                            finished=time.time(), result=m)
                return

            # 불량 → Python fallback
            fit = (m or {}).get("final_icp_fitness")
            fit_pct = f"{fit*100:.1f}%" if fit is not None else "?"
            _update_job(jid, stage="cpp_unreliable",
                        detail=f"C++ ICP fitness {fit_pct} < {MIN_FITNESS*100:.0f}% — Python pipeline 으로 재시도")
            worker_python(use_existing_jid=True)
        except Exception as e:
            _update_job(jid, status="error", stage="error",
                        detail=str(e), finished=time.time(), error=str(e))

    worker = worker_cpp if (_CPP_EXE and not group) else worker_python
    threading.Thread(target=worker, daemon=True).start()
    return {"job_id": jid}


@app.get("/api/plumb/{dataset}")
def get_plumb(dataset: str):
    p = DATA_ROOT / dataset / "plumb.json"
    if not p.is_file():
        raise HTTPException(404, f"plumb.json not found for dataset: {dataset}")
    return JSONResponse(json.loads(p.read_text(encoding="utf-8")))


@app.get("/api/wall-tilt/{dataset}/{stem}")
def get_wall_tilt(dataset: str, stem: str, recompute: bool = False):
    """Return wall_tilt_<stem>.json — computes on demand if missing or recompute=true."""
    folder = DATA_ROOT / dataset
    cache = folder / f"wall_tilt_{stem}.json"
    if cache.is_file() and not recompute:
        return JSONResponse(json.loads(cache.read_text(encoding="utf-8")))
    # compute now
    try:
        result = wt.compute_wall_tilt(dataset, stem, data_root=DATA_ROOT)
    except FileNotFoundError as e:
        raise HTTPException(404, f"prerequisite missing: {e}")
    except ValueError as e:
        raise HTTPException(400, str(e))
    wt.save_result(result, cache)
    return JSONResponse(result)


@app.get("/api/wall-tilt-debug/{dataset}/{stem}")
def get_wall_tilt_debug(dataset: str, stem: str, recompute: bool = False):
    """계산 과정 시각화용 복셀/평면 디버그 데이터 — 없으면 즉석 계산."""
    folder = DATA_ROOT / dataset
    cache = folder / f"wall_tilt_debug_{stem}.json"
    if cache.is_file() and not recompute:
        return JSONResponse(json.loads(cache.read_text(encoding="utf-8")))
    try:
        result = wt.compute_wall_tilt_debug(dataset, stem, data_root=DATA_ROOT)
    except FileNotFoundError as e:
        raise HTTPException(404, f"prerequisite missing: {e}")
    except ValueError as e:
        raise HTTPException(400, str(e))
    cache.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    return JSONResponse(result)


@app.get("/api/job/{jid}")
def job_status(jid: str):
    with _jobs_lock:
        job = _jobs.get(jid)
    if not job:
        raise HTTPException(404, "job not found")
    return job


# Static frontend
app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
