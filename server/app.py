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
def list_files(dataset: str):
    rows = pipeline.list_scan_files(dataset)
    if not rows:
        return {"dataset": dataset, "files": [], "reference": None}
    # The first (oldest by filename) is the reference.
    return {"dataset": dataset, "files": rows, "reference": rows[0]["name"]}


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
    return FileResponse(p, media_type="application/octet-stream")


@app.get("/api/disp/{dataset}/{stem}")
def get_disp(dataset: str, stem: str):
    p = DATA_ROOT / dataset / f"{stem}_disp.bin"
    if not p.is_file():
        raise HTTPException(404, f"disp not found: {stem}")
    return FileResponse(p, media_type="application/octet-stream")


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
    return FileResponse(rgb_path, media_type="application/octet-stream")


@app.post("/api/preprocess")
def start_preprocess(dataset: str, filename: str):
    folder = DATA_ROOT / dataset
    if not (folder / filename).is_file():
        raise HTTPException(404, f"file not found: {filename}")

    jid = _new_job()

    def progress(stage: str, detail: str = ""):
        _update_job(jid, stage=stage, detail=detail)

    def worker_cpp():
        """Run C++ preprocessor as subprocess, parse JSON-line progress."""
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
            else:
                _update_job(jid, status="done", stage="done", detail="",
                            finished=time.time())
        except Exception as e:
            _update_job(jid, status="error", stage="error",
                        detail=str(e), finished=time.time(), error=str(e))

    def worker_python():
        """Fallback: run Python pipeline."""
        try:
            result = pipeline.preprocess(dataset, filename, progress=progress)
            _update_job(
                jid,
                status="done",
                stage="done",
                detail="",
                finished=time.time(),
                result=result,
            )
        except Exception as e:  # noqa: BLE001
            _update_job(
                jid,
                status="error",
                stage="error",
                detail=str(e),
                finished=time.time(),
                error=str(e),
            )

    worker = worker_cpp if _CPP_EXE else worker_python
    threading.Thread(target=worker, daemon=True).start()
    return {"job_id": jid}


@app.get("/api/job/{jid}")
def job_status(jid: str):
    with _jobs_lock:
        job = _jobs.get(jid)
    if not job:
        raise HTTPException(404, "job not found")
    return job


# Static frontend
app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
