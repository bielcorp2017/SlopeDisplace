# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SlopeDisplace is a 3D point cloud analysis and visualization system for monitoring slope displacement in retaining walls. Python backend processes large .pts scan files (multi-GB) through registration and displacement computation; a Three.js frontend renders the results interactively.

## Running the Application

```bash
# Install dependencies
pip install fastapi uvicorn open3d numpy scipy pandas

# Start development server
python -m uvicorn server.app:app --reload --host 127.0.0.1 --port 8000

# Open http://localhost:8000/
```

There is no build step, test suite, or linter configured.

## Architecture

**Backend (`server/`):**
- `app.py` — FastAPI server. REST endpoints serve datasets, PLY/binary files, metadata, wall tilt analysis, and manage async preprocessing jobs via an in-memory job tracker.
- `pipeline.py` — Core processing pipeline. Loads PLY files, voxel-downsamples to ~500k points, runs FGR then multi-scale ICP (6 scales: 1.0→0.01m) with overlap cropping on full clouds, computes 4-channel displacement field (signed_normal, magnitude, horizontal, vertical). Supports scan groups (`group` parameter) for independent file sets like `*_all.ply`.
- `wall_tilt.py` — Wall tilt analysis. Detects wall face via normal filtering, fits local patch planes, computes absolute tilt (vs plumb) and tilt change between scans using robust linear regression.
- `inject_synthetic.py` — Dev utility to create synthetic displacement data without running full preprocessing.

**Frontend (`web/`):**
- `index.html` — Single-page app with embedded CSS. Left panel (file list, group selector, job log), right panel (3D viewport with tilt panel).
- `app.js` — Three.js visualization. Loads binary PLY + displacement `.bin` files, renders point clouds with displacement colormaps (divergent for signed channels, jet for magnitude). Controls: clamp range (±5–500mm), point size, mode switching, lazy RGB loading, wall tilt analysis with 3D line overlay.
- `explain.html` / `explain.js` — Step-by-step 3D visualization of the wall tilt computation process (voxel planes, regression fitting).

**Data (`data/`):**
- Each dataset folder (e.g., `CH2_RETAINWALL/`) contains PLY raw scans plus derived files: `*_simple.ply` (downsampled), `*_disp.bin` (N×4 float32 displacement), `*_rgb.bin` (N×3 uint8 colors), `*_meta.json` (transform, ICP history, stats).
- Scan groups: `YYYYMMDD.ply` = default group, `YYYYMMDD_all.ply` = "all" group. Each group has its own reference (oldest file).
- `DATA_ROOT` defaults to `<project>/data`, overridable via `SLOPE_DATA_ROOT` env var.

## Processing Pipeline (pipeline.py)

For a target scan against a reference:
1. Load & voxel-downsample both scans to ~500k points → save `_simple.ply`
2. FGR on simplified clouds → initial transform T0
3. Overlap crop: mutual NN search at coarse resolution to find the shared region, crop both clouds to this bbox before ICP (prevents non-overlapping geometry from misleading ICP)
4. Multi-scale ICP on **cropped full** clouds (using T0 as init) → final transform T
5. Apply T to simplified target, compute per-point displacement vs reference using surface normals and nearest-neighbor (scipy cKDTree)
6. Save `_disp.bin`, `_meta.json`, optionally `_rgb.bin`

## Key API Endpoints

| Endpoint | Purpose |
|----------|---------|
| `GET /api/datasets` | List dataset folders |
| `GET /api/groups?dataset=X` | List scan groups (default, "all", etc.) |
| `GET /api/files?dataset=X&group=G` | List PLY files with preprocessing status |
| `GET /api/ply/{dataset}/{stem}` | Stream simplified PLY binary |
| `GET /api/disp/{dataset}/{stem}` | Stream displacement binary (N×4 float32) |
| `GET /api/rgb/{dataset}/{stem}` | Stream RGB binary (lazy backfill) |
| `POST /api/preprocess?dataset=X&filename=F&group=G` | Start async preprocessing job |
| `GET /api/job/{jid}` | Poll job status |
| `GET /api/wall-tilt/{dataset}/{stem}` | Wall tilt analysis (on-demand computation) |
| `GET /api/wall-tilt-debug/{dataset}/{stem}` | Tilt debug data (voxel planes) |

## Key Dependencies

Python 3.12, FastAPI, Open3D (registration/ICP), NumPy, SciPy (cKDTree), Pandas (.pts parsing). Frontend uses Three.js + PLYLoader + Chart.js from CDN.
