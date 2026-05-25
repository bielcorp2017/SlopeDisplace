"""특정 스캔을 Python pipeline 으로 강제 재처리 (C++ 우회용 진단).

기존 *_simple.ply / *_disp.bin / *_meta.json 을 덮어쓴다.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from server import pipeline


def progress(stage: str, detail: str = ""):
    print(f"[{stage}] {detail}", flush=True)


if __name__ == "__main__":
    dataset = sys.argv[1]
    target = sys.argv[2]
    result = pipeline.preprocess(dataset, target, progress=progress)
    print("---DONE---")
    print(f"is_reference         : {result.get('is_reference')}")
    print(f"final_icp_fitness    : {result.get('final_icp_fitness')}")
    print(f"registration_reliable: {result.get('registration_reliable')}")
    print(f"fgr_rotation_deg     : {result.get('fgr_rotation_deg')}")
    print(f"center_distance      : {result.get('center_distance')}")
    print(f"total_seconds        : {result.get('total_seconds')}")
