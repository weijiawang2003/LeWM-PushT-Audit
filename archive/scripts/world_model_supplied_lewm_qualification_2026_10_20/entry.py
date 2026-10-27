"""Fresh OS entry checks; numerical work starts only in the guarded child."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "results/world_model_supplied_lewm_qualification_2026_10_20"

if __name__ == "__main__":
    admission_path, mode = Path(sys.argv[1]).resolve(), sys.argv[2]
    admission_path.relative_to(BASE)
    a = json.loads(admission_path.read_text())
    assert mode in ("probe", "audit") and a["mode"] == mode
    assert a["passed"] and a["admitted"] and a["outer_guard_required"]
    assert a["not_before_unix"] <= time.time() < a["not_after_unix"] - 5
    assert a["entry_source_sha256"] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    available = psutil.virtual_memory().available
    disk_free = shutil.disk_usage(ROOT).free
    row = {"unix": time.time(), "available_bytes": available,
           "required_available_bytes": 6 * 1024**3,
           "disk_free_bytes": disk_free, "required_disk_free_bytes": 10 * 1024**3,
           "passed": available >= 6 * 1024**3 and disk_free >= 10 * 1024**3}
    with (BASE / ("ENTRY_" + mode + ".json")).open("x") as stream:
        json.dump(row, stream, indent=2, sort_keys=True)
        stream.write("\n")
    assert row["passed"]
    os.execv(sys.argv[3], sys.argv[3:])
