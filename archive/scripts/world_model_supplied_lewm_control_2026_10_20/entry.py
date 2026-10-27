"""Fresh resource check for one admitted, externally guarded control/audit."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import sys
import time
import psutil

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'results/world_model_supplied_lewm_control_2026_10_20'

if __name__ == '__main__':
    path, mode = Path(sys.argv[1]).resolve(), sys.argv[2]
    path.relative_to(BASE)
    a = json.loads(path.read_text())
    assert mode in ('control', 'audit') and a['mode'] == mode
    assert a['schema'] == 'SUPPLIED_LEWM_CONTROL_ADMISSION_v1'
    assert a['passed'] and a['admitted'] and a['outer_guard_required']
    assert a['not_before_unix'] <= time.time() < a['not_after_unix'] - 5
    assert a['entry_source_sha256'] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    available, disk = psutil.virtual_memory().available, shutil.disk_usage(ROOT).free
    row = {'unix': time.time(), 'available_bytes': available,
           'required_available_bytes': 6442450944, 'disk_free_bytes': disk,
           'required_disk_free_bytes': 10737418240,
           'passed': available >= 6442450944 and disk >= 10737418240}
    with (BASE / ('ENTRY_' + mode + '.json')).open('x') as stream:
        json.dump(row, stream, sort_keys=True, indent=2); stream.write('\n')
    assert row['passed']
    os.execv(sys.argv[3], sys.argv[3:])
