"""Post hoc description of saved outputs only; no model, training or simulator."""
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'results/world_model_supplied_lewm_control_2026_10_20'

def read(p):
    return json.loads(p.read_text())

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()

def main():
    a = read(BASE / 'ADMISSION_DIAGNOSTIC.json')
    assert a['source_sha256'] == sha(Path(__file__))
    assert a['saved_outputs_only'] and a['shared_CPU_reservation_s'] == 4
    assert a['not_before_unix'] <= time.time() < a['not_after_unix'] - 5
    assert a['not_after_unix'] == read(BASE / 'ADMISSION_CONTROL.json')['not_after_unix']
    audit_path = BASE / 'saved_audit/AUDIT.json'
    assert sha(audit_path) == a['control_audit_sha256']
    audit = read(audit_path)
    assert audit['passed'] and audit['audit_complete'] and audit['panel_complete']
    import numpy as np
    assert np.__version__ == '2.5.3'
    control = BASE / 'control'
    panel = read(control / 'RESULT.json')
    rows = []
    for ep in panel['episodes']:
        p = control / ep['trajectory']['path']
        assert sha(p) == ep['trajectory']['sha256']
        with np.load(p, allow_pickle=False) as f:
            states, u, coverage = f['state11'], f['issued_u'], f['native_coverage']
        rows.append({'seed': ep['seed'], 'raw_controls': len(u),
            'exact_zero_commands': int((u == 0).all(-1).sum()),
            'maximum_command_L2': float(np.linalg.norm(u, axis=-1).max()),
            'maximum_pusher_displacement_from_initial': float(np.linalg.norm(states[:, :2] - states[0, :2], axis=-1).max()),
            'maximum_block_center_displacement_from_initial': float(np.linalg.norm(states[:, 4:6] - states[0, 4:6], axis=-1).max()),
            'maximum_block_sin_cos_change_from_initial': float(np.abs(states[:, 8:10] - states[0, 8:10]).max()),
            'maximum_coverage': float(coverage.max())})
    ids, ranges, rgb = [], [], [set() for _ in range(8)]
    for index in range(panel['policy_group_calls_returned']):
        path = control / 'policy_queries' / f'QUERY_{index}.npz'
        assert sha(path) == read(path.with_suffix('.json'))['arrays_sha256']
        with np.load(path, allow_pickle=False) as f:
            ids.extend(f['selected_global_id'].tolist())
            for slot, image in enumerate(f['current_RGB']):
                rgb[slot].add(hashlib.sha256(image.tobytes()).hexdigest())
            for i in range(3):
                costs = f[f'i{i}_cost']
                ranges.extend((costs.max(-1) - costs.min(-1)).tolist())
    histogram = {str(i): ids.count(i) for i in sorted(set(ids))}
    result = {'schema': 'SUPPLIED_LEWM_SAVED_FAILURE_DESCRIPTION_v1', 'passed': True,
        'post_hoc_descriptive_only': True, 'new_model_or_native_or_training_calls': 0,
        'not_a_causal_failure_mechanism_identification': True,
        'episodes': rows, 'selected_global_id_histogram': histogram,
        'selected_hold_slot_count': sum(i % 64 == 0 for i in ids),
        'total_episode_decisions': len(ids), 'unique_query_RGB_images_per_slot': [len(x) for x in rgb],
        'candidate_cost_range_min': min(ranges), 'candidate_cost_range_max': max(ranges),
        'zero_candidate_cost_range_slot_iterations': sum(x == 0 for x in ranges),
        'candidate_cost_slot_iterations': len(ranges), 'source_sha256': sha(Path(__file__))}
    with (BASE / 'SAVED_FAILURE_DESCRIPTION.json').open('x') as f:
        json.dump(result, f, indent=2, sort_keys=True, allow_nan=False); f.write('\n')

if __name__ == '__main__':
    main()
