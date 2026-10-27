"""Root-only, frozen supplied-model train/val transition diagnosis; no training.

Imports at module scope are standard library only. This is a source-specified
DP-demonstration conversion, not a physics-replay or deployment equivalence test.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'results/world_model_supplied_transition_audit_2026_10_20'
OUT = BASE / 'producer'
VISUAL = ROOT / 'results/world_model_visual_transfer_2026_10_19'
QUAL = ROOT / 'scripts/world_model_supplied_lewm_qualification_2026_10_20/qualify_full_model.py'
QUAL_SHA = 'cf77a53c4dcc0108a747898959ce5d56746ccb4be0132d33af88a69f772a233d'
ADAPTER = ROOT / 'scripts/world_model_visual_transfer_2026_10_19/data_adapter.py'
ADAPTER_SHA = '2b44f1f171fc813c3341f71e6d220867a78f4efb62e3644af9d5927f13e5a442'
CODEC = ROOT / 'results/world_model_exploration_2026_10_19/c_dependencies'
ALIGNMENT = ROOT / 'reports/world_model_supplied_transition_audit_2026_10_20/DATA_INTERFACE_READINESS.md'
ALIGNMENT_SHA = 'ed6bd4dec6c76ee76d272f4d35493c58729ef1dc0a16d4d61bbf7ba98ea3bf28'
PARENT = ROOT / 'results/world_model_supplied_lewm_control_2026_10_20/PHASE_CLOSURE.json'
PARENT_SHA = '69ba6f017872e77668c730bbcf97280a4a9263be279a4dda388e9327b4f71cb8'
RESOURCES = {'phase_CPU_s': 40, 'producer_CPU_s': 24, 'producer_wall_s': 120,
    'audit_CPU_s': 8, 'audit_wall_s': 60, 'global_wall_s': 240,
    'artifact_bytes': 134217728, 'memory_bytes': 2147483648,
    'OS_available_bytes': 6442450944, 'disk_reserve_bytes': 10737418240}
ARMS = ('true', 'wrong', 'hold')
CASES = ((1, 1), (1, 3), (3, 1), (3, 3))
EXPECTED_CACHE = {'train': 18221, 'val': 2680}
CACHE_SHA = {'train': 'bdfa73d8fcdae0495b0325ffb0a7e82b204a51528010a02b05c5181a664e6add',
    'val': 'c854f4fee3f47d8e979900c243e51844fa842579b40be4f5947134e6baee38bb'}
CACHE_MANIFEST_SHA = '430bf2bc51f7f73a02787c83e5deca008e43bd52f5ae96fd5c3922fb31dccfde'
BRANCH_SHA = 'b6f4e36b82e48eee721013f28a3b0e0af0cef536f5bf1377d76677e599150b8b'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def event(phase, **extra):
    with (OUT / 'EVENTS.jsonl').open('a') as stream:
        stream.write(json.dumps({'phase': phase, **extra}, sort_keys=True, allow_nan=False) + '\n')
        stream.flush()


def budget(admission, origins):
    assert time.process_time() - origins[0] < 23
    assert time.perf_counter() - origins[1] < 115
    assert time.time() < admission['not_after_unix'] - 10


def admit(path):
    path = Path(path).resolve(); path.relative_to(BASE)
    a = read(path)
    assert a['schema'] == 'SUPPLIED_TRANSITION_AUDIT_ADMISSION_v1' and a['mode'] == 'producer'
    assert a['admitted'] and a['passed'] and a['outer_guard_required']
    assert a['source_sha256'] == sha(__file__) and a['resource_contract'] == RESOURCES
    entry = Path(__file__).with_name('entry.py')
    assert a['entry_source_sha256'] == sha(entry)
    assert 0 < a['not_after_unix'] - a['not_before_unix'] <= 240
    assert a['not_before_unix'] <= time.time() < a['not_after_unix'] - 15
    parent = ROOT / a['parent_closure_path']; assert parent == PARENT
    assert sha(parent) == a['parent_closure_sha256'] == PARENT_SHA
    p = read(parent)
    assert p['phase_complete'] and p['panel_complete'] and p['saved_native_replay_audit_complete']
    assert a['inherited_allocation_CPU_s'] == 2155.83259
    assert sha(QUAL) == QUAL_SHA and sha(ADAPTER) == ADAPTER_SHA
    assert sha(ALIGNMENT) == ALIGNMENT_SHA
    receipt = BASE / 'ENTRY_producer.json'; r = read(receipt)
    assert r['passed'] and r['available_bytes'] >= 6442450944 and r['disk_free_bytes'] >= 10737418240
    assert a['not_before_unix'] <= r['unix'] <= time.time()
    return a, {str(p.relative_to(ROOT)): sha(p) for p in (path, entry, receipt, parent, QUAL, ADAPTER, ALIGNMENT)}


def select_windows(adapter, split):
    """Fixed existing hash-split order; no observations or outcomes select rows."""
    episodes = adapter.split[split][:32] if split == 'train' else adapter.split[split]
    assert len(episodes) == (32 if split == 'train' else 20)
    selected, qualifications = [], []
    for ep in episodes:
        begin, end = 0 if ep == 0 else adapter.ends[ep - 1], adapter.ends[ep]
        length = end - begin
        # Common H3/F3 eligibility. All comparisons use these same windows.
        legal = max(0, length - 25)
        times = sorted(set((10, 10 + (legal - 1) // 2, 10 + legal - 1))) if legal else []
        qualifications.append({'episode': ep, 'length': length, 'legal_windows': legal, 'selected_t': times})
        for t in times:
            assert t >= 10 and t + 15 < length
            selected.append({'episode': ep, 't': t, 'raw_row': begin + t, 'episode_begin': begin})
    assert selected and len({row['episode'] for row in selected}) >= 2
    # A fixed next eligible episode supplies future actions, never the same episode.
    eligible = [ep for ep in episodes if any(r['episode'] == ep for r in selected)]
    by_ep = {ep: [i for i, row in enumerate(selected) if row['episode'] == ep] for ep in eligible}
    donors = []
    for row in selected:
        ep = row['episode']; position = by_ep[ep].index(len(donors))
        donor_ep = eligible[(eligible.index(ep) + 1) % len(eligible)]
        donors.append(by_ep[donor_ep][position % len(by_ep[donor_ep])])
    assert all(selected[d]['episode'] != row['episode'] for row, d in zip(selected, donors))
    return selected, donors, qualifications


def array_id(np, value):
    value = np.ascontiguousarray(value)
    return {'shape': list(value.shape), 'dtype': str(value.dtype),
        'sha256': hashlib.sha256(memoryview(value).cast('B')).hexdigest()}


def prepare(np, adapter, bundle, split):
    rows, donors, qualifications = select_windows(adapter, split)
    branch_path = VISUAL / 'E_CACHE_BRANCH.json'; branch = read(branch_path)
    assert sha(branch_path) == BRANCH_SHA
    assert branch['cache_source_sha256'] == bundle['feature'].digest(bundle['feature'].__file__)
    cache = ROOT / branch['cache_directory']; manifest_path = cache / 'CACHE_MANIFEST.json'
    manifest = read(manifest_path); record = manifest['splits'][split]
    assert sha(manifest_path) == CACHE_MANIFEST_SHA
    path = cache / (split + '.npz')
    assert manifest['passed'] and manifest['frozen_state_identity_unchanged']
    assert record['path'] == str(path.relative_to(ROOT)) and record['frames'] == EXPECTED_CACHE[split]
    assert sha(path) == record['sha256'] == CACHE_SHA[split]
    with np.load(path, allow_pickle=False) as saved:
        assert set(saved.files) == {'z', 'raw_row', 'episode', 't'}
        z, raw, ep, t = (saved[k] for k in ('z', 'raw_row', 'episode', 't'))
    assert z.shape == (EXPECTED_CACHE[split], 192) and z.dtype == np.float32 and np.isfinite(z).all()
    assert all(v.shape == (len(z),) and v.dtype == np.int64 for v in (raw, ep, t))
    expected = [(r, e, i) for e in sorted(adapter.split[split])
        for i, r in enumerate(range(0 if e == 0 else adapter.ends[e - 1], adapter.ends[e]))]
    assert list(zip(raw.tolist(), ep.tolist(), t.tolist())) == expected
    assert hashlib.sha256(raw.tobytes()).hexdigest() == record['raw_row_sha256']
    lookup = {int(r): i for i, r in enumerate(raw)}
    frame_rows = np.asarray([[row['raw_row'] + d for d in (-10, -5, 0, 5, 10, 15)] for row in rows], dtype=np.int64)
    action_rows = np.asarray([[row['raw_row'] + d for d in range(-10, 15)] for row in rows], dtype=np.int64)
    used = sorted(set(action_rows.reshape(-1).tolist()))
    controls = adapter.rows('data/action', used, split, 'supplied_transition_actual_absolute_action_rows')
    state5 = adapter.rows('data/state', used, split, 'supplied_transition_only_pusher_xy_used_for_relative_conversion')
    assert controls.shape == (len(used), 2) and state5.shape == (len(used), 5)
    assert np.isfinite(controls).all() and np.isfinite(state5[:, :2]).all()
    where = {r: i for i, r in enumerate(used)}
    positions = np.asarray([[where[int(r)] for r in item] for item in action_rows], dtype=np.int64)
    absolute, pusher = controls[positions], state5[:, :2][positions]
    # Float64 subtraction preserves source float32 values; final model inputs are FP32.
    u = ((absolute.astype(np.float64) - pusher.astype(np.float64)) / 100.).astype(np.float32)
    features = z[np.asarray([[lookup[int(r)] for r in item] for item in frame_rows], dtype=np.int64)]
    arrays = {'episode': np.asarray([r['episode'] for r in rows], dtype=np.int64),
        't': np.asarray([r['t'] for r in rows], dtype=np.int64),
        'frame_raw_rows': frame_rows, 'action_raw_rows': action_rows,
        'history_E': features[:, :3].copy(), 'target_E': features[:, 3:].copy(),
        'absolute_actions': absolute, 'pusher_xy': pusher, 'relative_u': u,
        'donor_window_index': np.asarray(donors, dtype=np.int64),
        'past_blocks': bundle['qual'].norm(np, u[:, :10]).reshape(len(rows), 2, 10),
        'true_blocks': bundle['qual'].norm(np, u[:, 10:]).reshape(len(rows), 3, 10),
        'wrong_blocks': bundle['qual'].norm(np, u[donors, 10:]).reshape(len(rows), 3, 10),
        'hold_blocks': bundle['qual'].norm(np, np.zeros_like(u[:, 10:])).reshape(len(rows), 3, 10)}
    return arrays, {'episodes_in_existing_split_order': adapter.split[split][:32] if split == 'train' else adapter.split[split],
        'episode_qualification': qualifications, 'windows': len(rows),
        'relative_u_outside_unit_box_coordinate_count': int((np.abs(u) > 1).sum()),
        'relative_u_coordinate_count': int(u.size), 'relative_u_min': u.min(0).min(0).tolist(),
        'relative_u_max': u.max(0).max(0).tolist(),
        'input_bindings': {str(p.relative_to(ROOT)): sha(p) for p in (branch_path, manifest_path, path)}}


def metrics(np, arrays):
    result = {}
    for h, f in CASES:
        truth = arrays['target_E'][:, :f]
        persistence = np.mean((arrays['history_E'][:, -1:, :] - truth) ** 2, axis=-1)
        values = arrays[f'prediction_H{h}_F{f}']
        error = np.mean((values - truth[:, None]) ** 2, axis=-1)
        result[f'H{h}_F{f}'] = {'primary_policy_related_case': (h, f) == (1, 3),
            'primary_metric': 'terminal_at_raw_t_plus_' + str(5 * f) + '_MSE',
            'window_weighted_MSE': {arm: float(error[:, i].mean()) for i, arm in enumerate(ARMS)},
            'window_weighted_MSE_is_all_future_frames_auxiliary': True,
            'persistence_window_weighted_MSE': float(persistence.mean()),
            'terminal_window_weighted_MSE': {arm: float(error[:, i, -1].mean()) for i, arm in enumerate(ARMS)},
            'persistence_terminal_MSE': float(persistence[:, -1].mean()), 'episodes': []}
        result[f'H{h}_F{f}']['per_future_block_window_weighted_MSE'] = {
            arm: error[:, i].mean(0).astype(float).tolist() for i, arm in enumerate(ARMS)}
        result[f'H{h}_F{f}']['persistence_per_future_block_MSE'] = persistence.mean(0).astype(float).tolist()
        for ep in sorted(set(arrays['episode'].tolist())):
            keep = arrays['episode'] == ep
            p = float(persistence[keep].mean()); v = [float(error[keep, i].mean()) for i in range(3)]
            pt = float(persistence[keep, -1].mean()); vt = [float(error[keep, i, -1].mean()) for i in range(3)]
            result[f'H{h}_F{f}']['episodes'].append({'episode': ep, 'windows': int(keep.sum()),
                'MSE': dict(zip(ARMS, v)), 'persistence_MSE': p,
                'true_minus_persistence': v[0] - p, 'true_over_persistence': v[0] / p if p else None,
                'true_minus_wrong': v[0] - v[1], 'true_over_wrong': v[0] / v[1] if v[1] else None,
                'true_minus_hold': v[0] - v[2], 'true_over_hold': v[0] / v[2] if v[2] else None,
                'terminal_MSE': dict(zip(ARMS, vt)), 'persistence_terminal_MSE': pt,
                'terminal_true_minus_persistence': vt[0] - pt,
                'terminal_true_over_persistence': vt[0] / pt if pt else None,
                'terminal_true_minus_wrong': vt[0] - vt[1],
                'terminal_true_over_wrong': vt[0] / vt[1] if vt[1] else None,
                'terminal_true_minus_hold': vt[0] - vt[2],
                'terminal_true_over_hold': vt[0] / vt[2] if vt[2] else None})
    return result


def main(path):
    origins = time.process_time(), time.perf_counter()
    admission, bindings = admit(path)
    OUT.mkdir(parents=True, exist_ok=False)
    result = {'schema': 'SUPPLIED_TRANSITION_AUDIT_v1', 'passed': False, 'prediction_complete': False,
        'training_updates': 0, 'native_calls': 0, 'control_recipe_selection': False,
        'input_bindings': bindings, 'files': {}, 'phases': [], 'splits': {}}
    adapter = bundle = None
    try:
        sys.dont_write_bytecode = True; os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
        qual = load('supplied_transition_qualified_full_model', QUAL)
        event('full_load_started')
        bundle = qual.load_bundle(OUT); bundle['qual'] = qual
        np, torch = bundle['np'], bundle['torch']
        result['input_bindings'].update(bundle['bindings']); result['runtime'] = bundle['runtime']
        event('full_load_returned')
        source = load('supplied_transition_readonly_data_adapter', ADAPTER)
        adapter = source.VisualAdapter(CODEC)
        assert adapter.numcodecs.__version__ == '0.15.1'
        Path(adapter.numcodecs.__file__).resolve().relative_to(CODEC.resolve())
        result['input_bindings'].update(adapter.input_bindings)
        assert adapter.input_bindings['results/world_model_exploration_2026_10_19/c_public/PUBLIC_DATA.zip'] == '63d52a114a3f010861f0181309d165b7d69133ccae426ece2fc94caed147bdf9'
        assert adapter.input_bindings['results/world_model_exploration_2026_10_19/c_public/DATA_MANIFEST.json'] == '5702c13dd48454e3ef8152e2fe2920ff5e16cb3ecd8f40f29b0e37589604263f'
        result['codec_files_sha256'] = {str(p.relative_to(ROOT)): sha(p) for p in sorted(CODEC.rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}
        for split in ('train', 'val'):
            budget(admission, origins)
            phase = {'name': 'prepare_' + split, 'CPU_start': time.process_time(), 'wall_start': time.perf_counter()}
            arrays, info = prepare(np, adapter, bundle, split)
            result['input_bindings'].update(info['input_bindings'])
            n = len(arrays['t']); event('split_prepared', split=split, windows=n)
            for h, f in CASES:
                prediction = np.empty((n, 3, f, 192), dtype=np.float32)
                for begin in range(0, n, 32):
                    budget(admission, origins); end = min(n, begin + 32); count = end - begin
                    hist = arrays['history_E'][begin:end, -h:]
                    initial = np.repeat(hist[:, None], 3, axis=1).reshape(count * 3, h, 192)
                    past = arrays['past_blocks'][begin:end, -(h - 1):] if h > 1 else None
                    arms = []
                    for arm in ARMS:
                        future = arrays[arm + '_blocks'][begin:end, :f]
                        arms.append(np.concatenate((past, future), axis=1) if h > 1 else future)
                    actions = np.stack(arms, axis=1).reshape(count * 3, h - 1 + f, 10)
                    event('predict_started', split=split, H=h, F=f, begin=begin, batch=count * 3)
                    tensor = qual.predict_cached(bundle, torch.from_numpy(initial).to(device='mps', dtype=torch.bfloat16),
                        torch.from_numpy(actions).to('mps'), future=f)
                    torch.mps.synchronize()
                    prediction[begin:end] = tensor[:, h:].float().cpu().numpy().reshape(count, 3, f, 192)
                    assert np.isfinite(prediction[begin:end]).all()
                    del tensor
                    event('predict_returned', split=split, H=h, F=f, begin=begin)
                    qual.memory(torch)
                arrays[f'prediction_H{h}_F{f}'] = prediction
            file = OUT / (split + '.npz')
            with file.open('xb') as stream:
                np.savez(stream, **arrays)
            result['files'][split] = {'path': str(file.relative_to(BASE)), 'sha256': sha(file),
                'arrays': {key: array_id(np, value) for key, value in arrays.items()}}
            result['splits'][split] = {**info, 'metrics': metrics(np, arrays)}
            phase.update(CPU_s=time.process_time() - phase.pop('CPU_start'), wall_s=time.perf_counter() - phase.pop('wall_start'))
            result['phases'].append(phase); event('split_saved', split=split)
        after = bundle['feature'].state_identity(torch, bundle['model'])
        write(OUT / 'STRICT_FULL_STATE_AFTER.json', after)
        assert after == bundle['before']
        result.update(passed=True, prediction_complete=True, full_state_unchanged=True,
            primary_case='H1_F3', diagnostic_cases=['H1_F1', 'H3_F1', 'H3_F3'],
            no_statistical_or_control_advantage_claim=True, action_alignment_source_only_not_physics_replay=True)
    except Exception as exc:
        result['failure'] = {'type': type(exc).__name__, 'stage': 'see_EVENTS',
            'prediction_failure_is_not_negative_prediction_evidence': True}
    finally:
        if adapter is not None:
            result['access_ledger'] = adapter.ledger
            try:
                adapter.close()
            except Exception as exc:
                result.update(passed=False, prediction_complete=False,
                    close_failure={'type': type(exc).__name__})
        result.update(body_CPU_s=time.process_time() - origins[0], body_wall_s=time.perf_counter() - origins[1])
        write(OUT / 'RESULT.json', result)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--admission', required=True)
    raise SystemExit(main(parser.parse_args().admission))
