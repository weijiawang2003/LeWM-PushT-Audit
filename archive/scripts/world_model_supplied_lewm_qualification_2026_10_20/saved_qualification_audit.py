"""Root-only saved-array/JSON audit; no Torch, model or native calls."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'results/world_model_supplied_lewm_qualification_2026_10_20'
PRODUCER = Path(__file__).with_name('qualify_full_model.py')
PRODUCER_SHA = 'cf77a53c4dcc0108a747898959ce5d56746ccb4be0132d33af88a69f772a233d'
PARENT = ROOT / 'results/world_model_external_action_storage_map_2026_10_20/PHASE_CLOSURE.json'
PARENT_SHA = '387c08db9190b39f27665a6ab50d76e842200c638e1080f6570ef07ff98e294a'
SOURCE_CLOSURE = ROOT / 'results/world_model_external_action_normalization_readiness_2026_10_20/PHASE_CLOSURE.json'
SOURCE_CLOSURE_SHA = 'cd41262c933f6c874509da8328c5d7f23e6ddda9949b84d0af53fdc310de61ac'
MU, SIGMA = (-.0087, .0068), (.2019, .2002)
RTOL, ATOL = 1 / 64, 1 / 128
MEMORY = 2147483648
RESOURCES = {'phase_CPU_s': 24, 'probe_CPU_s': 16, 'probe_wall_s': 180,
             'audit_CPU_s': 4, 'audit_wall_s': 60, 'global_wall_s': 300,
             'artifact_bytes': 134217728, 'memory_bytes': MEMORY,
             'OS_available_bytes': 6442450944, 'disk_reserve_bytes': 10737418240}


def bound(name, directory=ROOT):
    path = Path(name)
    if not path.is_absolute():
        path = ROOT / path
    assert not path.is_symlink()
    path = path.resolve()
    path.relative_to(directory.resolve())
    assert path.is_file()
    return path


def digest(path):
    value = hashlib.sha256()
    with bound(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1048576), b''):
            value.update(part)
    return value.hexdigest()


def load(path):
    return json.loads(bound(path).read_text())


def close(a, b, tolerance=1e-8):
    assert math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, rel_tol=0, abs_tol=tolerance)


def memory_check(row):
    assert row['proxy_is_not_measured_unified_memory'] is True
    assert type(row['peak_RSS_bytes']) is int and type(row['MPS_driver_bytes']) is int
    assert 0 <= row['peak_RSS_bytes'] and 0 <= row['MPS_driver_bytes'] < MEMORY
    assert row['sampled_overlapping_stop_proxy_bytes'] == row['peak_RSS_bytes'] + row['MPS_driver_bytes'] < MEMORY - 64 * 1024**2


def custody(args):
    gp, fp, ap = (bound(p, BASE) for p in (args.guard, args.freeze, args.admission))
    g, f, a = load(gp), load(fp), load(ap)
    assert g['verified_freeze_hashes'][str(fp.relative_to(ROOT))] == digest(fp)
    for name, expected in f['sources'].items():
        assert digest(name) == expected, name
    assert digest(PRODUCER) == PRODUCER_SHA == f['sources'][str(PRODUCER.relative_to(ROOT))]
    assert f['sources'][str(ap.relative_to(ROOT))] == digest(ap)
    assert a['schema'] == 'SUPPLIED_LEWM_QUALIFICATION_ADMISSION_v1' and a['mode'] == 'probe'
    assert a['admitted'] is True and a['passed'] is True and a['outer_guard_required'] is True
    assert a['source_sha256'] == PRODUCER_SHA and a['entry_source_sha256'] == digest(PRODUCER.with_name('entry.py'))
    assert a['resource_contract'] == RESOURCES and a['inherited_allocation_CPU_s'] == 2137.838031
    assert a['parent_closure_sha256'] == PARENT_SHA == digest(PARENT)
    assert digest(SOURCE_CLOSURE) == SOURCE_CLOSURE_SHA
    assert 0 < a['not_after_unix'] - a['not_before_unix'] <= 300
    expected = {'cpu_cap_s': 16, 'wall_cap_s': 180, 'global_cpu_cap_s': 24,
                'global_wall_cap_s': 300, 'memory_cap_bytes': MEMORY,
                'artifact_cap_bytes': 134217728, 'disk_reserve_bytes': 10737418240}
    assert all(g[key] == value for key, value in expected.items())
    entry = load(BASE / 'ENTRY_probe.json')
    assert entry['passed'] is True and entry['available_bytes'] >= 6442450944 and entry['disk_free_bytes'] >= 10737418240
    assert a['not_before_unix'] <= g['started_unix'] <= entry['unix'] <= g['started_unix'] + g['wall_s']
    normal = (g['cpu_accounting_complete'] is True and g['returncode'] in (0, 1)
              and not g['termination_reason'] and not g['live_descendants_after_stop']
              and not g['final_assertion_violations'] and g['charged_total_cpu_s'] <= 16
              and g['wall_s'] <= 180 and g['started_unix'] + g['wall_s'] <= a['not_after_unix'])
    return g, a, normal


def arrays(np, path, expected_keys):
    path = bound(path, BASE / 'probe')
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        assert len({row.filename for row in members}) == len(members)
        assert {row.filename for row in members} == {key + '.npy' for key in expected_keys}
        assert sum(row.file_size for row in members) <= 64 * 1024**2
    with np.load(path, allow_pickle=False) as saved:
        assert set(saved.files) == set(expected_keys)
        result = {name: saved[name].copy() for name in expected_keys}
    for value in result.values():
        assert not value.dtype.hasobject and value.flags.c_contiguous and np.isfinite(value).all()
    return result


def norm(np, u):
    return ((u - np.asarray(MU, dtype=np.float32)) / np.asarray(SIGMA, dtype=np.float32)).astype(np.float32)


def unnorm(np, z):
    return (z * np.asarray(SIGMA, dtype=np.float32) + np.asarray(MU, dtype=np.float32)).astype(np.float32)


def shape_dtype(np, value, shape, dtype):
    assert value.shape == shape and value.dtype == dtype and np.isfinite(value).all()


def marker_check(np, receipt):
    keys = {'goal_E_cached', 'goal_E_reencoded', 'clip_input_u', 'clipped_u', 'clipped_actual_z', 'hold_u', 'hold_z'}
    for h in (1, 3):
        keys.update(f'H{h}_{name}' for name in ('raw_u', 'normalized_blocks', 'original', 'cached', 'raw_images', 'observation_raw_t', 'action_block_raw_t'))
    path = BASE / 'probe/MARKERS.npz'
    assert receipt['file_sha256'] == digest(path) and receipt['passed'] is True
    assert receipt['rtol'] == RTOL and receipt['atol'] == ATOL
    data = arrays(np, path, keys)
    summaries = []
    for h in (1, 3):
        u, z = data[f'H{h}_raw_u'], data[f'H{h}_normalized_blocks']
        original, optimized = data[f'H{h}_original'], data[f'H{h}_cached']
        shape_dtype(np, u, (2, 3, h + 2, 5, 2), np.float32)
        shape_dtype(np, z, (2, 3, h + 2, 10), np.float32)
        shape_dtype(np, original, (2, 3, h + 3, 192), np.float32)
        shape_dtype(np, optimized, original.shape, np.float32)
        shape_dtype(np, data[f'H{h}_raw_images'], (2, h, 96, 96, 3), np.uint8)
        assert np.array_equal(u, np.linspace(-.8, .8, u.size, dtype=np.float32).reshape(u.shape))
        assert np.array_equal(z, norm(np, u).reshape(z.shape))
        assert np.allclose(unnorm(np, norm(np, u)), u, rtol=1e-6, atol=1e-7)
        assert np.allclose(original, optimized, rtol=RTOL, atol=ATOL)
        observation = np.arange(-(h - 1) * 5, 1, 5, dtype=np.int64)
        starts = np.arange(-(h - 1) * 5, 15, 5, dtype=np.int64)
        assert np.array_equal(data[f'H{h}_observation_raw_t'], observation)
        assert np.array_equal(data[f'H{h}_action_block_raw_t'], starts)
        row = {'initial_H': h, 'HS': 3, 'F': 3, 'action_blocks': h + 2, 'output_frames': h + 3,
               'max_abs': float(np.abs(original - optimized).max()),
               'RMSE': float(np.sqrt(((original - optimized) ** 2).mean()))}
        declared = receipt['cases'][(h - 1) // 2]
        assert all(declared[k] == row[k] for k in ('initial_H', 'HS', 'F', 'action_blocks', 'output_frames'))
        close(declared['max_abs'], row['max_abs']); close(declared['RMSE'], row['RMSE'])
        assert declared['original_dtype'] == declared['cached_dtype'] == 'torch.bfloat16'
        summaries.append(row)
    shape_dtype(np, data['goal_E_cached'], (8, 192), np.float32)
    shape_dtype(np, data['goal_E_reencoded'], (8, 192), np.float32)
    assert np.allclose(data['goal_E_cached'], data['goal_E_reencoded'], rtol=RTOL, atol=ATOL)
    goal_delta = float(np.abs(data['goal_E_cached'] - data['goal_E_reencoded']).max())
    close(goal_delta, receipt['goal_E_max_abs'])
    assert np.array_equal(data['clip_input_u'], np.asarray([[-2., 2.], [.4, -.3]], dtype=np.float32))
    assert np.array_equal(data['clipped_u'], np.clip(data['clip_input_u'], -1, 1))
    assert np.array_equal(data['clipped_actual_z'], norm(np, data['clipped_u']))
    assert data['hold_u'].shape == (3, 2) and np.array_equal(data['hold_u'], np.zeros((3, 2), dtype=np.float32))
    assert np.array_equal(data['hold_z'], norm(np, data['hold_u'])) and not np.array_equal(data['hold_z'], data['hold_u'])
    return data, {'cases': summaries, 'goal_E_max_abs': goal_delta, 'normalization_and_chronological_packing': True}


def state_check(runtime):
    before = load(BASE / 'probe/STRICT_FULL_STATE_BEFORE.json')
    after = load(BASE / 'probe/STRICT_FULL_STATE_AFTER.json')
    assert before == after and len(before) == 303
    bn = {prefix + '.' + suffix for prefix in ('projector.net.1', 'pred_proj.net.1')
          for suffix in ('running_mean', 'running_var', 'num_batches_tracked')}
    assert bn <= set(before)
    parameters = 0
    for name, row in before.items():
        assert all(type(n) is int and n >= 0 for n in row['shape'])
        assert len(row['sha256']) == 64 and all(c in '0123456789abcdef' for c in row['sha256'])
        assert row['dtype'] == ('torch.int64' if name.endswith('num_batches_tracked') else 'torch.float32')
        if name not in bn:
            parameters += math.prod(row['shape'])
    assert parameters == runtime['parameters'] == 18034478 and runtime['state_tensors'] == 303
    assert runtime['full_network_retained'] is True and runtime['all_eval_and_frozen'] is True
    assert runtime['numpy'] == '2.5.3' and runtime['torch'].split('+')[0] == '2.14.0' and runtime['transformers'] == '4.57.6'
    memory_check(runtime['memory'])
    return {'state_tensors': 303, 'parameter_tensors': 297, 'parameter_count_from_saved_shapes': parameters,
            'six_BN_buffers_and_all_saved_hashes_unchanged': True,
            'saved_source_assertion_evidence_not_independent_weight_reload': True}


def query_check(np, index, receipt, marker, rngs, previous_selected):
    stem = BASE / 'probe' / f'QUERY_{index}'
    assert load(stem.with_suffix('.json')) == receipt and receipt['query_index'] == index
    payload_path = BASE / 'probe' / f'QUERY_{index}_PAYLOAD.json'
    payload = load(payload_path)
    assert receipt['payload_sidecar_sha256'] == digest(payload_path)
    assert all(receipt[key] == value for key, value in payload.items())
    assert receipt['excludes_only_final_timing_sidecar_write'] is True and receipt['metering_tail_fully_charged_by_outer_guard'] is True
    for key, value in {'E_calls_B1': 8, 'predictor_calls': 9, 'predictor_batch': 512,
                       'candidate_plans': 1536, 'latent_predicted_block_state_rows': 4608,
                       'real_environment_steps': 0}.items():
        assert receipt[key] == value
    memory_check(receipt['memory'])
    assert math.isfinite(receipt['CPU_s']) and receipt['CPU_s'] > 0
    assert math.isfinite(receipt['synchronized_wall_s']) and receipt['synchronized_wall_s'] > 0
    keys = {'current_RGB', 'goal_E', 'initial_mean_z', 'warm_candidate_u',
            'selected_u', 'selected_global_id', 'selected_cost', 'selected_first5_u', 'shifted_warm_u'}
    for iteration in range(3):
        keys.update(f'i{iteration}_{name}' for name in ('noise_f64', 'actual_z', 'actual_u', 'terminal_E', 'cost', 'elite',
                    'mean_before', 'std_before', 'mean_after', 'std_after'))
    path = stem.with_suffix('.npz')
    assert receipt['arrays_sha256'] == digest(path)
    data = arrays(np, path, keys)
    shape_dtype(np, data['current_RGB'], (8, 96, 96, 3), np.uint8)
    assert np.array_equal(data['goal_E'], marker['goal_E_cached'])
    # Marker frames select from the same fixed eight public engineering images.
    for h in (1, 3):
        assert np.array_equal(marker[f'H{h}_raw_images'], data['current_RGB'][np.arange(2 * h).reshape(2, h) % 8])
    warm = np.zeros((8, 3, 2), dtype=np.float32) if previous_selected is None else np.concatenate((previous_selected[:, 1:], previous_selected[:, -1:]), axis=1)
    mean = np.zeros_like(warm) if previous_selected is None else norm(np, warm)
    std = np.ones_like(mean)
    assert np.array_equal(data['initial_mean_z'], mean) and np.array_equal(data['warm_candidate_u'], warm)
    best_cost = np.full(8, np.inf, dtype=np.float32)
    best_id = np.full(8, -1, dtype=np.int64)
    best_u = np.zeros((8, 3, 2), dtype=np.float32)
    for iteration in range(3):
        prefix = f'i{iteration}_'
        noise = np.stack([rng.standard_normal((64, 3, 2)) for rng in rngs])
        shape_dtype(np, data[prefix + 'noise_f64'], (8, 64, 3, 2), np.float64)
        assert np.array_equal(data[prefix + 'noise_f64'], noise)
        assert np.array_equal(data[prefix + 'mean_before'], mean) and np.array_equal(data[prefix + 'std_before'], std)
        sampled = (mean[:, None] + std[:, None] * noise).astype(np.float32)
        u = np.clip(unnorm(np, sampled), -1, 1)
        u[:, 0] = 0.
        u[:, 1] = warm
        z = norm(np, u)
        shape_dtype(np, data[prefix + 'actual_u'], u.shape, np.float32)
        shape_dtype(np, data[prefix + 'actual_z'], z.shape, np.float32)
        assert np.array_equal(data[prefix + 'actual_u'], u) and np.array_equal(data[prefix + 'actual_z'], z)
        terminal = data[prefix + 'terminal_E']
        shape_dtype(np, terminal, (8, 64, 192), np.float32)
        cost = ((terminal[:, :, None] - data['goal_E'][None, None]) ** 2).mean(-1).min(-1)
        assert cost.dtype == np.float32 and np.array_equal(data[prefix + 'cost'], cost)
        elite = np.argsort(cost, axis=1, kind='stable')[:, :8]
        assert np.array_equal(data[prefix + 'elite'], elite)
        for slot in range(8):
            chosen = int(np.argmin(cost[slot]))
            # Strict improvement preserves the lower global ID on exact ties.
            if cost[slot, chosen] < best_cost[slot]:
                best_cost[slot] = cost[slot, chosen]
                best_id[slot] = iteration * 64 + chosen
                best_u[slot] = u[slot, chosen]
            postclip_elites = z[slot, elite[slot]]
            mean[slot] = .25 * mean[slot] + .75 * postclip_elites.mean(0)
            std[slot] = np.maximum(.1, .25 * std[slot] + .75 * postclip_elites.std(0))
        assert np.array_equal(data[prefix + 'mean_after'], mean) and np.array_equal(data[prefix + 'std_after'], std)
    assert np.array_equal(data['selected_u'], best_u)
    assert np.array_equal(data['selected_global_id'], best_id) and np.array_equal(data['selected_cost'], best_cost)
    assert np.array_equal(data['selected_first5_u'], np.repeat(best_u[:, 0, None, :], 5, axis=1))
    assert np.array_equal(data['shifted_warm_u'], np.concatenate((best_u[:, 1:], best_u[:, -1:]), axis=1))
    return best_u, {'query_index': index, 'candidate_plans_checked': 1536,
                    'persistent_RNG_arrays_reproduced': 24,
                    'source_loop_counts_and_saved_shapes_not_model_reexecution': True,
                    'CPU_s': receipt['CPU_s'], 'synchronized_wall_s': receipt['synchronized_wall_s']}


def forecast_check(result):
    queries = result['queries']
    assert len(queries) == 4
    q = max(row['CPU_s'] for row in queries[1:])
    qw = max(row['synchronized_wall_s'] for row in queries[1:])
    first, first_wall = queries[0]['CPU_s'], queries[0]['synchronized_wall_s']
    cold, cold_wall = result['cold_setup_CPU_s'], result['cold_setup_wall_s']
    assert all(math.isfinite(x) and x >= 0 for x in (cold, cold_wall))
    startup = max(8., 1.5 * cold + max(0., first - q))
    startup_wall = max(45., 1.5 * cold_wall + max(0., first_wall - qw))
    cpu, wall = 90 * q + startup + 12, 90 * qw + startup_wall + 120
    declared = result['forecast']
    for name, value in {'warm_max_CPU_s': q, 'warm_max_wall_s': qw, 'startup_CPU_s': startup,
                        'startup_wall_s': startup_wall, 'control_CPU_s': cpu, 'control_wall_s': wall}.items():
        close(declared[name], value)
    for name, value in {'batched_group_rounds': 60, 'individual_decisions': 480,
                        'batched_predictor_calls': 540, 'latent_block_state_rows': 276480,
                        'real_raw_step_upper_count': 2400, 'native_and_other_CPU_allowance_s': 12,
                        'native_and_other_wall_allowance_s': 120,
                        'conditional_CPU_eligible': cpu <= 180, 'conditional_wall_eligible': wall <= 900,
                        'allowances_are_forecasts_not_hard_physical_bounds': True,
                        'control_audit_10_and_shared_10_CPU_not_added_to_control_180': True}.items():
        assert declared[name] == value
    return {'control_CPU_forecast_s': cpu, 'control_wall_forecast_s': wall,
            'conditional_CPU_eligible': cpu <= 180, 'conditional_wall_eligible': wall <= 900,
            'separate_future_audit_CPU_s': 10, 'separate_future_shared_CPU_s': 10,
            'forecast_not_actual_fee_or_execution_admission': True,
            'finite_sample_forecast_not_proven_runtime_upper_bound': True}


FIXTURE_U = ((1., 0.), (.25, .5), (-.5, -.25), (0., -.75), (.75, .25),
             (-.25, .75), (.5, -.5), (-.75, 0.), (.25, -.25), (0., .5),
             (-.5, .25), (.75, -.75), (-.25, -.5), (.5, .75), (0., 0.))


def actuator_check(np, fixture):
    path = BASE / 'probe/ACTUATOR_FIXTURE.json'
    assert load(path) == fixture and fixture['passed'] is True
    assert fixture['schema'] == 'RELATIVE_NATIVE_PARITY_RESULT_v1'
    assert fixture['same_native_backend_not_independent_physics'] is True
    assert fixture['no_old_absolute_restore_or_replay'] is True
    assert fixture['no_model_training_task_policy_or_closed_test'] is True and fixture['no_policy_performance_selection'] is True
    plan = fixture['plan']
    assert plan['seeds'] == [97000, 97001] and plan['raw_controls_per_sequence'] == 15
    assert plan['paired_native_sequences'] == 4 and plan['own_relative_replay_sequences'] == 2
    assert plan['native_raw_controls'] == 90 and plan['expected_physics_substeps'] == 900
    assert plan['render_calls'] == 4 and plan['invalid_actions_rejected_before_physics'] == 8
    assert len(fixture['cases']) == 2
    joined_events, checked_controls = [], 0
    for index, case in enumerate(fixture['cases']):
        assert case['seed'] == 97000 + index and case['passed'] is True
        expected_pose = [32., 80., 256., 256., 0.] if index == 0 else [480., 432., 256., 256., 0.]
        assert case['requested_pose'] == expected_pose and case['invalid_action_rejections'] == 4
        expected_u = [list(u) for u in FIXTURE_U]
        expected_u[0][0] = -1. if index == 0 else 1.
        assert case['actions'] == expected_u and len(case['rows']) == len(case['own_relative_replay_records']) == 15
        initial = np.asarray(case['initial_state11'], dtype=np.float64)
        shape_dtype(np, initial, (11,), np.float64)
        previous = initial.copy()
        for raw_index, (item, replay) in enumerate(zip(case['rows'], case['own_relative_replay_records'])):
            row = item['wrapper']
            u = np.asarray(expected_u[raw_index], dtype=np.float64)
            assert item['raw_t'] == raw_index + 1 and item['u'] == expected_u[raw_index]
            before = np.asarray(row['before_state11'], dtype=np.float64)
            after = np.asarray(row['after_state11'], dtype=np.float64)
            shape_dtype(np, before, (11,), np.float64); shape_dtype(np, after, (11,), np.float64)
            assert np.array_equal(before, previous)
            assert item['reference_before_state11'] == before.tolist() and item['reference_after_state11'] == after.tolist()
            assert item['state11_elementwise_exact'] is True and row['issued_u'] == expected_u[raw_index]
            target = before[:2] + 100. * u
            assert row['actual_target'] == item['reference_target'] == target.tolist()
            assert row['target_outside_old_absolute_box'] == bool(((target < 0.) | (target > 512.)).any())
            for actual in (row, replay):
                assert actual['native_step_returned'] is True and actual['native_step_calls_started'] == 1
                assert actual['expected_native_step_calls'] == 1 and actual['expected_native_substeps'] == 10
                assert actual['raw_t_before'] == raw_index and actual['control_steps_after'] == raw_index + 1
                assert actual['no_absolute_target_clip'] is True and actual['not_a_planner_observation'] is True
                assert actual['issued_u'] == expected_u[raw_index]
                assert actual['before_state11'] == before.tolist() and actual['after_state11'] == after.tolist()
                assert actual['actual_target'] == target.tolist()
                assert actual['seed'] == 97000 + index
            assert row['session'] == 0 and replay['session'] == 1
            previous = after
            checked_controls += 3  # Wrapper + separate native reference + own replay.
        assert case['out_of_box_target_exercised'] is True and case['rows'][0]['wrapper']['target_outside_old_absolute_box'] is True
        events = case['private_events']
        expected_names = (['relative_reset_started', 'relative_reset_returned']
                          + ['relative_control_started', 'relative_control_closed'] * 15) * 2
        assert [row['event'] for row in events] == expected_names
        closed = [row for row in events if row['event'] == 'relative_control_closed']
        assert len(closed) == 30
        for actual, expected in zip(closed, [row['wrapper'] for row in case['rows']] + case['own_relative_replay_records']):
            assert {k: v for k, v in actual.items() if k != 'event'} == expected
        joined_events.extend(events)
    journal = [json.loads(x) for x in bound(BASE / 'probe/PRIVATE_ACTUATOR_EVENTS.jsonl').read_text().splitlines() if x]
    assert journal == joined_events and checked_controls == 90
    for name, expected in fixture['source_bindings'].items():
        assert digest(name) == expected
    return {'cases': 2, 'saved_native_controls_checked': checked_controls,
            'source_derived_expected_microsteps': 900, 'private_event_rows': len(journal),
            'outside_old_box_per_raw_target_equation_checked': True,
            'reference_and_replay_saved_states_exactly_equal': True,
            'same_simulator_saved_evidence_not_independent_physics_reexecution': True,
            'fixture_CPU_s_nested': fixture['fixture_body_CPU_s_nested'],
            'fixture_wall_s_nested': fixture['fixture_body_wall_s_nested'],
            'native12CPU120wall_allowance_requires_root_review_before_control': True}


def event_check(result, complete):
    path = BASE / 'probe/EVENTS.jsonl'
    events = [json.loads(row) for row in path.read_text().splitlines() if row] if path.is_file() else []
    expected = [('cold_load_started', None), ('cold_load_returned', None)]
    for index in range(4):
        expected += [('query_started', index), ('query_returned', index)]
    expected += [('markers_started', None), ('markers_returned', None),
                 ('actuator_fixture_started', None), ('actuator_fixture_returned', None)]
    actual = [(row['phase'], row.get('query')) for row in events]
    assert actual == expected[:len(actual)]
    if complete:
        assert actual == expected and len(result['queries']) == 4
    returned = [row['query'] for row in events if row['phase'] == 'query_returned']
    assert returned == list(range(len(result['queries'])))
    return {'events': len(events), 'queries_returned': len(returned), 'source_order_prefix_consistent': True}


def audit(args):
    guard, admission, normal = custody(args)
    base = {'schema': 'SUPPLIED_LEWM_SAVED_QUALIFICATION_AUDIT_v1', 'passed': False,
            'audit_complete': False, 'qualification_passed': False, 'forensic_consistency_passed': False,
            'new_model_native_or_training_calls': 0, 'control_execution_admitted': False,
            'producer_guard_charged_CPU_s': guard['charged_total_cpu_s'],
            'producer_CPU_accounting_complete': guard['cpu_accounting_complete'],
            'nested_body_fee_not_added': True, 'full_checkpoint_not_deserialized_by_auditor': True}
    directory = BASE / 'probe'
    terminals = [p for p in (directory / 'RESULT.json', directory / 'FAILURE.json') if p.is_file()]
    if not normal or len(terminals) != 1:
        return {**base, 'reason': 'outer_stop_or_missing_terminal'}
    result = load(terminals[0])
    assert result['schema'] == 'SUPPLIED_LEWM_QUALIFICATION_RESULT_v1'
    assert result['scientific_training_updates'] == result['task_policy_rollouts'] == result['closed_episodes_opened'] == 0
    assert result['control_execution_admitted'] is False
    for name, expected in result['input_bindings'].items():
        assert digest(name) == expected
    assert result['input_bindings'][str(bound(args.admission).relative_to(ROOT))] == digest(args.admission)
    workflow = result['workflow_complete']
    assert type(workflow) is bool and result['qualification_passed'] == workflow
    assert (guard['returncode'] == 0) == workflow
    events = event_check(result, workflow)
    saved = {str(path.relative_to(ROOT)): digest(path) for path in directory.rglob('*') if path.is_file()}
    if not workflow:
        for index, receipt in enumerate(result['queries']):
            assert load(directory / f'QUERY_{index}.json') == receipt
            assert digest(directory / f'QUERY_{index}.npz') == receipt['arrays_sha256']
            assert digest(directory / f'QUERY_{index}_PAYLOAD.json') == receipt['payload_sidecar_sha256']
        return {**base, 'passed': True, 'audit_complete': True, 'forensic_consistency_passed': True,
                'events': events, 'saved_partial_artifact_sha256': saved,
                'incomplete_qualification_not_task_failure': True,
                'partial_scope_is_hash_journal_consistency_without_runtime_retry': True}
    assert not any(name.split('.')[0] in ('torch', 'jax') for name in sys.modules)
    assert Path(sys.executable).absolute() == ROOT / '.venv/bin/python'
    import numpy as np
    assert np.__version__ == '2.5.3'
    assert Path(np.__file__).resolve().is_relative_to(ROOT / '.venv')
    state = state_check(result['runtime'])
    assert result['full_state_and_BN_bytes_unchanged'] is True
    if 'memory_after_goal_setup' in result['runtime']:
        memory_check(result['runtime']['memory_after_goal_setup'])
    assert result['runtime']['goal_setup_includes_eight_B1_reencodes'] is True
    marker_data, marker = marker_check(np, result['markers'])
    rngs = [np.random.default_rng(34199 + 97000 + slot) for slot in range(8)]
    previous = None
    queries = []
    for index, receipt in enumerate(result['queries']):
        previous, checked = query_check(np, index, receipt, marker_data, rngs, previous)
        queries.append(checked)
    assert len(queries) == 4
    fixture = actuator_check(np, result['actuator_fixture'])
    forecast = forecast_check(result)
    assert not any(name.split('.')[0] in ('torch', 'jax') for name in sys.modules)
    return {**base, 'passed': True, 'audit_complete': True, 'qualification_passed': True,
            'forensic_consistency_passed': True, 'events': events, 'full_state': state,
            'markers': marker, 'queries': queries, 'actuator_fixture': fixture,
            'forecast': forecast, 'saved_artifact_sha256': saved,
            'saved_numeric_evidence_not_new_prediction_or_control_value': True,
            'source_fixed_adaptation_not_historical_normalizer_recovery': True}


def main():
    parser = argparse.ArgumentParser()
    for name in ('guard', 'freeze', 'admission'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    out = BASE / 'saved_audit/AUDIT.json'
    assert not out.exists()
    cpu, wall = time.process_time(), time.perf_counter()
    try:
        result = audit(args)
    except Exception as error:
        result = {'schema': 'SUPPLIED_LEWM_SAVED_QUALIFICATION_AUDIT_v1', 'passed': False,
                  'audit_complete': False, 'qualification_passed': False,
                  'forensic_consistency_passed': False, 'control_execution_admitted': False,
                  'safe_error_code': 'saved_qualification_custody_or_numeric_check_failed',
                  'error_type': type(error).__name__}
    result.update(auditor_source_sha256=digest(__file__), audit_body_CPU_s=time.process_time() - cpu,
                  audit_body_wall_s=time.perf_counter() - wall, audit_body_not_own_outer_guard_fee=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print('Saved full-model qualification audit closed; no task/control result follows.')


if __name__ == '__main__':
    main()
