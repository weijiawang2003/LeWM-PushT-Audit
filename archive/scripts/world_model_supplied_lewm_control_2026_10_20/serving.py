"""Root-only serving entry for the already qualified supplied-model package.

No new model, CEM, optimizer or execution recipe is defined here. Importing
this source defines standard-library code only; root must separately admit it.
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
BASE = ROOT / 'results/world_model_supplied_lewm_control_2026_10_20'
OUT = BASE / 'control'
QUAL_BASE = ROOT / 'results/world_model_supplied_lewm_qualification_2026_10_20'
PARENT_SHA = '2cf8041d83ea8f5f8b842359173d0e2b90ea68f2b1c0423865126c78a108adf5'
QUAL = ROOT / 'scripts/world_model_supplied_lewm_qualification_2026_10_20/qualify_full_model.py'
QUAL_SHA = 'cf77a53c4dcc0108a747898959ce5d56746ccb4be0132d33af88a69f772a233d'
DRIVER = Path(__file__).with_name('control_driver.py')
DRIVER_SHA = '848e0e5593b741da9d5cbeb921d2e81fa2c30bb1fada1ebe22881dd7a53c23b7'
RELATIVE = Path(__file__).with_name('relative_executor.py')
RELATIVE_SHA = 'baf1dc572308672bfc34b7cc1bc4cb4c85d43fad4714328ac611c02b9e7848b8'
RESOURCES = {'phase_CPU_s': 200, 'control_CPU_s': 180, 'control_wall_s': 900,
    'audit_CPU_s': 10, 'audit_wall_s': 120, 'global_wall_s': 1200,
    'memory_bytes': 2147483648, 'artifact_bytes': 134217728,
    'OS_available_bytes': 6442450944, 'disk_reserve_bytes': 10737418240}


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            value.update(block)
    return value.hexdigest()


def bound(name):
    path = Path(name)
    if not path.is_absolute():
        path = ROOT / path
    assert not path.is_symlink()
    path = path.resolve()
    path.relative_to(ROOT)
    assert path.is_file()
    return path


def load(path):
    return json.loads(bound(path).read_text())


def write(path, data):
    with Path(path).open('x') as stream:
        json.dump(data, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def admit(path):
    path = bound(path)
    path.relative_to(BASE)
    a = load(path)
    assert a['schema'] == 'SUPPLIED_LEWM_CONTROL_ADMISSION_v1' and a['mode'] == 'control'
    assert a['admitted'] is True and a['passed'] is True and a['outer_guard_required'] is True
    assert a['source_sha256'] == sha(__file__)
    entry = Path(__file__).with_name('entry.py')
    assert a['entry_source_sha256'] == sha(entry)
    assert a['resource_contract'] == RESOURCES and a['inherited_allocation_CPU_s'] == 2142.979764
    parent = bound(a['parent_closure_path'])
    assert parent == QUAL_BASE / 'PHASE_CLOSURE.json' and sha(parent) == a['parent_closure_sha256'] == PARENT_SHA
    closure = load(parent)
    assert closure['phase_complete'] is True and closure['qualification_passed'] is True
    assert closure['saved_numeric_audit_complete'] is True
    assert closure['conservative_visual_plus_annex_allocation_CPU_s'] == 2142.979764
    assert 0 < a['not_after_unix'] - a['not_before_unix'] <= 1200
    assert a['not_before_unix'] <= time.time() < a['not_after_unix'] - 15
    result_path, audit_path = QUAL_BASE / 'probe/RESULT.json', QUAL_BASE / 'saved_audit/AUDIT.json'
    assert sha(result_path) == a['qualification_result_sha256']
    assert sha(audit_path) == a['qualification_audit_sha256']
    result, audit = load(result_path), load(audit_path)
    assert result['workflow_complete'] is True and result['qualification_passed'] is True
    assert result['full_state_and_BN_bytes_unchanged'] is True
    assert result['forecast']['conditional_CPU_eligible'] is True
    assert result['forecast']['conditional_wall_eligible'] is True
    assert audit['passed'] is True and audit['qualification_passed'] is True
    assert audit['forensic_consistency_passed'] is True
    for name, expected in result['input_bindings'].items():
        assert sha(bound(name)) == expected
    for p, expected in ((QUAL, QUAL_SHA), (DRIVER, DRIVER_SHA), (RELATIVE, RELATIVE_SHA)):
        assert sha(p) == expected
    receipt = BASE / 'ENTRY_control.json'
    r = load(receipt)
    assert r['passed'] and r['available_bytes'] >= 6442450944 and r['disk_free_bytes'] >= 10737418240
    assert a['not_before_unix'] <= r['unix'] <= time.time()
    bindings = {str(p.relative_to(ROOT)): sha(p) for p in
        (path, entry, parent, result_path, audit_path, receipt, QUAL, DRIVER, RELATIVE)}
    return a, bindings


def run(a, bindings, cpu_origin, wall_origin):
    OUT.mkdir(exist_ok=False)
    queries = OUT / 'policy_queries'
    queries.mkdir(exist_ok=False)
    state = {'schema': 'SUPPLIED_LEWM_SERVING_RESULT_v1', 'workflow_complete': False,
        'passed': False, 'panel_complete': False, 'input_bindings': bindings,
        'scientific_training_updates': 0, 'closed_episodes_opened': 0,
        'fixed_public_validation_seeds': list(range(88000, 88008)),
        'one_fixed_goal_not_cross_goal_reuse': True,
        'supplied_checkpoint_adaptation_not_paper_reproduction': True,
        'external_pretraining_cost_and_overlap_unknown': True,
        'CPU_is_guard_tree_CPU_not_complete_host_or_energy': True,
        'nested_body_clocks_not_added_to_complete_guard_fee': True}
    bundle = None
    try:
        sys.dont_write_bytecode = True
        os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
        spec = importlib.util.spec_from_file_location('qualified_supplied_lewm_serving', QUAL)
        qual = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = qual
        spec.loader.exec_module(qual)
        cold_cpu, cold_wall = time.process_time(), time.perf_counter()
        bundle = qual.load_bundle(queries)
        state['cold_model_and_goal_setup_CPU_s'] = time.process_time() - cold_cpu
        state['cold_model_and_goal_setup_wall_s'] = time.perf_counter() - cold_wall
        state['runtime'] = bundle['runtime']
        state['input_bindings'].update(bundle['bindings'])
        np, torch = bundle['np'], bundle['torch']
        os.environ['SDL_VIDEODRIVER'] = 'dummy'
        os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
        sys.path.append(str(ROOT / '.venv_object/lib/python3.12/site-packages'))
        sys.path.insert(0, str(ROOT / 'src'))
        import cv2
        cv2.setNumThreads(1)
        from jepa4object.environment import PushTStateEnv, coverage_from_state
        relative = qual.load_source('supplied_control_relative_executor', RELATIVE)
        driver = qual.load_source('supplied_control_fixed_B8_driver', DRIVER)
        policy = qual.SuppliedPolicy(bundle, queries)
        panel = driver.run_panel(np, lambda: PushTStateEnv(goal=driver.GOAL), coverage_from_state,
            policy, OUT, executor_class=relative.RelativeExecutor,
            deadline_unix=a['not_after_unix'], cpu_budget_s=180., wall_budget_s=900.,
            phase_cpu_origin=cpu_origin, phase_wall_origin=wall_origin,
            reserve_CPU_s=2., reserve_wall_s=5., metadata={
                'qualification_source_sha256': QUAL_SHA, 'driver_source_sha256': DRIVER_SHA,
                'relative_executor_source_sha256': RELATIVE_SHA,
                'fixed_B8_queries_match_qualified_serialization': True,
                'actual_full_state_parameters': 18034478,
                'model_E_goal_bank_not_changed': True})
        candidates = [p for p in (OUT / 'RESULT.json', OUT / 'FAILURE.json') if p.is_file()]
        assert len(candidates) == 1
        state['panel_result_binding'] = {'path': str(candidates[0].relative_to(ROOT)),
            'sha256': sha(candidates[0]), 'bytes': candidates[0].stat().st_size}
        assert load(candidates[0]) == panel
        state['panel_complete'] = panel['panel_complete']
        state['panel_successes_if_complete'] = panel['complete_panel_successes']
        state['returned_raw_controls'] = panel['returned_raw_controls']
        state['policy_group_calls_started'] = panel['policy_group_calls_started']
        state['policy_group_calls_returned'] = panel['policy_group_calls_returned']
        after = bundle['feature'].state_identity(torch, bundle['model'])
        write(queries / 'STRICT_FULL_STATE_AFTER.json', after)
        state['full_state_identity_bindings'] = {str(p.relative_to(ROOT)): sha(p) for p in
            (queries / 'STRICT_FULL_STATE_BEFORE.json', queries / 'STRICT_FULL_STATE_AFTER.json')}
        state['full_state_and_BN_bytes_unchanged'] = bundle['before'] == after
        assert state['full_state_and_BN_bytes_unchanged']
        state['final_memory'] = qual.memory(torch)
        state['workflow_complete'] = True
        state['passed'] = state['panel_complete'] is True
    except Exception as error:
        state['error_type'] = type(error).__name__
        state['safe_error_code'] = 'supplied_lewm_serving_incomplete'
    state['preterminal_phase_CPU_s'] = time.process_time() - cpu_origin
    state['preterminal_phase_wall_s'] = time.perf_counter() - wall_origin
    state['final_write_and_exit_remain_complete_outer_guard_cost'] = True
    write(OUT / 'SERVING_RESULT.json', state)
    return state['passed']


def main():
    cpu_origin, wall_origin = time.process_time(), time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument('--admission', required=True)
    args = parser.parse_args()
    try:
        a, bindings = admit(args.admission)
        passed = run(a, bindings, cpu_origin, wall_origin)
        print('Supplied fixed-package control stage closed; inspect complete-panel and cost scope.')
        return 0 if passed else 1
    except Exception as error:
        print('Supplied control entry incomplete:', type(error).__name__)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
