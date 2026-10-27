"""Root-only supplied full-LeWM engineering qualification; no training.

Import defines standard-library code only. Queries contain no environment,
privileged state, task reward, or task-policy execution.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'results/world_model_supplied_lewm_qualification_2026_10_20'
VISUAL = ROOT / 'results/world_model_visual_transfer_2026_10_19'
FEATURE = ROOT / 'scripts/world_model_visual_transfer_2026_10_19/feature_cache_small.py'
FEATURE_SHA = 'ad854db4f5800fc55fa67f8414b683a9e50afc29f82cafe6f6d660d69137bb2a'
PARENT = ROOT / 'results/world_model_external_action_storage_map_2026_10_20/PHASE_CLOSURE.json'
PARENT_SHA = '387c08db9190b39f27665a6ab50d76e842200c638e1080f6570ef07ff98e294a'
READINESS = ROOT / 'results/world_model_external_action_normalization_readiness_2026_10_20/PHASE_CLOSURE.json'
READINESS_SHA = 'cd41262c933f6c874509da8328c5d7f23e6ddda9949b84d0af53fdc310de61ac'
PROTOCOL = ROOT / 'reports/world_model_external_action_normalization_readiness_2026_10_20/ROOT_PROTOCOL_DECISION.md'
PROTOCOL_SHA = 'a8891e8d9e995c90d77e0954e5524de76ba33fbc109187f0fe3a1cc21ac9d51c'
RELATIVE = ROOT / 'scripts/world_model_supplied_lewm_control_2026_10_20/relative_executor.py'
RELATIVE_SHA = 'baf1dc572308672bfc34b7cc1bc4cb4c85d43fad4714328ac611c02b9e7848b8'
MEMORY = 2147483648
RESOURCES = {'phase_CPU_s': 24, 'probe_CPU_s': 16, 'probe_wall_s': 180,
    'audit_CPU_s': 4, 'audit_wall_s': 60, 'global_wall_s': 300,
    'artifact_bytes': 134217728, 'memory_bytes': MEMORY,
    'OS_available_bytes': 6442450944, 'disk_reserve_bytes': 10737418240}
MU = (-.0087, .0068)
SIGMA = (.2019, .2002)
RTOL, ATOL = 1 / 64, 1 / 128


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1048576), b''):
            digest.update(part)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def event(out, phase, **extra):
    with (out / 'EVENTS.jsonl').open('a') as stream:
        stream.write(json.dumps({'phase': phase, **extra}, sort_keys=True) + '\n')
        stream.flush()


def memory(torch):
    torch.mps.synchronize()
    driver = int(torch.mps.driver_allocated_memory())
    rss_peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    row = {'peak_RSS_bytes': rss_peak, 'MPS_driver_bytes': driver,
        'sampled_overlapping_stop_proxy_bytes': rss_peak + driver,
        'proxy_is_not_measured_unified_memory': True}
    assert driver < MEMORY and rss_peak + driver < MEMORY - 64 * 1024**2
    return row


def admit(path):
    path = Path(path).resolve()
    path.relative_to(BASE)
    a = read(path)
    assert a['schema'] == 'SUPPLIED_LEWM_QUALIFICATION_ADMISSION_v1' and a['mode'] == 'probe'
    assert a['admitted'] is True and a['passed'] is True and a['outer_guard_required'] is True
    assert a['source_sha256'] == sha(__file__)
    entry = Path(__file__).with_name('entry.py')
    assert a['entry_source_sha256'] == sha(entry)
    assert a['resource_contract'] == RESOURCES and a['inherited_allocation_CPU_s'] == 2137.838031
    assert a['parent_closure_sha256'] == PARENT_SHA == sha(PARENT)
    assert sha(READINESS) == READINESS_SHA and sha(PROTOCOL) == PROTOCOL_SHA
    assert sha(RELATIVE) == RELATIVE_SHA
    assert 0 < a['not_after_unix'] - a['not_before_unix'] <= 300
    assert a['not_before_unix'] <= time.time() < a['not_after_unix'] - 15
    receipt = BASE / 'ENTRY_probe.json'
    r = read(receipt)
    assert r['passed'] and r['available_bytes'] >= 6442450944 and r['disk_free_bytes'] >= 10737418240
    assert a['not_before_unix'] <= r['unix'] <= time.time()
    return a, {str(p.relative_to(ROOT)): sha(p) for p in (path, entry, receipt, PARENT, READINESS, PROTOCOL, RELATIVE)}


def load_bundle(out):
    assert Path(sys.executable).absolute() == ROOT / '.venv/bin/python'
    assert sys.version_info[:3] == (3, 12, 14)
    assert sha(FEATURE) == FEATURE_SHA
    sys.dont_write_bytecode = True
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    feature = load_source('supplied_lewm_frozen_source_helpers', FEATURE)
    assert sha(feature.CORE) == feature.CORE_SHA and sha(feature.ADAPTER) == feature.ADAPTER_SHA
    core = load_source('supplied_lewm_official_constructor', feature.CORE)
    for name, value in core.THREAD_ENV.items():
        assert os.environ.get(name) == value, name
    bindings = feature.checkpoint_contract(feature.WEIGHTS)
    official = core.verify_inputs()
    assert official.pop('no_data_or_checkpoint_inputs') is True
    sys.path.insert(0, str(core.SOURCE))
    sys.path.insert(0, str(core.VENDOR))
    import numpy as np
    import torch
    import transformers
    from transformers import ViTConfig, ViTModel
    assert np.__version__ == '2.5.3' and torch.__version__.split('+')[0] == '2.14.0'
    assert transformers.__version__ == '4.57.6' and torch.backends.mps.is_available()
    if torch.get_num_threads() != 1:
        torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.manual_seed(34199)
    torch.mps.manual_seed(34199)
    recommended = int(torch.mps.recommended_max_memory())
    torch.mps.set_per_process_memory_fraction(min(1., math.nextafter(MEMORY / recommended, 0.)))
    module = core.official_module('supplied_lewm_official_module', core.SOURCE / 'module.py')
    jepa = core.official_module('supplied_lewm_official_jepa', core.SOURCE / 'jepa.py')
    model = core.construct(torch, module, jepa, ViTConfig, ViTModel)
    assert sum(p.numel() for p in model.parameters()) == 18034478
    checkpoint = torch.load(feature.WEIGHTS, map_location='cpu', weights_only=True)
    expected = model.state_dict()
    assert len(checkpoint) == len(expected) == 303 and set(checkpoint) == set(expected)
    assert all(isinstance(v, torch.Tensor) and v.shape == expected[k].shape and v.dtype == expected[k].dtype
               and bool(torch.isfinite(v).all()) for k, v in checkpoint.items())
    loaded = model.load_state_dict(checkpoint, strict=True)
    assert not loaded.missing_keys and not loaded.unexpected_keys
    before = feature.state_identity(torch, model)
    write(out / 'STRICT_FULL_STATE_BEFORE.json', before)
    del expected, checkpoint
    model.requires_grad_(False)
    model.eval()
    model.to('mps')
    assert set(dict(model.named_children())) == {'encoder', 'projector', 'action_encoder', 'predictor', 'pred_proj'}
    assert all(not m.training for m in model.modules()) and all(not p.requires_grad for p in model.parameters())
    adapter = load_source('supplied_lewm_pixel_transform', feature.ADAPTER)
    branch_path = VISUAL / 'E_CACHE_BRANCH.json'
    branch = read(branch_path)
    assert branch['cache_source_sha256'] == FEATURE_SHA and branch['cache_source'] == str(FEATURE.relative_to(ROOT))
    cache = ROOT / branch['cache_directory']
    manifest_path = cache / 'CACHE_MANIFEST.json'
    manifest = read(manifest_path)
    assert manifest['passed'] and manifest['source_sha256'] == FEATURE_SHA
    goal_z_path = cache / 'goalbank.npz'
    assert sha(goal_z_path) == manifest['goalbank']['sha256']
    goal_path = VISUAL / 'goalbank_v2/GOAL_BANK.npz'
    goal_proof = VISUAL / 'goalbank_v2/QUALIFICATION.json'
    proof = read(goal_proof)
    assert proof['passed'] and proof['goal_bank_npz_sha256'] == sha(goal_path)
    with np.load(goal_path, allow_pickle=False) as saved:
        images = saved['images'].copy()  # No goal pose/state11 array is accessed.
    with np.load(goal_z_path, allow_pickle=False) as saved:
        goal_z = saved['z'].copy()
    assert images.shape == (8, 96, 96, 3) and images.dtype == np.uint8
    assert goal_z.shape == (8, 192) and goal_z.dtype == np.float32 and np.isfinite(goal_z).all()
    bindings.update({str(p.relative_to(ROOT)): sha(p) for p in
        (FEATURE, feature.CORE, feature.ADAPTER, branch_path, manifest_path, goal_z_path, goal_path, goal_proof)})
    bundle = {'np': np, 'torch': torch, 'model': model, 'adapter': adapter,
        'feature': feature, 'before': before, 'goal_z': goal_z, 'images': images,
        'bindings': bindings, 'runtime': {'numpy': np.__version__, 'torch': torch.__version__,
            'transformers': transformers.__version__, 'parameters': 18034478, 'state_tensors': 303,
            'full_network_retained': True, 'all_eval_and_frozen': True,
            'external_checkpoint_adaptation_not_historical_scaler_recovery': True,
            'memory': memory(torch)}}
    # This goal-setup work is repeated and charged in serving's cold loader.
    # Keep the original cached bank for all scoring; never replace it.
    bundle['goal_reencoded'] = np.concatenate([encode(bundle, image[None, None]).float().cpu().numpy().reshape(1, 192)
                                               for image in images], axis=0)
    assert np.allclose(bundle['goal_reencoded'], goal_z, rtol=RTOL, atol=ATOL)
    bundle['runtime']['goal_setup_includes_eight_B1_reencodes'] = True
    bundle['runtime']['memory_after_goal_setup'] = memory(torch)
    return bundle


def pixels(bundle, raw):
    np, torch, adapter = bundle['np'], bundle['torch'], bundle['adapter']
    shape = raw.shape[:-3]
    flat = np.asarray(raw, dtype=np.float32).reshape(-1, 96, 96, 3)
    padded = (len(flat) + 3) // 4 * 4
    pack = np.empty((padded, 96, 96, 3), dtype=np.float32)
    pack[:len(flat)] = flat
    pack[len(flat):] = flat[-1]
    tensor = adapter.transform_pixels(torch, pack.reshape(-1, 4, 96, 96, 3)).reshape(-1, 3, 224, 224)
    return tensor[:len(flat)].reshape(*shape, 3, 224, 224).to('mps')


def amp(bundle):
    return bundle['torch'].autocast(device_type='mps', dtype=bundle['torch'].bfloat16)


def assert_amp(bundle):
    torch = bundle['torch']
    assert torch.is_autocast_enabled('mps') and torch.get_autocast_dtype('mps') == torch.bfloat16


def encode(bundle, raw):
    torch, model = bundle['torch'], bundle['model']
    with torch.inference_mode(), amp(bundle):
        assert_amp(bundle)
        z = model.encode({'pixels': pixels(bundle, raw)})['emb']
        assert z.dtype == torch.bfloat16
    return z


def predict_cached(bundle, initial_z, actions, future=3):
    """Same official modules/HS3 operations, with reusable encoded history."""
    torch, model = bundle['torch'], bundle['model']
    h = initial_z.shape[1]
    assert actions.shape[1:] == (h - 1 + future, 10)
    emb = initial_z.clone()
    with torch.inference_mode(), amp(bundle):
        assert_amp(bundle)
        for step in range(future):
            action_emb = model.action_encoder(actions[:, :h + step])
            prediction = model.predict(emb[:, -3:], action_emb[:, -3:])[:, -1:]
            assert prediction.dtype == torch.bfloat16
            emb = torch.cat((emb, prediction), dim=1)
    return emb


def norm(np, u):
    return ((u - np.asarray(MU, dtype=np.float32)) / np.asarray(SIGMA, dtype=np.float32)).astype(np.float32)


def unnorm(np, z):
    return (z * np.asarray(SIGMA, dtype=np.float32) + np.asarray(MU, dtype=np.float32)).astype(np.float32)


def markers(bundle, out):
    np, torch, model = bundle['np'], bundle['torch'], bundle['model']
    arrays, receipt = {}, {'rtol': RTOL, 'atol': ATOL, 'cases': []}
    for h in (1, 3):
        raw = bundle['images'][np.arange(2 * h).reshape(2, h) % 8]
        p = pixels(bundle, raw)
        u = np.linspace(-.8, .8, 2 * 3 * (h + 2) * 5 * 2, dtype=np.float32).reshape(2, 3, h + 2, 5, 2)
        z = norm(np, u).reshape(2, 3, h + 2, 10)
        actions = torch.from_numpy(z).to('mps')
        with torch.inference_mode(), amp(bundle):
            assert_amp(bundle)
            original = model.rollout({'pixels': p[:, None].expand(-1, 3, -1, -1, -1, -1)},
                actions, history_size=3)['predicted_emb']
            initial = model.encode({'pixels': p})['emb']
            cached = predict_cached(bundle, initial[:, None].expand(-1, 3, -1, -1).reshape(6, h, 192),
                actions.reshape(6, h + 2, 10)).reshape(2, 3, h + 3, 192)
        a, b = original.float().cpu().numpy(), cached.float().cpu().numpy()
        assert a.shape == b.shape == (2, 3, h + 3, 192) and np.allclose(a, b, rtol=RTOL, atol=ATOL)
        assert np.allclose(unnorm(np, norm(np, u)), u, rtol=1e-6, atol=1e-7)
        arrays.update({f'H{h}_raw_u': u, f'H{h}_normalized_blocks': z,
            f'H{h}_original': a, f'H{h}_cached': b, f'H{h}_raw_images': raw,
            f'H{h}_observation_raw_t': np.arange(-(h - 1) * 5, 1, 5, dtype=np.int64),
            f'H{h}_action_block_raw_t': np.arange(-(h - 1) * 5, 11, 5, dtype=np.int64)})
        receipt['cases'].append({'initial_H': h, 'HS': 3, 'F': 3, 'action_blocks': h + 2,
            'output_frames': h + 3, 'max_abs': float(np.abs(a - b).max()),
            'RMSE': float(np.sqrt(((a - b) ** 2).mean())),
            'original_dtype': str(original.dtype), 'cached_dtype': str(cached.dtype)})
    reencoded = bundle['goal_reencoded']
    assert np.allclose(reencoded, bundle['goal_z'], rtol=RTOL, atol=ATOL)
    arrays.update(goal_E_cached=bundle['goal_z'], goal_E_reencoded=reencoded)
    hold = np.zeros((3, 2), dtype=np.float32)
    assert not np.array_equal(norm(np, hold), hold)
    clip_fixture = np.asarray([[-2., 2.], [.4, -.3]], dtype=np.float32)
    arrays.update(clip_input_u=clip_fixture, clipped_u=np.clip(clip_fixture, -1, 1),
                  clipped_actual_z=norm(np, np.clip(clip_fixture, -1, 1)), hold_u=hold, hold_z=norm(np, hold))
    with (out / 'MARKERS.npz').open('xb') as stream:
        np.savez(stream, **arrays)
    receipt.update(passed=True, goal_E_max_abs=float(np.abs(reencoded - bundle['goal_z']).max()),
        file_sha256=sha(out / 'MARKERS.npz'), all_goal_frames_public=True)
    return receipt


def complete_query(bundle, current_images, warm_u, rngs, out, query_index):
    np, torch = bundle['np'], bundle['torch']
    assert current_images.shape == (8, 96, 96, 3) and current_images.dtype == np.uint8
    out = Path(out).resolve()
    out.relative_to(ROOT / 'results')
    assert out.is_dir()
    torch.mps.synchronize()
    cpu, wall = time.process_time(), time.perf_counter()
    current = torch.cat([encode(bundle, image[None, None]) for image in current_images], dim=0)
    mean = np.zeros((8, 3, 2), dtype=np.float32) if warm_u is None else norm(np, warm_u)
    warm = np.zeros((8, 3, 2), dtype=np.float32) if warm_u is None else warm_u.copy()
    std = np.ones_like(mean)
    best_cost = np.full(8, np.inf, dtype=np.float32)
    best_id = np.full(8, -1, dtype=np.int64)
    best_u = np.zeros_like(mean)
    arrays = {'current_RGB': current_images, 'goal_E': bundle['goal_z'], 'initial_mean_z': mean.copy(),
        'warm_candidate_u': warm.copy()}
    for iteration in range(3):
        noise = np.stack([rng.standard_normal((64, 3, 2)) for rng in rngs])
        z = (mean[:, None] + std[:, None] * noise).astype(np.float32)
        u = np.clip(unnorm(np, z), -1, 1)
        u[:, 0] = 0.
        u[:, 1] = warm
        actual_z = norm(np, u)
        blocks = np.repeat(actual_z[..., None, :], 5, axis=-2).reshape(8 * 64, 3, 10)
        with torch.inference_mode():
            initial = current[:, None].expand(-1, 64, -1, -1).reshape(512, 1, 192)
            pred = predict_cached(bundle, initial, torch.from_numpy(blocks).to('mps'))[:, -1]
            terminal = pred.float().cpu().numpy().reshape(8, 64, 192)
        cost = ((terminal[:, :, None] - bundle['goal_z'][None, None]) ** 2).mean(-1).min(-1)
        assert cost.dtype == np.float32 and np.isfinite(cost).all()
        elite = np.argsort(cost, axis=1, kind='stable')[:, :8]
        arrays.update({f'i{iteration}_noise_f64': noise, f'i{iteration}_actual_z': actual_z,
            f'i{iteration}_actual_u': u, f'i{iteration}_terminal_E': terminal,
            f'i{iteration}_cost': cost, f'i{iteration}_elite': elite,
            f'i{iteration}_mean_before': mean.copy(), f'i{iteration}_std_before': std.copy()})
        for slot in range(8):
            chosen = int(np.argmin(cost[slot]))
            if cost[slot, chosen] < best_cost[slot]:
                best_cost[slot] = cost[slot, chosen]
                best_id[slot] = iteration * 64 + chosen
                best_u[slot] = u[slot, chosen]
            values = actual_z[slot, elite[slot]]
            mean[slot] = .25 * mean[slot] + .75 * values.mean(0)
            std[slot] = np.maximum(.1, .25 * std[slot] + .75 * values.std(0))
        arrays.update({f'i{iteration}_mean_after': mean.copy(), f'i{iteration}_std_after': std.copy()})
    shifted = np.concatenate((best_u[:, 1:], best_u[:, -1:]), axis=1)
    first_five = np.repeat(best_u[:, 0, None, :], 5, axis=1)
    arrays.update(selected_u=best_u, selected_global_id=best_id, selected_cost=best_cost,
                  selected_first5_u=first_five, shifted_warm_u=shifted)
    torch.mps.synchronize()
    path = out / f'QUERY_{query_index}.npz'
    with path.open('xb') as stream:
        np.savez(stream, **arrays)
    payload = {'query_index': query_index, 'E_calls_B1': 8, 'predictor_calls': 9,
        'predictor_batch': 512, 'candidate_plans': 1536, 'latent_predicted_block_state_rows': 4608,
        'real_environment_steps': 0, 'arrays_sha256': sha(path), 'memory': memory(torch)}
    payload_path = out / f'QUERY_{query_index}_PAYLOAD.json'
    write(payload_path, payload)
    payload_sha = sha(payload_path)
    # q includes actual NPZ, its hash, memory sample, payload receipt and hash.
    # Only this final timing-sidecar write is a separately disclosed metering tail.
    torch.mps.synchronize()
    receipt = {**payload, 'CPU_s': time.process_time() - cpu,
        'synchronized_wall_s': time.perf_counter() - wall, 'payload_sidecar_sha256': payload_sha,
        'excludes_only_final_timing_sidecar_write': True,
        'metering_tail_fully_charged_by_outer_guard': True}
    write(out / f'QUERY_{query_index}.json', receipt)
    return shifted, receipt, first_five


class SuppliedPolicy:
    """B8 RGB-only policy; no environment/state/active-mask input is accepted."""
    def __init__(self, bundle, outdir, seeds=tuple(range(88000, 88008))):
        self.bundle, self.out = bundle, Path(outdir)
        assert tuple(seeds) == tuple(range(88000, 88008))
        self.rngs = [bundle['np'].random.default_rng(34199 + seed) for seed in seeds]
        self.warm, self.round = None, 0

    def decide(self, images8):
        assert self.round < 60
        self.warm, receipt, chunk = complete_query(self.bundle, images8, self.warm, self.rngs, self.out, self.round)
        self.round += 1
        return chunk, receipt


def actuator_fixture(bundle, out):
    # A finite engineering fixture is separate from the learned query timer.
    np = bundle['np']
    os.environ['SDL_VIDEODRIVER'] = 'dummy'
    os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
    sys.path.append(str(ROOT / '.venv_object/lib/python3.12/site-packages'))
    sys.path.insert(0, str(ROOT / 'src'))
    import cv2
    cv2.setNumThreads(1)
    from jepa4object.environment import PushTStateEnv
    relative = load_source('supplied_relative_executor_fixture', RELATIVE)
    def sink(row):
        with (out / 'PRIVATE_ACTUATOR_EVENTS.jsonl').open('a') as stream:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
            stream.flush()
    result = relative.run_parity_fixture(np, lambda: PushTStateEnv(goal=relative.GOAL), sink)
    assert result['passed'] and result['plan']['native_raw_controls'] == 90
    write(out / 'ACTUATOR_FIXTURE.json', result)
    return result


def run(a, bindings):
    out = BASE / 'probe'
    out.mkdir(exist_ok=False)
    state = {'schema': 'SUPPLIED_LEWM_QUALIFICATION_RESULT_v1', 'workflow_complete': False,
        'qualification_passed': False, 'input_bindings': bindings, 'queries': [],
        'scientific_training_updates': 0, 'task_policy_rollouts': 0, 'closed_episodes_opened': 0,
        'control_execution_admitted': False}
    cpu, wall = time.process_time(), time.perf_counter()
    try:
        event(out, 'cold_load_started')
        bundle = load_bundle(out)
        state['input_bindings'].update(bundle['bindings'])
        state['runtime'] = bundle['runtime']
        state['cold_setup_CPU_s'] = time.process_time() - cpu
        state['cold_setup_wall_s'] = time.perf_counter() - wall
        event(out, 'cold_load_returned')
        np, torch = bundle['np'], bundle['torch']
        rngs = [np.random.default_rng(34199 + 97000 + slot) for slot in range(8)]
        warm = None
        for index in range(4):
            assert time.time() < a['not_after_unix'] - 10
            event(out, 'query_started', query=index)
            warm, receipt, _ = complete_query(bundle, bundle['images'], warm, rngs, out, index)
            state['queries'].append(receipt)
            event(out, 'query_returned', query=index)
        event(out, 'markers_started')
        state['markers'] = markers(bundle, out)
        event(out, 'markers_returned')
        event(out, 'actuator_fixture_started')
        state['actuator_fixture'] = actuator_fixture(bundle, out)
        event(out, 'actuator_fixture_returned')
        after = bundle['feature'].state_identity(torch, bundle['model'])
        write(out / 'STRICT_FULL_STATE_AFTER.json', after)
        assert bundle['before'] == after
        state['full_state_and_BN_bytes_unchanged'] = True
        q = max(x['CPU_s'] for x in state['queries'][1:])
        qw = max(x['synchronized_wall_s'] for x in state['queries'][1:])
        f, fw = state['queries'][0]['CPU_s'], state['queries'][0]['synchronized_wall_s']
        startup = max(8., 1.5 * state['cold_setup_CPU_s'] + max(0., f - q))
        startup_w = max(45., 1.5 * state['cold_setup_wall_s'] + max(0., fw - qw))
        state['forecast'] = {'warm_max_CPU_s': q, 'warm_max_wall_s': qw,
            'batched_group_rounds': 60, 'individual_decisions': 480,
            'batched_predictor_calls': 540, 'latent_block_state_rows': 276480,
            'real_raw_step_upper_count': 2400, 'startup_CPU_s': startup,
            'startup_wall_s': startup_w, 'native_and_other_CPU_allowance_s': 12,
            'native_and_other_wall_allowance_s': 120,
            'control_CPU_s': 90 * q + startup + 12,
            'control_wall_s': 90 * qw + startup_w + 120,
            'conditional_CPU_eligible': 90 * q + startup + 12 <= 180,
            'conditional_wall_eligible': 90 * qw + startup_w + 120 <= 900,
            'allowances_are_forecasts_not_hard_physical_bounds': True,
            'control_audit_10_and_shared_10_CPU_not_added_to_control_180': True}
        state['workflow_complete'] = state['qualification_passed'] = True
        write(out / 'RESULT.json', state)
        return True
    except Exception as error:
        state['error_type'] = type(error).__name__
        state['safe_error_code'] = 'supplied_full_model_qualification_incomplete'
        write(out / 'FAILURE.json', state)
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--admission', required=True)
    args = parser.parse_args()
    try:
        a, bindings = admit(args.admission)
        complete = run(a, bindings)
        print('Supplied full-model engineering stage closed; no task performance was measured.')
        return 0 if complete else 1
    except Exception as error:
        print('Supplied full-model stage did not enter complete qualification:', type(error).__name__)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
