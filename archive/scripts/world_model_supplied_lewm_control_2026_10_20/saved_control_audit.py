"""Root-only NumPy/scalar/native-replay audit. Never imports Torch or predicts."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'results/world_model_supplied_lewm_control_2026_10_20'
CONTROL = BASE / 'control'
QUERIES = CONTROL / 'policy_queries'
SERVING = Path(__file__).with_name('serving.py')
SERVING_SHA = '4c8af572d432420722dcc97ac93baebe8068ac0f9c3b66d9aceb08b5bec0ad0b'
PARENT = ROOT / 'results/world_model_supplied_lewm_qualification_2026_10_20/PHASE_CLOSURE.json'
PARENT_SHA = '2cf8041d83ea8f5f8b842359173d0e2b90ea68f2b1c0423865126c78a108adf5'
HELPER = ROOT / 'scripts/world_model_supplied_lewm_qualification_2026_10_20/saved_qualification_audit.py'
HELPER_SHA = 'bd8e2f0ff4c8a46047735d7832d2cc1003b24724d521a325324ed0808cf7c1be'
POLYGON = ROOT / 'scripts/world_model_visual_transfer_2026_10_19/saved_native_polygon_audit.py'
POLYGON_SHA = '39a95a87e266b73d378a9ea5932558cb99d933e275dfec8758d1fcaa983f1db0'
SEEDS = tuple(range(88000, 88008))
GOAL = (256., 256., math.pi / 4)
RESOURCES = {'phase_CPU_s': 200, 'control_CPU_s': 180, 'control_wall_s': 900,
             'audit_CPU_s': 10, 'audit_wall_s': 120, 'global_wall_s': 1200,
             'memory_bytes': 2147483648, 'artifact_bytes': 134217728,
             'OS_available_bytes': 6442450944, 'disk_reserve_bytes': 10737418240}


def bound(name, directory=ROOT):
    p = Path(name)
    if not p.is_absolute():
        p = ROOT / p
    assert not p.is_symlink()
    p = p.resolve(); p.relative_to(directory.resolve())
    assert p.is_file()
    return p


def sha(path):
    value = hashlib.sha256()
    with bound(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(bound(path).read_text())


def lines(path):
    return [json.loads(row) for row in bound(path).read_text().splitlines() if row]


def module(name, path, expected):
    assert sha(path) == expected
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


def source_guard(args):
    gp, fp, ap = (bound(p, BASE) for p in (args.guard, args.freeze, args.admission))
    g, f, a = read(gp), read(fp), read(ap)
    assert g['verified_freeze_hashes'][str(fp.relative_to(ROOT))] == sha(fp)
    for name, expected in f['sources'].items():
        assert sha(name) == expected, name
    assert sha(SERVING) == SERVING_SHA == a['source_sha256']
    assert f['sources'][str(SERVING.relative_to(ROOT))] == SERVING_SHA
    assert f['sources'][str(ap.relative_to(ROOT))] == sha(ap)
    assert a['schema'] == 'SUPPLIED_LEWM_CONTROL_ADMISSION_v1' and a['mode'] == 'control'
    assert a['passed'] is True and a['admitted'] is True and a['outer_guard_required'] is True
    assert a['resource_contract'] == RESOURCES and a['inherited_allocation_CPU_s'] == 2142.979764
    assert bound(a['parent_closure_path']) == PARENT and a['parent_closure_sha256'] == PARENT_SHA == sha(PARENT)
    assert 0 < a['not_after_unix'] - a['not_before_unix'] <= 1200
    assert a['entry_source_sha256'] == sha(SERVING.with_name('entry.py'))
    expected = {'cpu_cap_s': 180, 'wall_cap_s': 900, 'global_cpu_cap_s': 200,
                'global_wall_cap_s': 1200, 'memory_cap_bytes': 2147483648,
                'artifact_cap_bytes': 134217728, 'disk_reserve_bytes': 10737418240}
    assert all(g[key] == value for key, value in expected.items())
    entry = read(BASE / 'ENTRY_control.json')
    assert entry['passed'] is True and entry['available_bytes'] >= 6442450944 and entry['disk_free_bytes'] >= 10737418240
    assert a['not_before_unix'] <= g['started_unix'] <= entry['unix'] <= g['started_unix'] + g['wall_s']
    normal = (g['cpu_accounting_complete'] is True and g['returncode'] in (0, 1)
              and not g['termination_reason'] and not g['live_descendants_after_stop']
              and not g['final_assertion_violations'] and g['charged_total_cpu_s'] <= 180
              and g['wall_s'] <= 900 and g['started_unix'] + g['wall_s'] <= a['not_after_unix'])
    qual = PARENT.parent
    assert sha(qual / 'probe/RESULT.json') == a['qualification_result_sha256']
    assert sha(qual / 'saved_audit/AUDIT.json') == a['qualification_audit_sha256']
    audit = read(qual / 'saved_audit/AUDIT.json')
    assert audit['passed'] and audit['qualification_passed']
    assert audit['forecast']['conditional_CPU_eligible'] is True and audit['forecast']['conditional_wall_eligible'] is True
    return g, a, normal


def query(np, helper, index, receipt, rngs, previous, goal):
    path = QUERIES / f'QUERY_{index}.npz'
    assert read(QUERIES / f'QUERY_{index}.json') == receipt
    payload = QUERIES / f'QUERY_{index}_PAYLOAD.json'
    assert receipt['payload_sidecar_sha256'] == sha(payload)
    assert all(receipt[k] == v for k, v in read(payload).items())
    assert receipt['arrays_sha256'] == sha(path)
    for k, v in {'query_index': index, 'E_calls_B1': 8, 'predictor_calls': 9,
                 'predictor_batch': 512, 'candidate_plans': 1536,
                 'latent_predicted_block_state_rows': 4608, 'real_environment_steps': 0}.items():
        assert receipt[k] == v
    helper.memory_check(receipt['memory'])
    keys = {'current_RGB','goal_E','initial_mean_z','warm_candidate_u','selected_u','selected_global_id',
            'selected_cost','selected_first5_u','shifted_warm_u'}
    for i in range(3):
        keys.update(f'i{i}_{n}' for n in ('noise_f64','actual_z','actual_u','terminal_E','cost','elite',
                    'mean_before','std_before','mean_after','std_after'))
    with np.load(bound(path, QUERIES), allow_pickle=False) as saved:
        assert set(saved.files) == keys
        data = {k: saved[k].copy() for k in keys}
    assert all(not v.dtype.hasobject and np.isfinite(v).all() for v in data.values())
    assert data['current_RGB'].shape == (8,96,96,3) and data['current_RGB'].dtype == np.uint8
    assert np.array_equal(data['goal_E'], goal)
    warm = np.zeros((8,3,2),dtype=np.float32) if previous is None else np.concatenate((previous[:,1:],previous[:,-1:]),axis=1)
    mean = np.zeros_like(warm) if previous is None else helper.norm(np,warm)
    std = np.ones_like(mean)
    assert np.array_equal(data['initial_mean_z'],mean) and np.array_equal(data['warm_candidate_u'],warm)
    best = np.zeros_like(mean); costs = np.full(8,np.inf,dtype=np.float32); ids = np.full(8,-1,dtype=np.int64)
    for i in range(3):
        p=f'i{i}_'; noise=np.stack([rng.standard_normal((64,3,2)) for rng in rngs])
        assert data[p+'noise_f64'].dtype == np.float64 and np.array_equal(data[p+'noise_f64'],noise)
        assert np.array_equal(data[p+'mean_before'],mean) and np.array_equal(data[p+'std_before'],std)
        sampled=(mean[:,None]+std[:,None]*noise).astype(np.float32)
        u=np.clip(helper.unnorm(np,sampled),-1,1);u[:,0]=0.;u[:,1]=warm;z=helper.norm(np,u)
        assert data[p+'actual_u'].dtype==data[p+'actual_z'].dtype==np.float32
        assert np.array_equal(data[p+'actual_u'],u) and np.array_equal(data[p+'actual_z'],z)
        terminal=data[p+'terminal_E'];assert terminal.shape==(8,64,192) and terminal.dtype==np.float32
        cost=((terminal[:,:,None]-goal[None,None])**2).mean(-1).min(-1)
        assert data[p+'cost'].dtype==cost.dtype==np.float32
        assert np.array_equal(data[p+'cost'],cost)
        elite=np.argsort(cost,axis=1,kind='stable')[:,:8];assert np.array_equal(data[p+'elite'],elite)
        for slot in range(8):
            chosen=int(np.argmin(cost[slot]))
            if cost[slot,chosen]<costs[slot]:
                costs[slot]=cost[slot,chosen];ids[slot]=i*64+chosen;best[slot]=u[slot,chosen]
            values=z[slot,elite[slot]]
            mean[slot]=.25*mean[slot]+.75*values.mean(0)
            std[slot]=np.maximum(.1,.25*std[slot]+.75*values.std(0))
        assert np.array_equal(data[p+'mean_after'],mean) and np.array_equal(data[p+'std_after'],std)
    first=np.repeat(best[:,0,None,:],5,axis=1);shift=np.concatenate((best[:,1:],best[:,-1:]),axis=1)
    assert np.array_equal(data['selected_u'],best) and np.array_equal(data['selected_global_id'],ids)
    assert np.array_equal(data['selected_cost'],costs) and np.array_equal(data['selected_first5_u'],first)
    assert np.array_equal(data['shifted_warm_u'],shift)
    hashes=[hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest() for image in data['current_RGB']]
    return best,first,hashes


def trajectories(np, polygon, panel, decisions):
    transitions=lines(CONTROL/'TRANSITIONS.jsonl');private=lines(CONTROL/'PRIVATE_EXECUTOR_EVENTS.jsonl')
    grouped={s:[] for s in SEEDS};closed={s:[] for s in SEEDS};events={s:[] for s in SEEDS}
    for row in transitions:
        assert row['episode_seed'] in grouped and row['slot']==row['episode_seed']-88000
        grouped[row['episode_seed']].append(row)
    for row in private:
        assert row['episode_seed'] in closed and row['slot']==row['episode_seed']-88000
        events[row['episode_seed']].append(row)
        if row['event']=='relative_control_closed':closed[row['episode_seed']].append(row)
    arrays_by_seed={};maxcoverage=0.;total=0
    episodes=panel['episodes'];assert [e['seed'] for e in episodes]==list(SEEDS)
    for slot,ep in enumerate(episodes):
        seed=SEEDS[slot];rows=grouped[seed];n=ep['returned_raw_controls']
        assert ep['initialized'] and ep['closed'] and ep['artifact_saved'] and ep['complete']
        assert 0<=n<=300 and len(rows)==n+1 and [r['raw_t'] for r in rows]==list(range(n+1))
        assert len(closed[seed])==n
        assert [row['event'] for row in events[seed]]==['relative_reset_started','relative_reset_returned']+['relative_control_started','relative_control_closed']*n
        entry=ep['trajectory'];path=CONTROL/entry['path']
        assert path.stat().st_size==entry['bytes'] and sha(path)==entry['sha256']
        with np.load(bound(path,CONTROL),allow_pickle=False) as saved:
            assert set(saved.files)=={'state11','state_raw_t','state_verified','issued_u','actual_target','native_coverage','independent_coverage','goal'}
            data={k:saved[k].copy() for k in saved.files}
        assert data['state11'].shape==(n+1,11) and data['state11'].dtype==np.float64
        assert data['issued_u'].shape==(n,2) and data['issued_u'].dtype==np.float32
        assert data['actual_target'].shape==(n,2) and data['actual_target'].dtype==np.float64
        assert np.array_equal(data['state_raw_t'],np.arange(n+1)) and data['state_verified'].all()
        assert np.array_equal(data['goal'],np.asarray(GOAL))
        hits=[]
        for raw,row in enumerate(rows):
            assert row['verified'] is True and np.isfinite(data['state11'][raw]).all()
            assert np.array_equal(data['state11'][raw],np.asarray(row['state11'],dtype=np.float64))
            coverage,normerror=polygon.coverage_state(row['state11'])
            assert abs(coverage-row['native_coverage'])<1e-8 and abs(coverage-row['independent_coverage'])<1e-8
            assert data['native_coverage'][raw]==row['native_coverage'] and data['independent_coverage'][raw]==row['independent_coverage']
            maxcoverage=max(maxcoverage,abs(coverage-row['native_coverage']))
            assert (coverage>.95)==(row['native_coverage']>.95)==row['success_gt_0_95']
            if coverage>.95:hits.append(raw)
            if raw:
                event=closed[seed][raw-1];u=data['issued_u'][raw-1].astype(np.float64)
                target=data['state11'][raw-1,:2]+100.*u
                assert np.isfinite(u).all() and np.max(np.abs(u))<=1
                assert row['issued_u']==u.tolist() and row['actual_target']==target.tolist()
                assert np.array_equal(data['actual_target'][raw-1],target)
                assert event['raw_t_before']==raw-1 and event['control_steps_after']==raw and event['native_step_returned'] is True
                assert event['expected_native_step_calls']==event['native_step_calls_started']==1 and event['expected_native_substeps']==10
                assert event['no_absolute_target_clip'] is True and event['not_a_planner_observation'] is True
                assert event['before_state11']==data['state11'][raw-1].tolist() and event['after_state11']==row['state11']
                assert event['issued_u']==u.tolist() and event['actual_target']==target.tolist()
            else:assert row['issued_u'] is None and row['actual_target'] is None
        first=min(hits) if hits else None
        assert first==ep['first_hit_raw_t'] and ep['success']==(first is not None)
        assert (first==n) if first is not None else n==300
        arrays_by_seed[seed]=data;total+=n
    counts=[0]*8;last_hashes=[None]*8;render_hashes={s:{} for s in SEEDS}
    assert len(decisions)==len(panel['rounds'])==panel['policy_group_calls_returned']==panel['policy_group_calls_started']<=60
    roundlog=lines(CONTROL/'ROUNDS.jsonl')
    assert [row['event'] for row in roundlog]==['round_started','round_closed']*len(decisions)
    assert roundlog[1::2]==panel['rounds']
    for index,(roundrow,decision) in enumerate(zip(panel['rounds'],decisions)):
        first,hashes=decision;active=[slot for slot,e in enumerate(episodes) if counts[slot]<e['returned_raw_controls']]
        # A zero-control t0 first hit is inactive but receives a cached initial image.
        assert roundrow['round']==index and roundrow['active_slots_private']==active and roundrow['fixed_policy_batch']==8
        binding=roundrow['decision_receipt'];path=CONTROL/binding['path']
        assert path.stat().st_size==binding['bytes'] and sha(path)==binding['sha256']
        receipt=read(path);assert receipt['round']==index and receipt['RGB_sha256']==hashes==roundrow['RGB_sha256']
        assert receipt['policy_receipt']==read(QUERIES/f'QUERY_{index}.json')
        assert roundlog[2*index]['round']==index and roundlog[2*index]['RGB_sha256']==hashes and roundlog[2*index]['active_slots_private']==active
        assert np.array_equal(np.asarray(receipt['actual_u_8x5x2'],dtype=np.float32),first)
        for slot,seed in enumerate(SEEDS):
            if slot in active or last_hashes[slot] is None:render_hashes[seed][counts[slot]]=hashes[slot]
            elif last_hashes[slot] is not None:assert hashes[slot]==last_hashes[slot]
            take=min(5,episodes[slot]['returned_raw_controls']-counts[slot]) if slot in active else 0
            assert roundrow['executed_raw_controls_per_slot'][slot]==take
            if take:assert np.array_equal(arrays_by_seed[seed]['issued_u'][counts[slot]:counts[slot]+take],first[slot,:take])
            counts[slot]+=take
        last_hashes=hashes
    assert counts==[e['returned_raw_controls'] for e in episodes] and total==panel['returned_raw_controls']
    assert panel['complete_panel_successes']==sum(e['success'] for e in episodes)
    assert panel['complete_panel_success_rate']==panel['complete_panel_successes']/8
    return arrays_by_seed,render_hashes,{'episodes':8,'raw_controls':total,'states':total+8,
        'independent_polygon_max_abs_coverage_error':maxcoverage,'queries':len(decisions),
        'complete_panel_successes':panel['complete_panel_successes']}


def replay(np, arrays_by_seed, render_hashes):
    os.environ['SDL_VIDEODRIVER']='dummy';os.environ['PYGAME_HIDE_SUPPORT_PROMPT']='1'
    sys.path.append(str(ROOT/'.venv_object/lib/python3.12/site-packages'));sys.path.insert(0,str(ROOT/'src'))
    import cv2
    cv2.setNumThreads(1)
    from jepa4object.environment import PushTStateEnv
    maxima=0.;renders=0;controls=0
    for seed,data in arrays_by_seed.items():
        env=PushTStateEnv(goal=GOAL)
        try:
            env.reset(seed=seed,goal=GOAL)
            for raw in range(len(data['issued_u'])+1):
                actual=np.asarray(env.state(),dtype=np.float64)
                error=float(np.abs(actual-data['state11'][raw]).max());maxima=max(maxima,error)
                assert np.allclose(actual,data['state11'][raw],rtol=0,atol=1e-8)
                native=float(env.env._get_coverage());assert abs(native-data['native_coverage'][raw])<1e-8
                if raw in render_hashes[seed]:
                    image=np.ascontiguousarray(env.env._render(visualize=False))
                    assert image.shape==(96,96,3) and image.dtype==np.uint8
                    assert hashlib.sha256(image.tobytes()).hexdigest()==render_hashes[seed][raw];renders+=1
                if raw<len(data['issued_u']):
                    u=data['issued_u'][raw].astype(np.float64)
                    p=env.env.agent.position
                    target=np.asarray([float(p.x)+100.*float(u[0]),float(p.y)+100.*float(u[1])],dtype=np.float64)
                    assert np.array_equal(target,data['actual_target'][raw])
                    env.env.step(target);env.control_steps+=1;controls+=1
        finally:env.close()
    assert not any(n.split('.')[0] in ('torch','jax') for n in sys.modules)
    return {'native_replay_executed':True,'raw_controls':controls,'decision_RGB_hashes_checked':renders,
            'state11_max_abs_error':maxima,'state11_atol':1e-8,
            'same_native_physics_backend_not_independent_physical_model':True,
            'replayed_fixed_actions_not_extra_policy_search_or_prediction':True}


def audit(args):
    g,a,normal=source_guard(args)
    base={'schema':'SUPPLIED_LEWM_CONTROL_SAVED_AUDIT_v1','passed':False,'audit_complete':False,
          'forensic_consistency_passed':False,'panel_complete':False,'new_Torch_model_calls':0,
          'producer_guard_charged_CPU_s':g['charged_total_cpu_s'],
          'producer_CPU_accounting_complete':g['cpu_accounting_complete'],'nested_fees_not_added':True}
    serving_path=CONTROL/'SERVING_RESULT.json'
    if not normal or not serving_path.is_file():return {**base,'reason':'outer_stop_or_missing_serving_terminal'}
    serving=read(serving_path)
    assert serving['schema']=='SUPPLIED_LEWM_SERVING_RESULT_v1'
    assert serving['scientific_training_updates']==serving['closed_episodes_opened']==0
    for name,expected in serving['input_bindings'].items():assert sha(name)==expected
    assert serving['input_bindings'][str(bound(args.admission).relative_to(ROOT))]==sha(args.admission)
    if not serving.get('panel_result_binding'):
        return {**base,'passed':True,'audit_complete':True,'forensic_consistency_passed':True,
                'reason':'normal_failed_serving_before_panel','saved_partial_sha256':{str(p.relative_to(ROOT)):sha(p) for p in CONTROL.rglob('*') if p.is_file()}}
    binding=serving['panel_result_binding'];pp=bound(binding['path'],CONTROL)
    assert pp.stat().st_size==binding['bytes'] and sha(pp)==binding['sha256'];panel=read(pp)
    assert panel['schema']=='SUPPLIED_LEWM_FIXED_B8_PANEL_RESULT_v1'
    for name,item in panel['artifacts_before_terminal'].items():
        path=bound(CONTROL/name,CONTROL);assert path.stat().st_size==item['bytes'] and sha(path)==item['sha256']
    if not serving['passed'] or not panel['panel_complete']:
        if not panel['panel_complete']:assert panel['complete_panel_successes'] is None and panel['complete_panel_success_rate'] is None
        return {**base,'passed':True,'audit_complete':True,'forensic_consistency_passed':True,
                'partial_panel_not_zero_failure_rate':True,'saved_partial_sha256':{str(p.relative_to(ROOT)):sha(p) for p in CONTROL.rglob('*') if p.is_file()}}
    assert g['returncode']==0 and serving['workflow_complete'] is True and serving['panel_complete'] is True
    assert panel['passed'] is True and panel['partial'] is False and panel['stop_reason'] is None and not panel['cleanup_errors']
    assert serving['full_state_and_BN_bytes_unchanged'] is True
    for name,expected in serving['full_state_identity_bindings'].items():assert sha(name)==expected
    before=read(QUERIES/'STRICT_FULL_STATE_BEFORE.json');after=read(QUERIES/'STRICT_FULL_STATE_AFTER.json')
    assert before==after and len(before)==303
    assert before==read(PARENT.parent/'probe/STRICT_FULL_STATE_BEFORE.json')
    assert not any(n.split('.')[0] in ('torch','jax') for n in sys.modules)
    assert Path(sys.executable).absolute()==ROOT/'.venv/bin/python'
    import numpy as np
    assert np.__version__=='2.5.3' and Path(np.__file__).resolve().is_relative_to(ROOT/'.venv')
    helper=module('saved_pure_qualification_definitions',HELPER,HELPER_SHA)
    polygon=module('saved_pure_polygon_definitions',POLYGON,POLYGON_SHA)
    fixtures=polygon.fixtures()
    branch=read(ROOT/'results/world_model_visual_transfer_2026_10_19/E_CACHE_BRANCH.json')
    goal_path=ROOT/branch['cache_directory']/'goalbank.npz'
    assert serving['input_bindings'][str(goal_path.relative_to(ROOT))]==sha(goal_path)
    with np.load(bound(goal_path),allow_pickle=False) as saved:goal=saved['z'].copy()
    assert goal.shape==(8,192) and goal.dtype==np.float32
    rngs=[np.random.default_rng(34199+seed) for seed in SEEDS];previous=None;decisions=[]
    n=panel['policy_group_calls_returned']
    for index in range(n):
        receipt=read(QUERIES/f'QUERY_{index}.json');previous,first,hashes=query(np,helper,index,receipt,rngs,previous,goal)
        decisions.append((first,hashes))
    assert {p.name for p in QUERIES.glob('QUERY_*.npz')}=={f'QUERY_{i}.npz' for i in range(n)}
    data,render_hashes,summary=trajectories(np,polygon,panel,decisions)
    native=replay(np,data,render_hashes)
    return {**base,'passed':True,'audit_complete':True,'forensic_consistency_passed':True,'panel_complete':True,
            'summary':summary,'native_replay':native,'polygon_fixtures':fixtures,
            'all_saved_RNG_CEM_cost_elite_warm_and_first5_checked':True,
            '303_before_after_state_hashes_unchanged_not_independent_weight_reload':True,
            'source_fixed_external_package_not_same_data_JEPA_causal_comparison':True}


def main():
    parser=argparse.ArgumentParser()
    for name in ('guard','freeze','admission'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args();out=BASE/'saved_audit/AUDIT.json';assert not out.exists()
    cpu,wall=time.process_time(),time.perf_counter()
    try:result=audit(args)
    except Exception as error:
        result={'schema':'SUPPLIED_LEWM_CONTROL_SAVED_AUDIT_v1','passed':False,'audit_complete':False,
                'forensic_consistency_passed':False,'panel_complete':False,
                'safe_error_code':'saved_control_custody_numeric_or_replay_failed','error_type':type(error).__name__}
    result.update(auditor_source_sha256=sha(__file__),audit_body_CPU_s=time.process_time()-cpu,
                  audit_body_wall_s=time.perf_counter()-wall,audit_body_clock_not_own_outer_fee=True)
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as stream:json.dump(result,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
    print('Supplied-package saved control audit closed; no new learned prediction performed.')
    return 0 if result['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
