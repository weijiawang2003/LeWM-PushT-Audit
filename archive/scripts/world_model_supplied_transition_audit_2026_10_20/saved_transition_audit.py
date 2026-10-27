"""Root-only saved NumPy arithmetic audit; no Torch, codec or native call.

Source authors do not execute this file. Raw numeric rows are checked against
the sealed train/val numeric cache; E rows against the sealed feature cache.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'results/world_model_supplied_transition_audit_2026_10_20'
OUT = BASE / 'saved_audit'
PRODUCER = Path(__file__).with_name('audit_transitions.py')
PRODUCER_SHA = '9dfdaa4088e5f01dd37ca4a2e343f75a9b0e2aee59ce128daf5295b49f510bb2'
PARENT = ROOT / 'results/world_model_supplied_lewm_control_2026_10_20/PHASE_CLOSURE.json'
PARENT_SHA = '69ba6f017872e77668c730bbcf97280a4a9263be279a4dda388e9327b4f71cb8'
DATA = ROOT / 'results/world_model_exploration_2026_10_19/c_public'
VISUAL = ROOT / 'results/world_model_visual_transfer_2026_10_19'
READINESS = ROOT / 'reports/world_model_supplied_transition_audit_2026_10_20/DATA_INTERFACE_READINESS.md'
READINESS_SHA = 'ed6bd4dec6c76ee76d272f4d35493c58729ef1dc0a16d4d61bbf7ba98ea3bf28'
PROTOCOL = READINESS.with_name('TRANSITION_AUDIT_PROTOCOL.md')
PROTOCOL_SHA = '1516c0db7c7e3dd6b2bdef40836411984da227267052093fbc488828b1498589'
DATA_SHA = '5702c13dd48454e3ef8152e2fe2920ff5e16cb3ecd8f40f29b0e37589604263f'
BRANCH_SHA = 'b6f4e36b82e48eee721013f28a3b0e0af0cef536f5bf1377d76677e599150b8b'
CACHE_MANIFEST_SHA = '430bf2bc51f7f73a02787c83e5deca008e43bd52f5ae96fd5c3922fb31dccfde'
CACHE_SHA = {'train':'bdfa73d8fcdae0495b0325ffb0a7e82b204a51528010a02b05c5181a664e6add',
    'val':'c854f4fee3f47d8e979900c243e51844fa842579b40be4f5947134e6baee38bb'}
SALT = 'jepa4m-dp-pusht-20261019:'
MU, SIGMA = (-.0087, .0068), (.2019, .2002)
CASES, ARMS = ((1, 1), (1, 3), (3, 1), (3, 3)), ('true', 'wrong', 'hold')
RESOURCES = {'phase_CPU_s': 40, 'producer_CPU_s': 24, 'producer_wall_s': 120,
    'audit_CPU_s': 8, 'audit_wall_s': 60, 'global_wall_s': 240,
    'artifact_bytes': 134217728, 'memory_bytes': 2147483648,
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
    h = hashlib.sha256()
    with bound(path).open('rb') as stream:
        for b in iter(lambda: stream.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(bound(path).read_text())


def close(a, b):
    assert math.isfinite(float(a)) and math.isfinite(float(b))
    assert math.isclose(float(a), float(b), rel_tol=3e-6, abs_tol=3e-7), (a, b)


def check_bindings(bindings):
    assert isinstance(bindings, dict) and bindings
    for name, expected in bindings.items():
        assert sha(name) == expected, name


def custody(args):
    gp, fp, ap = (bound(x, BASE) for x in (args.guard, args.freeze, args.admission))
    g, f, a = read(gp), read(fp), read(ap)
    assert g['verified_freeze_hashes'][str(fp.relative_to(ROOT))] == sha(fp)
    check_bindings(f['sources'])
    assert f['sources'][str(PRODUCER.relative_to(ROOT))] == PRODUCER_SHA == sha(PRODUCER)
    assert f['sources'][str(ap.relative_to(ROOT))] == sha(ap)
    assert a['schema'] == 'SUPPLIED_TRANSITION_AUDIT_ADMISSION_v1' and a['mode'] == 'producer'
    assert a['admitted'] and a['passed'] and a['outer_guard_required']
    assert a['source_sha256'] == PRODUCER_SHA
    assert a['entry_source_sha256'] == sha(PRODUCER.with_name('entry.py'))
    assert a['resource_contract'] == RESOURCES and a['inherited_allocation_CPU_s'] == 2155.83259
    assert a['parent_closure_sha256'] == PARENT_SHA == sha(PARENT)
    assert bound(a['parent_closure_path']) == PARENT and read(PARENT)['phase_complete']
    assert sha(READINESS) == READINESS_SHA and sha(PROTOCOL) == PROTOCOL_SHA
    assert 0 < a['not_after_unix'] - a['not_before_unix'] <= 240
    limits = {'cpu_cap_s': 24, 'wall_cap_s': 120, 'global_cpu_cap_s': 40,
        'global_wall_cap_s': 240, 'memory_cap_bytes': 2147483648,
        'artifact_cap_bytes': 134217728, 'disk_reserve_bytes': 10737418240}
    assert all(g[k] == v for k, v in limits.items())
    normal = (g['cpu_accounting_complete'] and g['returncode'] in (0, 1)
        and not g['termination_reason'] and not g['live_descendants_after_stop']
        and not g['final_assertion_violations'] and g['charged_total_cpu_s'] <= 24
        and g['wall_s'] <= 120 and g['started_unix'] + g['wall_s'] <= a['not_after_unix'])
    ep = BASE / 'ENTRY_producer.json'
    if ep.exists():
        e = read(ep)
        assert e['passed'] and e['available_bytes'] >= 6442450944 and e['disk_free_bytes'] >= 10737418240
        assert a['not_before_unix'] <= g['started_unix'] <= e['unix'] <= g['started_unix'] + g['wall_s']
    else:
        normal = False
    return g, a, normal


def load_arrays(np, path, identities=None):
    path = bound(path)
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        assert len(infos) == len({i.filename for i in infos})
        assert all(i.filename.endswith('.npy') and '/' not in i.filename for i in infos)
        assert sum(i.file_size for i in infos) <= 64 * 1024**2
    with np.load(path, allow_pickle=False) as saved:
        result = {k: saved[k].copy() for k in saved.files}
    if identities is not None:
        assert set(result) == set(identities)
    for key, v in result.items():
        assert not v.dtype.hasobject and v.flags.c_contiguous and np.isfinite(v).all()
        if identities is not None:
            r = identities[key]
            assert list(v.shape) == r['shape'] and str(v.dtype) == r['dtype']
            assert hashlib.sha256(memoryview(v).cast('B')).hexdigest() == r['sha256']
    return result


def fixed_catalog():
    m = read(DATA / 'DATA_MANIFEST.json')
    assert sha(DATA/'DATA_MANIFEST.json') == DATA_SHA and m['passed'] and len(m['episode_ends']) == 206
    order = sorted(range(206), key=lambda e: hashlib.sha256((SALT + str(e)).encode()).hexdigest())
    split = dict(train=order[:144], val=order[144:164], dev=order[164:184], test=order[184:])
    assert m['split'] == split and m['episode_ends'][-1] == 25650
    assert sha(DATA / 'train_val.npz') == m['npz_sha256']
    return m, split, m['episode_ends']


def selected_rows(split, ends, name):
    episodes = split[name][:32] if name == 'train' else split[name]
    rows, qual = [], []
    for ep in episodes:
        start, stop = (0 if ep == 0 else ends[ep-1]), ends[ep]
        length = stop - start; legal = max(length - 25, 0)
        times = sorted({10, 10 + (legal-1)//2, 10 + legal-1}) if legal else []
        qual.append(dict(episode=ep, length=length, legal_windows=legal, selected_t=times))
        rows.extend((ep, t, start+t) for t in times)
    eligible = [ep for ep in episodes if any(r[0] == ep for r in rows)]
    by_ep = {ep: [i for i,r in enumerate(rows) if r[0] == ep] for ep in eligible}
    donors = []
    for i, (ep, _, _) in enumerate(rows):
        next_ep = eligible[(eligible.index(ep)+1) % len(eligible)]
        donors.append(by_ep[next_ep][by_ep[ep].index(i) % len(by_ep[next_ep])])
    assert rows and len(eligible) >= 2 and all(rows[d][0] != row[0] for row,d in zip(rows, donors))
    return episodes, rows, donors, qual


def check_split(np, name, a, info, old, split, ends):
    episodes, rows, donors, qualification = selected_rows(split, ends, name)
    n = len(rows)
    assert info['episodes_in_existing_split_order'] == episodes and info['episode_qualification'] == qualification
    assert info['windows'] == n
    assert a['episode'].dtype == a['t'].dtype == a['frame_raw_rows'].dtype == a['action_raw_rows'].dtype == a['donor_window_index'].dtype == np.int64
    assert np.array_equal(a['episode'], [r[0] for r in rows]) and np.array_equal(a['t'], [r[1] for r in rows])
    frames = np.array([[r[2]+d for d in (-10,-5,0,5,10,15)] for r in rows], dtype=np.int64)
    actions = np.array([[r[2]+d for d in range(-10,15)] for r in rows], dtype=np.int64)
    assert np.array_equal(a['frame_raw_rows'], frames) and np.array_equal(a['action_raw_rows'], actions)
    assert np.array_equal(a['donor_window_index'], donors)
    # Frozen pre-existing cache uses exact power-of-two division of float32.
    old_lookup = {(int(e),int(t)):i for i,(e,t) in enumerate(zip(old[name+'_episode'],old[name+'_t']))}
    idx = np.array([[old_lookup[(ep,t+d)] for d in range(-10,15)] for ep,t,_ in rows])
    assert np.array_equal(a['absolute_actions'], old[name+'_a'][idx]*np.float32(512))
    assert np.array_equal(a['pusher_xy'], old[name+'_x'][idx,6:8]*np.float32(512))
    u = ((a['absolute_actions'].astype(np.float64)-a['pusher_xy'].astype(np.float64))/100.).astype(np.float32)
    assert a['relative_u'].shape == (n,25,2) and a['relative_u'].dtype == np.float32 and np.array_equal(a['relative_u'], u)
    norm = lambda x: ((x-np.array(MU,dtype=np.float32))/np.array(SIGMA,dtype=np.float32)).astype(np.float32)
    for key, expected in [('past_blocks',norm(u[:,:10]).reshape(n,2,10)),('true_blocks',norm(u[:,10:]).reshape(n,3,10)),('wrong_blocks',norm(u[donors,10:]).reshape(n,3,10)),('hold_blocks',norm(np.zeros_like(u[:,10:])).reshape(n,3,10))]:
        assert a[key].dtype == np.float32 and np.array_equal(a[key], expected),key
    assert info['relative_u_outside_unit_box_coordinate_count'] == int((abs(u)>1).sum()) and info['relative_u_coordinate_count'] == u.size
    assert np.array_equal(np.array(info['relative_u_min'],dtype=np.float32),u.min(axis=(0,1))) and np.array_equal(np.array(info['relative_u_max'],dtype=np.float32),u.max(axis=(0,1)))
    assert sha(VISUAL/'E_CACHE_BRANCH.json') == BRANCH_SHA
    branch = read(VISUAL / 'E_CACHE_BRANCH.json'); cache = ROOT / branch['cache_directory']
    cm = read(cache/'CACHE_MANIFEST.json'); record = cm['splits'][name]
    assert sha(cache/'CACHE_MANIFEST.json') == CACHE_MANIFEST_SHA
    assert cm['passed'] and cm['frozen_state_identity_unchanged'] and sha(cache/(name+'.npz')) == record['sha256'] == CACHE_SHA[name]
    c = load_arrays(np, cache/(name+'.npz'))
    assert set(c) == {'z','raw_row','episode','t'} and c['z'].shape == (record['frames'],192) and c['z'].dtype == np.float32
    assert all(c[k].dtype == np.int64 for k in ('raw_row','episode','t'))
    assert hashlib.sha256(c['raw_row'].tobytes()).hexdigest() == record['raw_row_sha256']
    expected = [(raw,ep,t) for ep in sorted(split[name]) for t,raw in enumerate(range(0 if ep==0 else ends[ep-1],ends[ep]))]
    assert list(zip(c['raw_row'].tolist(),c['episode'].tolist(),c['t'].tolist())) == expected
    lookup = {int(r):i for i,r in enumerate(c['raw_row'])}
    z = c['z'][np.array([[lookup[int(r)] for r in f] for f in frames])]
    assert np.array_equal(a['history_E'], z[:,:3]) and np.array_equal(a['target_E'], z[:,3:])
    used = set(actions.reshape(-1).tolist())
    return rows, used


def check_metrics(np, a, saved):
    """Recompute published FP32 reductions and separately save FP64 pairs."""
    independent = {}
    for h, f in CASES:
        key = f'H{h}_F{f}'; r = saved[key]; pred = a['prediction_'+key]
        n = len(a['episode'])
        assert pred.shape == (n,3,f,192) and pred.dtype == np.float32
        truth = a['target_E'][:,:f]
        p = ((a['history_E'][:,-1:]-truth)**2).mean(axis=-1)
        error = ((pred-truth[:,None])**2).mean(axis=-1)
        assert r['primary_policy_related_case'] == ((h,f)==(1,3))
        assert r['primary_metric'] == 'terminal_at_raw_t_plus_'+str(5*f)+'_MSE'
        assert r['window_weighted_MSE_is_all_future_frames_auxiliary']
        for i, arm in enumerate(ARMS):
            close(r['window_weighted_MSE'][arm],error[:,i].mean())
            close(r['terminal_window_weighted_MSE'][arm],error[:,i,-1].mean())
            assert len(r['per_future_block_window_weighted_MSE'][arm])==f
            for x,y in zip(r['per_future_block_window_weighted_MSE'][arm],error[:,i].mean(axis=0)): close(x,y)
        close(r['persistence_window_weighted_MSE'],p.mean());close(r['persistence_terminal_MSE'],p[:,-1].mean())
        assert len(r['persistence_per_future_block_MSE'])==f
        for x,y in zip(r['persistence_per_future_block_MSE'],p.mean(axis=0)): close(x,y)
        eps = sorted(set(a['episode'].tolist()))
        assert [e['episode'] for e in r['episodes']] == eps
        for ep,e in zip(eps,r['episodes']):
            keep=a['episode']==ep; assert e['windows']==int(keep.sum())
            for terminal in (False,True):
                prefix='terminal_' if terminal else ''
                v=[float(error[keep,i,-1].mean() if terminal else error[keep,i].mean()) for i in range(3)]
                persist=float(p[keep,-1].mean() if terminal else p[keep].mean())
                for arm,x in zip(ARMS,v): close(e[prefix+'MSE'][arm],x)
                close(e['persistence_terminal_MSE' if terminal else 'persistence_MSE'],persist)
                for label,y in zip(('persistence','wrong','hold'),(persist,v[1],v[2])):
                    close(e[prefix+'true_minus_'+label],v[0]-y)
                    ratio=e[prefix+'true_over_'+label]
                    assert (ratio is None)==(y==0)
                    if y: close(ratio,v[0]/y)
        ef=((pred.astype(np.float64)-truth[:,None].astype(np.float64))**2).mean(axis=-1)
        pf=((a['history_E'][:,-1:].astype(np.float64)-truth.astype(np.float64))**2).mean(axis=-1)
        independent[key]={'terminal_window_weighted_MSE_float64':dict(zip(ARMS,ef[:,:,-1].mean(axis=0).tolist())),
            'persistence_terminal_MSE_float64':float(pf[:,-1].mean()),'paired_windows':[]}
        for i,(ep,t) in enumerate(zip(a['episode'],a['t'])):
            v=ef[i,:,-1]; pers=float(pf[i,-1])
            independent[key]['paired_windows'].append({'episode':int(ep),'t':int(t),
                'terminal_MSE':dict(zip(ARMS,v.tolist())),'persistence_terminal_MSE':pers,
                'true_minus_persistence':float(v[0]-pers),'true_minus_wrong':float(v[0]-v[1]),'true_minus_hold':float(v[0]-v[2])})
    return independent


def check_exposure(ledger, used_by_split, split, ends, full):
    covered={}
    membership={ep:name for name,eps in split.items() for ep in eps}
    for row in ledger:
        name,s=row['array'],row['split_used'];assert name in ('data/action','data/state') and s in ('train','val')
        width=2 if name=='data/action' else 5
        start,stop=row['valid_raw_range'];assert start%161==0 and stop==min(start+161,25650)
        selected=row['selected_raw_rows'];assert selected==sorted(set(selected))
        assert all(start<=x<stop and x in used_by_split[s] for x in selected)
        assert row['heldout_rows_used']==0 and row['mixed_chunk_raw_exposure_is_not_blindness']
        exposures=[{'episode':ep,'split':membership[ep],'start':max(start,0 if ep==0 else ends[ep-1]),'stop':min(stop,end)}
            for ep,end in enumerate(ends) if (0 if ep==0 else ends[ep-1])<stop and end>start]
        assert row['passive_raw_episode_split_ranges']==exposures
        assert row['expected_decoded_bytes']==161*width*4 and row['requested_positions']==len(selected)
        if row['complete']:
            assert row['decode_started'] and row['decoded_bytes']==row['expected_decoded_bytes']
        elif full:
            raise AssertionError('Full result contains incomplete numeric read')
        covered.setdefault((s,name),set()).update(selected)
    if full:
        for s in ('train','val'):
            for name in ('data/action','data/state'): assert covered[(s,name)]==used_by_split[s]
    return {'numeric_chunk_records':len(ledger),'passive_dev_test_chunk_records':sum(any(e['split'] in ('dev','test') for e in r['passive_raw_episode_split_ranges']) for r in ledger),
        'heldout_rows_used':0,'mixed_raw_chunk_exposure_not_feature_use_or_blindness':True,'new_codec_calls':0}


def check_events(result, split, ends, full):
    p=BASE/'producer/EVENTS.jsonl'
    actual=[json.loads(line) for line in p.read_text().splitlines()] if p.exists() else []
    expected=[{'phase':'full_load_started'},{'phase':'full_load_returned'}]
    for name in ('train','val'):
        n=len(selected_rows(split,ends,name)[1])
        expected.append({'phase':'split_prepared','split':name,'windows':n})
        for h,f in CASES:
            for begin in range(0,n,32):
                expected.extend([{'phase':'predict_started','split':name,'H':h,'F':f,'begin':begin,'batch':min(32,n-begin)*3},
                    {'phase':'predict_returned','split':name,'H':h,'F':f,'begin':begin}])
        expected.append({'phase':'split_saved','split':name})
    assert len(actual)<=len(expected)
    assert all(row==want for row,want in zip(actual,expected))
    if full: assert actual==expected
    return {'saved_events':len(actual),'expected_complete_events':len(expected),'complete':actual==expected}


def main(args):
    start=time.process_time(),time.perf_counter();OUT.mkdir(parents=True,exist_ok=False)
    output={'schema':'SUPPLIED_TRANSITION_SAVED_AUDIT_v1','passed':False,'prediction_complete':False,
        'new_Torch_model_native_codec_training_calls':0,'author_independent_of_producer':True,
        'all_prediction_checks_are_saved_arithmetic_not_independent_NN_forward':True}
    try:
        g,a,normal=custody(args)
        output.update(producer_guard_CPU_s=g['charged_total_cpu_s'],producer_CPU_accounting_complete=g['cpu_accounting_complete'],nested_fees_not_added=True)
        path=BASE/'producer/RESULT.json'
        assert normal and path.exists(),'Incomplete outer/terminal receipt; no prediction inference'
        r=read(path);assert r['schema']=='SUPPLIED_TRANSITION_AUDIT_v1'
        assert r['training_updates']==r['native_calls']==0 and not r['control_recipe_selection']
        check_bindings(r['input_bindings'])
        if 'codec_files_sha256' in r: check_bindings(r['codec_files_sha256'])
        m,split,ends=fixed_catalog()
        expected_keys={'episode','t','frame_raw_rows','action_raw_rows','history_E','target_E','absolute_actions','pusher_xy','relative_u','donor_window_index','past_blocks','true_blocks','wrong_blocks','hold_blocks'}|{f'prediction_H{h}_F{f}' for h,f in CASES}
        full=r['passed'] and r['prediction_complete'] and g['returncode']==0
        assert set(r['files'])<=set(('train','val')) and set(r['splits'])==set(r['files'])
        output['events']=check_events(r,split,ends,full)
        used_by_split={s:{raw+d for _,_,raw in selected_rows(split,ends,s)[1] for d in range(-10,15)} for s in ('train','val')}
        output['passive_exposure']=check_exposure(r.get('access_ledger',[]),used_by_split,split,ends,full)
        output['splits']={}
        if r['files']:
            assert Path(sys.executable).absolute()==ROOT/'.venv/bin/python'
            import numpy as np
            assert np.__version__=='2.5.3'
            old=load_arrays(np,DATA/'train_val.npz')
            for s,record in r['files'].items():
                p=bound(BASE/record['path'],BASE/'producer');assert sha(p)==record['sha256']
                arr=load_arrays(np,p,record['arrays']);assert set(arr)==expected_keys
                check_split(np,s,arr,r['splits'][s],old,split,ends)
                output['splits'][s]={'windows':len(arr['t']),'episodes':len(set(arr['episode'].tolist())),
                    'metrics_independently_recomputed':check_metrics(np,arr,r['splits'][s]['metrics'])}
        if full:
            assert set(r['files'])=={'train','val'} and r['full_state_unchanged']
            assert r['primary_case']=='H1_F3' and r['diagnostic_cases']==['H1_F1','H3_F1','H3_F3']
            assert r['no_statistical_or_control_advantage_claim'] and r['action_alignment_source_only_not_physics_replay']
            before=read(BASE/'producer/STRICT_FULL_STATE_BEFORE.json');after=read(BASE/'producer/STRICT_FULL_STATE_AFTER.json')
            assert before==after and len(before)==303
            assert before==read(ROOT/'results/world_model_supplied_lewm_qualification_2026_10_20/probe/STRICT_FULL_STATE_BEFORE.json')
            assert [p['name'] for p in r['phases']]==['prepare_train','prepare_val']
            for phase in r['phases']:
                assert all(math.isfinite(phase[k]) and phase[k]>=0 for k in ('CPU_s','wall_s'))
            assert sum(p['CPU_s'] for p in r['phases'])<=r['body_CPU_s']+1e-7
            assert sum(p['wall_s'] for p in r['phases'])<=r['body_wall_s']+1e-7
            output.update(prediction_complete=True,full_state_303_hashes_unchanged_not_weight_reload=True,
                primary_case='H1_F3_terminal_t_plus_15',all_published_metrics_recomputed=True)
        else:
            assert g['returncode']==1 and not r['passed'] and ('failure' in r or 'close_failure' in r)
            output['failure_scope']='Available saved files/events only; incomplete prediction is not negative evidence'
        output.update(passed=True,forensic_consistency_passed=True,
            source_and_result_sha256={str(p.relative_to(ROOT)):sha(p) for p in (PRODUCER,Path(__file__),path,READINESS,DATA/'DATA_MANIFEST.json',DATA/'train_val.npz')})
    except Exception as exc:
        output['failure']={'type':type(exc).__name__,'message':str(exc)}
    output.update(audit_body_CPU_s=time.process_time()-start[0],audit_body_wall_s=time.perf_counter()-start[1],
        audit_body_not_own_outer_fee=True)
    with (OUT/'AUDIT.json').open('x') as stream:
        json.dump(output,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
    return 0 if output['passed'] else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('guard','freeze','admission'): parser.add_argument('--'+name,required=True)
    raise SystemExit(main(parser.parse_args()))
