"""Frozen external E and train/val-only unique-frame cache, root guard only.

No backend imports or file access on import. Full checkpoint qualification is
strict before retaining only encoder/projector. This is external pretraining,
not a same-data fresh JEPA result. Every actual call is performed by root.
"""
import argparse
from hashlib import sha256
import importlib.util
import json
import math
import os
import resource
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'results/world_model_visual_transfer_2026_10_19'
OLD = ROOT/'results/world_model_exploration_2026_10_19/official_mps_probe'
CORE = ROOT/'scripts/world_model_exploration_2026_10_19/official_mps_probe.py'
CORE_SHA = '569c04801a11fbd06b64acb893e062ab7a8d3aee94ed486ce388ff793d5c3a6d'
ADAPTER = Path(__file__).with_name('data_adapter.py')
ADAPTER_SHA = '2b44f1f171fc813c3341f71e6d220867a78f4efb62e3644af9d5927f13e5a442'
GOAL_SOURCE = Path(__file__).with_name('goalbank_v2.py')
GOAL_SOURCE_SHA = 'fec47c65b89d12371ccf5565660bfa440bd2b2b77c21c33c4808bdc43b9071ec'
CONFIG_SHA = '2564086e961e7b5c7c04dffc451091115b389a590645ff19653c64fd0bc16e09'
WEIGHT_SHA = '48938400ae3464c9680731287f583a9cb516f55a8ec64ea13a91be47fb15b607'
WEIGHT_BYTES = 72290721
WEIGHTS = BASE/'pretrained_reference/weights.pt'
CODEC = ROOT/'results/world_model_exploration_2026_10_19/c_dependencies'
BATCH_SIZE = 16
MEMORY = 2*1024**3
EXPECTED = {'train':18221,'val':2680}


def digest(path):
    value = sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(1048576),b''):
            value.update(part)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path,value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')


def load(name,path):
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec);sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def tensor_hash(torch,value):
    byte = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    return sha256(byte.numpy().tobytes()).hexdigest()


def state_identity(torch,module):
    return {key:{'shape':list(value.shape),'dtype':str(value.dtype),
                 'sha256':tensor_hash(torch,value)} for key,value in module.state_dict().items()}


def checkpoint_contract(weights_path):
    """Only immutable metadata/bytes here; no checkpoint unpickling."""
    weights_path = Path(weights_path).resolve();weights_path.relative_to(BASE)
    assert weights_path == WEIGHTS.resolve() and not weights_path.is_symlink()
    metadata_path = BASE/'pretrained_reference/METADATA.json'
    config_path = BASE/'pretrained_reference/config.json'
    metadata,config = read(metadata_path),read(config_path)
    assert metadata['passed'] and metadata['repo'] == 'quentinll/lewm-pusht'
    assert metadata['revision'] == '22b330c28c27ead4bfd1888615af1340e3fe9052'
    assert metadata['payload_expected_sha256'] == WEIGHT_SHA and metadata['payload_expected_bytes'] == WEIGHT_BYTES
    assert digest(config_path) == CONFIG_SHA
    pointer_path = BASE/'pretrained_reference/weights.lfs.txt'
    assert pointer_path.read_text().splitlines() == ['version https://git-lfs.github.com/spec/v1',
        'oid sha256:'+WEIGHT_SHA,'size '+str(WEIGHT_BYTES)]
    records = {row['path']:row for row in metadata['records']}
    for path in (config_path,pointer_path):
        row = records[str(path.relative_to(ROOT))]
        assert digest(path) == row['sha256'] and path.stat().st_size == row['bytes']
    # This data-only exact config check maps SWM names to the frozen official
    # constructor, not a Hydra import or a permissive checkpoint key mapping.
    assert config == {
        '_target_':'stable_worldmodel.wm.lewm.LeWM',
        'encoder':{'_target_':'stable_pretraining.backbone.utils.vit_hf','size':'tiny',
            'patch_size':14,'image_size':224,'pretrained':False,'use_mask_token':False},
        'predictor':{'_target_':'stable_worldmodel.wm.lewm.module.Predictor','num_frames':3,
            'input_dim':192,'hidden_dim':192,'output_dim':192,'depth':6,'heads':16,
            'mlp_dim':2048,'dim_head':64,'dropout':.1,'emb_dropout':0.},
        'action_encoder':{'_target_':'stable_worldmodel.wm.lewm.module.Embedder','input_dim':10,'emb_dim':192},
        'projector':{'_target_':'stable_worldmodel.wm.lewm.module.MLP','input_dim':192,
            'output_dim':192,'hidden_dim':2048,'norm_fn':{'_target_':'torch.nn.BatchNorm1d','_partial_':True}},
        'pred_proj':{'_target_':'stable_worldmodel.wm.lewm.module.MLP','input_dim':192,
            'output_dim':192,'hidden_dim':2048,'norm_fn':{'_target_':'torch.nn.BatchNorm1d','_partial_':True}}}
    assert weights_path.stat().st_size == WEIGHT_BYTES and digest(weights_path) == WEIGHT_SHA
    return {str(path.relative_to(ROOT)):digest(path) for path in
        (metadata_path,config_path,pointer_path,weights_path)}


def load_frozen_encoder(weights_path=WEIGHTS,diagnostics=None):
    """Full strict-load -> E only. Returns (E, JSON-safe qualification receipt).

    E.encode_raw96 accepts N,H,W,C uint8 or float255; it uses exactly the
    qualified CPU transform and BF16 MPS eval, returning CPU float32 N,192.
    It does not access an archive, action normalizer, future frame or dynamics.
    Callers must account for loading and all encoding/decision costs.
    """
    cpu,wall = time.process_time(),time.perf_counter()
    assert digest(CORE) == CORE_SHA and digest(ADAPTER) == ADAPTER_SHA
    bindings = checkpoint_contract(weights_path)
    core = load('frozen_E_original_constructor',CORE)
    for key,value in core.THREAD_ENV.items():
        assert os.environ.get(key) == value, 'Runtime thread/offline/fallback contract: '+key
    official = core.verify_inputs()
    assert official.pop('no_data_or_checkpoint_inputs') is True
    sys.path.insert(0,str(core.SOURCE));sys.path.insert(0,str(core.VENDOR))
    import numpy as np
    import torch
    import transformers
    from transformers import ViTConfig,ViTModel
    assert torch.__version__.split('+')[0] == '2.14.0' and transformers.__version__ == '4.57.6'
    assert torch.backends.mps.is_available()
    if torch.get_num_threads() != 1:
        torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.manual_seed(3073);torch.mps.manual_seed(3073)
    recommended = int(torch.mps.recommended_max_memory())
    torch.mps.set_per_process_memory_fraction(min(1.,math.nextafter(MEMORY/recommended,0.)))
    module = core.official_module('frozen_E_official_module',core.SOURCE/'module.py')
    jepa = core.official_module('frozen_E_official_jepa',core.SOURCE/'jepa.py')
    model = core.construct(torch,module,jepa,ViTConfig,ViTModel)
    assert sum(p.numel() for p in model.parameters()) == 18034478
    checkpoint = torch.load(weights_path,map_location='cpu',weights_only=True)
    assert isinstance(checkpoint,dict) and checkpoint and all(isinstance(key,str) for key in checkpoint)
    assert all(isinstance(value,torch.Tensor) for value in checkpoint.values()), 'Only a raw tensor state_dict is admitted'
    expected = model.state_dict()
    missing,unexpected = sorted(set(expected)-set(checkpoint)),sorted(set(checkpoint)-set(expected))
    mismatches = [{'key':key,'checkpoint_shape':list(checkpoint[key].shape),
        'expected_shape':list(expected[key].shape),'checkpoint_dtype':str(checkpoint[key].dtype),
        'expected_dtype':str(expected[key].dtype)} for key in sorted(set(expected)&set(checkpoint))
        if checkpoint[key].shape != expected[key].shape or checkpoint[key].dtype != expected[key].dtype]
    strict = {'missing_keys':missing,'unexpected_keys':unexpected,'shape_or_dtype_mismatches':mismatches,
        'raw_state_dict_only':True,'key_renaming_or_prefix_stripping':False,
        'checkpoint_tensor_count':len(checkpoint),'expected_tensor_count':len(expected),
        'key_shape_dtype_sha256':sha256(json.dumps({key:[list(v.shape),str(v.dtype)]
            for key,v in sorted(checkpoint.items())},sort_keys=True).encode()).hexdigest()}
    if diagnostics is not None:
        write(Path(diagnostics)/'STRICT_LOAD_KEYS.json',strict)
    assert not missing and not unexpected and not mismatches, 'Full checkpoint/official constructor mismatch; stop without fallback'
    assert all(bool(torch.isfinite(value).all()) for value in checkpoint.values())
    loaded = model.load_state_dict(checkpoint,strict=True)
    assert not loaded.missing_keys and not loaded.unexpected_keys
    del checkpoint,expected
    model.requires_grad_(False);model.eval();model.to('mps')
    adapter = load('frozen_E_qualified_adapter',ADAPTER)
    raw = torch.zeros((1,4,96,96,3),dtype=torch.float32)
    raw[0,1].fill_(255.);raw[0,2,:,:,0].fill_(127.);raw[0,3,:,:,2].fill_(63.)
    fixture_pixels = adapter.transform_pixels(torch,raw).to('mps')
    with torch.no_grad(),torch.autocast(device_type='mps',dtype=torch.bfloat16):
        original_z = model.encode({'pixels':fixture_pixels})['emb'].reshape(4,192).float().cpu()
    # Transfer ownership of exactly two modules, then explicitly destroy all
    # pretrained dynamics references before exposing E to a downstream trainer.
    encoder,projector = model.encoder,model.projector
    discarded = {name:sum(p.numel() for p in getattr(model,name).parameters())
                 for name in ('predictor','action_encoder','pred_proj')}
    for name in ('encoder','projector','predictor','action_encoder','pred_proj'):
        delattr(model,name)
    assert not list(model.children());del model

    class FrozenEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__();self.encoder,self.projector = encoder,projector
            self.requires_grad_(False);super().train(False)

        def train(self,mode=True):
            # No caller may accidentally update the frozen projector BN.
            return super().train(False)

        def forward(self,pixels):
            assert tuple(pixels.shape[1:]) == (3,224,224) and pixels.dtype == torch.float32
            assert not self.training and all(not child.training for child in self.modules())
            assert all(not parameter.requires_grad and parameter.dtype == torch.float32 for parameter in self.parameters())
            with torch.no_grad(),torch.autocast(device_type='mps',dtype=torch.bfloat16):
                cls = self.encoder(pixels.to('mps'),interpolate_pos_encoding=True).last_hidden_state[:,0]
                z = self.projector(cls)
            assert z.dtype == torch.bfloat16 and tuple(z.shape) == (len(pixels),192)
            torch.mps.synchronize()
            assert bool(torch.isfinite(z).all())
            assert torch.mps.driver_allocated_memory() < MEMORY
            assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss + torch.mps.driver_allocated_memory() < MEMORY-64*1024**2, "Sampled overlapping peak-RSS plus driver proxy exceeds the small request"
            return z.float().cpu()

        def encode_raw96(self,pixels_raw):
            # Four-slot packing is only the preprocessing function's interface;
            # padded slots are removed before the encoder, never cached samples.
            array = np.asarray(pixels_raw)
            if array.ndim == 3:
                array = array[None]
            assert array.ndim == 4 and tuple(array.shape[1:]) == (96,96,3) and len(array) > 0
            assert array.dtype in (np.dtype('uint8'),np.dtype('float32'))
            array = np.asarray(array,dtype=np.float32)
            assert np.isfinite(array).all() and array.min() >= 0 and array.max() <= 255
            padded = (len(array)+3)//4*4
            pack = np.empty((padded,96,96,3),dtype=np.float32);pack[:len(array)] = array
            if padded > len(array):
                pack[len(array):] = array[-1]
            pixels = adapter.transform_pixels(torch,pack.reshape(-1,4,96,96,3)).reshape(-1,3,224,224)
            return self(pixels[:len(array)]).numpy()

    frozen = FrozenEncoder()
    assert set(dict(frozen.named_children())) == {'encoder','projector'}
    with torch.no_grad():
        z = frozen(fixture_pixels.reshape(4,3,224,224))
    error = float((z-original_z).abs().max())
    assert torch.allclose(z,original_z,rtol=1e-5,atol=1e-6)
    initial_identity = state_identity(torch,frozen)
    assert all(not p.requires_grad for p in frozen.parameters())
    receipt = {'passed':True,'checkpoint_bindings':bindings,'strict_full_model_load':strict,
        'official_source_vendor_identity_only':official,'constructor_source_sha256':CORE_SHA,
        'adapter_source_sha256':ADAPTER_SHA,'external_pretraining_data_unknown_overlap':True,
        'full_model_parameters':18034478,'retained_children':['encoder','projector'],
        'retained_parameters':sum(p.numel() for p in frozen.parameters()),
        'discarded_pretrained_dynamics_parameters':discarded,'pretrained_dynamics_exposed':False,
        'all_parameters_frozen':True,'all_modules_eval':True,'projector_BN_frozen':True,
        'parameters_dtype':'float32','forward_autocast':'mps bfloat16','output_cache_dtype':'float32',
        'fixture_original_encode_vs_E_max_abs_error':error,'fixture_tolerance':{'rtol':1e-5,'atol':1e-6},
        'synthetic_fixture_forward_calls':2,'synthetic_fixture_frames_per_call':4,
        'frozen_state_identity':initial_identity,'body_CPU_s':time.process_time()-cpu,'body_wall_s':time.perf_counter()-wall}
    return frozen,receipt


def split_rows(np,adapter,split):
    rows,episodes,times = [],[],[]
    for episode in sorted(adapter.split[split]):
        begin,end = 0 if episode == 0 else adapter.ends[episode-1],adapter.ends[episode]
        rows.extend(range(begin,end));episodes.extend([episode]*(end-begin));times.extend(range(end-begin))
    assert len(rows) == EXPECTED[split] and len(set(rows)) == len(rows)
    return [np.asarray(value,dtype=np.int64) for value in (rows,episodes,times)]


def build_split(np,torch,adapter,frozen,split,out,events):
    rows,episodes,times = split_rows(np,adapter,split)
    z = np.empty((len(rows),192),dtype=np.float32)
    seen = np.zeros(len(rows),dtype=bool);buffer,positions = [],[]
    def flush():
        if not positions:
            return
        cpu,wall = time.process_time(),time.perf_counter()
        result = frozen.encode_raw96(np.stack(buffer))
        assert result.shape == (len(positions),192) and result.dtype == np.float32
        assert not seen[positions].any()
        z[positions] = result;seen[positions] = True
        row = {'split':split,'actual_frames':len(positions),'cache_positions':[positions[0],positions[-1]+1],
            'preprocess_only_tail_padding':(-len(positions))%4,'encoder_padding_frames':0,
            'CPU_s':time.process_time()-cpu,'wall_s':time.perf_counter()-wall,
            'MPS_current_bytes':torch.mps.current_allocated_memory(),'MPS_driver_bytes':torch.mps.driver_allocated_memory()}
        events.append(row)
        print(json.dumps({'encoded_split':split,'completed':int(seen.sum()),'total':len(rows)}),flush=True)
        buffer.clear();positions.clear()
    def consume(value,locations):
        assert len(locations) == 1 and value.min() >= 0 and value.max() <= 255
        position = locations[0]
        assert position == int(seen.sum())+len(positions), 'Expected chronological, unique frame traversal'
        buffer.append(value.copy());positions.append(position)
        if len(positions) == BATCH_SIZE:
            flush()
    adapter._consume('data/img',rows.tolist(),split,'unique_actual_frame_frozen_E_cache',consume)
    flush();assert seen.all() and np.isfinite(z).all()
    path = out/(split+'.npz')
    with path.open('xb') as stream:
        np.savez_compressed(stream,z=z,raw_row=rows,episode=episodes,t=times)
    with np.load(path,allow_pickle=False) as saved:
        assert set(saved.files) == {'z','raw_row','episode','t'}
        for name,value in (('z',z),('raw_row',rows),('episode',episodes),('t',times)):
            np.testing.assert_array_equal(saved[name],value)
    return {'path':str(path.relative_to(ROOT)),'sha256':digest(path),'frames':len(rows),
        'z_shape':[len(rows),192],'z_dtype':'float32','index_dtype':'int64',
        'raw_row_sha256':sha256(rows.tobytes()).hexdigest(),'each_actual_frame_encoded_once':True,
        'no_window_duplication_or_created_padding_samples':True}


def main(mode,weights_path,codec_target):
    cpu,wall = time.process_time(),time.perf_counter()
    out = BASE/('frozen_encoder_qualification' if mode == 'load' else 'feature_cache_small')
    out.mkdir(parents=True,exist_ok=False)
    adapter,events = None,[]
    try:
        # Set runtime contract before importing NumPy/Torch/transformers.
        assert digest(CORE) == CORE_SHA
        core = load('frozen_E_runtime_contract',CORE)
        for key,value in core.THREAD_ENV.items():
            os.environ[key] = value
        frozen,qualification = load_frozen_encoder(weights_path,out)
        import numpy as np
        import torch
        bindings = {str(path.relative_to(ROOT)):digest(path) for path in
            (Path(__file__),ADAPTER,BASE/'P_HEAD_DATASET_CONTRACT.json',
             ROOT/'reports/world_model_visual_transfer_2026-10-19/P_HEAD_DATASET.md')}
        head_contract = read(BASE/'P_HEAD_DATASET_CONTRACT.json')
        assert head_contract['train_windows_expected'] == 17501 and head_contract['val_windows_expected'] == 2580
        assert head_contract['no_dev_test_features'] and head_contract['world_model_action_dim'] == 15
        assert head_contract['direct_condition_dim'] == 207
        assert head_contract['document_sha256'] == digest(ROOT/'reports/world_model_visual_transfer_2026-10-19/P_HEAD_DATASET.md')
        split_results,goal_result,codec_identity = {},None,None
        if mode == 'cache':
            assert digest(ADAPTER) == ADAPTER_SHA
            module = load('frozen_E_cache_adapter',ADAPTER)
            codec_path = Path(codec_target).resolve();codec_path.relative_to(ROOT)
            codec_identity = core.inventory(codec_path)
            adapter = module.VisualAdapter(codec_target)
            bindings.update(adapter.input_bindings)
            for split in ('train','val'):
                split_results[split] = build_split(np,torch,adapter,frozen,split,out,events)
            assert sum(row['frames'] for row in split_results.values()) == 20901
            assert sum(row['actual_frames'] for row in events) == 20901
            assert all(row['array'] == 'data/img' and row['heldout_rows_used'] == 0 and row['complete'] for row in adapter.ledger)
            goal_folder = BASE/'goalbank_v2';goal_proof = read(goal_folder/'QUALIFICATION.json')
            goal_path = goal_folder/'GOAL_BANK.npz'
            assert goal_proof['passed'] and goal_proof['public_task_template_bank']
            assert digest(GOAL_SOURCE) == GOAL_SOURCE_SHA
            assert goal_proof['source_bindings'][str(GOAL_SOURCE.relative_to(ROOT))] == GOAL_SOURCE_SHA
            assert goal_proof['OpenCV_thread_qualification']['not_a_single_thread_OpenCV_claim'] is True
            assert goal_proof['goal_bank_npz_sha256'] == digest(goal_path)
            bindings.update({str(path.relative_to(ROOT)):digest(path) for path in
                (GOAL_SOURCE,goal_path,goal_folder/'QUALIFICATION.json')})
            with np.load(goal_path,allow_pickle=False) as goal:
                assert set(goal.files) == {'images','goal_pose','pusher_xy','actual_state11'}
                images = goal['images'];assert images.shape == (8,96,96,3) and images.dtype == np.uint8
                goal_z = frozen.encode_raw96(images)
                path = out/'goalbank.npz'
                with path.open('xb') as stream:
                    np.savez_compressed(stream,z=goal_z,template_id=np.arange(8,dtype=np.int64))
                goal_result = {'path':str(path.relative_to(ROOT)),'sha256':digest(path),
                    'z_shape':[8,192],'actual_public_goal_frames':8,'same_E_and_transform':True,
                    'eight_appearances_of_one_goal_not_eight_goal_tasks':True}
            write(out/'ACCESS_LEDGER.json',{'mixed_chunk_raw_exposure_is_not_blindness':True,
                'heldout_rows_used':0,'selected_train_val_frames':20901,'records':adapter.ledger})
        final_state = state_identity(torch,frozen)
        assert final_state == qualification['frozen_state_identity'], 'E parameters/BN buffers changed'
        write(out/'LOAD_QUALIFICATION.json',qualification)
        value = {'passed':True,'mode':mode,'source_sha256':digest(__file__),'input_bindings':bindings,
            'load_qualification_sha256':digest(out/'LOAD_QUALIFICATION.json'),
            'codec_files_sha256':codec_identity,
            'splits':split_results,'goalbank':goal_result,'unique_train_val_frames':20901 if mode == 'cache' else 0,
            'batch_size_max':BATCH_SIZE,'small_memory_request_bytes':MEMORY,'overlapping_peak_RSS_plus_driver_sampled_stop_bytes':MEMORY-64*1024**2,'frozen_state_identity_unchanged':True,
            'BF16_MPS_eval_float32_storage_not_FP32_encoder_equivalence_claim':True,
            'encoder_batch_shape_can_change_rounding_not_information':True,
            'cached_state5_actions_or_224_pixels':False,'new_training_updates':0,'environment_calls':0,
            'new_downloads':0,'dev_test_features_or_statistics':False,'pretrained_dynamics_in_output':False,
            'encoding_batches':events,'phase_fees_are_nested_not_additive':True,
            'body_CPU_s':time.process_time()-cpu,'body_wall_s':time.perf_counter()-wall,
            'outer_guard_adds_complete_loading_encoding_compression_and_exit_cost_once':True}
        if adapter is not None:
            value['access_ledger_sha256'] = digest(out/'ACCESS_LEDGER.json')
        write(out/'CACHE_MANIFEST.json',value)
        print(json.dumps({'passed':True,'mode':mode,'unique_train_val_frames':value['unique_train_val_frames']}),flush=True)
    except BaseException as error:
        if adapter is not None:
            write(out/'PARTIAL_ACCESS_LEDGER.json',{'records':adapter.ledger})
        write(out/'FAILURE.json',{'passed':False,'error':repr(error),'traceback':traceback.format_exc(),
            'source_sha256':digest(__file__),'encoding_batches':events,'new_training_updates':0,
            'body_CPU_s':time.process_time()-cpu,'body_wall_s':time.perf_counter()-wall})
        raise
    finally:
        if adapter is not None:
            adapter.package.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',choices=('load','cache'),default='cache')
    parser.add_argument('--weights',type=Path,default=WEIGHTS)
    parser.add_argument('--codec-target',type=Path,default=CODEC)
    args = parser.parse_args();main(args.mode,args.weights,args.codec_target)
