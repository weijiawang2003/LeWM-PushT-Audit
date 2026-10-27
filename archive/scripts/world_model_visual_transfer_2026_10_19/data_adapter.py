"""Train/val-only DP pixel/action adapter; no heavy import or I/O on import.

Root runs all numerical/codec calls under its frozen guard. Old scopes are
read-only. Images are kept as original float255 until one explicit conversion.
"""
from collections import defaultdict
import ast
from hashlib import sha256
import json
import math
from pathlib import Path, PurePosixPath
import stat
import struct
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT/'results/world_model_exploration_2026_10_19'
BASE = ROOT/'results/world_model_visual_transfer_2026_10_19'
DATA = OLD/'c_public'
SALT = 'jepa4m-dp-pusht-20261019:'
TV_COMMIT = '268ddfce8e7938226c4ec8ed23fc7b9c46b97316'
TV_FILES = {'torchvision/transforms/v2/functional/_geometry.py':
    '619db678a3318cff253774fc0d13006f2ed263971bcd8ebc8e753fcbdbb8b31d',
    'torchvision/transforms/v2/functional/_misc.py':
    '1f84f5dff3361bb106233f6476df2e62567ed75379cc261ad44a6230285ad4d1'}
IMAGENET_MEAN, IMAGENET_STD = [.485,.456,.406],[.229,.224,.225]
ACTION_EPSILON = 1e-6
ARRAYS = {'data/img':([25650,96,96,3],[161,96,96,3],'<f4'),
          'data/action':([25650,2],[161,2],'<f4'),
          'data/state':([25650,5],[161,5],'<f4')}


def digest(path):
    h = sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):
            h.update(block)
    return h.hexdigest()


def json_hash(value):
    return sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def write_json(path,value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')


def catalog(manifest):
    """No data access. All frame/state/action indices are global raw-row ids."""
    ends = manifest['episode_ends']
    assert len(ends) == 206 and ends[-1] == 25650 and all(type(v) is int for v in ends)
    assert all(a < b for a,b in zip([0]+ends[:-1],ends))
    order = sorted(range(206),key=lambda ep:sha256((SALT+str(ep)).encode()).hexdigest())
    split = {'train':order[:144],'val':order[144:164],'dev':order[164:184],'test':order[184:]}
    assert manifest['split'] == split
    windows = {}
    for name in ('train','val'):
        entries = []
        for ep in sorted(split[name]):
            begin,end = 0 if ep == 0 else ends[ep-1],ends[ep]
            for start in range(max(0,end-begin-19)):
                raw = begin+start
                entries.append({'window_id':len(entries),'episode':ep,'start':start,'raw_start':raw,
                    'raw_frame_indices':[raw+i for i in (0,5,10,15)],
                    'raw_state5_indices':[raw+i for i in (0,5,10,15)],
                    'raw_action_ranges':[[raw+5*i,raw+5*(i+1)] for i in range(4)]})
        windows[name] = entries
    return split,windows


def group_controls(np,actions20):
    """Twenty distinct chronological rows; never repeat a target five times."""
    value = np.asarray(actions20)
    assert value.shape[-2:] == (20,2)
    return value.reshape(*value.shape[:-2],4,10)


def transform_pixels(torch,pixels_raw):
    """Float255 -> unit exactly once -> ImageNet normalize -> CPU AA resize.

This is the floating-tensor arithmetic adaptation of pinned SPT/TV semantics,
not an import/execution of torchvision or a native-render equivalence claim.
"""
    import torch.nn.functional as functional
    raw = torch.as_tensor(pixels_raw,device='cpu')
    assert raw.dtype == torch.float32 and raw.ndim == 5 and tuple(raw.shape[1:]) == (4,96,96,3)
    assert bool(torch.isfinite(raw).all()) and float(raw.min()) >= 0 and float(raw.max()) <= 255
    result = torch.empty((len(raw),4,3,224,224),dtype=torch.float32,device='cpu')
    mean = torch.tensor(IMAGENET_MEAN,dtype=torch.float32).reshape(3,1,1)
    std = torch.tensor(IMAGENET_STD,dtype=torch.float32).reshape(3,1,1)
    for b in range(len(raw)):
        for t in range(4):
            unit = raw[b,t].permute(2,0,1).contiguous().div(255.)
            normalized = unit.sub(mean).div(std)
            result[b,t].copy_(functional.interpolate(normalized.unsqueeze(0),size=[224,224],
                mode='bilinear',align_corners=False,antialias=True).squeeze(0))
    return result


def causal_view(batch):
    """BC input at frame s+10 excludes both future control blocks and frame4."""
    return {'pixels':batch['pixels'][:,:3],'past_action':batch['action'][:,:2],
            'target_action':batch['action'][:,2]}


class VisualAdapter:
    def __init__(self,codec_target,normalizer=None):
        import numpy as np
        self.np,self.normalizer,self.ledger = np,normalizer,[]
        target = Path(codec_target).resolve();target.relative_to(ROOT)
        assert target.is_dir() and (target/'numcodecs').is_dir() and not list(target.glob('numpy*'))
        sys.path.insert(0,str(target))
        import numcodecs
        assert numcodecs.__version__ == '0.15.1'
        numcodecs.blosc.set_nthreads(1)
        self.numcodecs,self.target = numcodecs,target
        self.manifest_path = DATA/'DATA_MANIFEST.json'
        self.manifest = json.loads(self.manifest_path.read_text())
        self.archive = DATA/'PUBLIC_DATA.zip'
        assert self.manifest['passed'] and self.archive.stat().st_size == self.manifest['download']['actual_bytes'] == 30988725
        assert digest(self.archive) == self.manifest['download']['actual_sha256']
        proof_path = OLD/'c_visual_value_qualification/QUALIFICATION.json'
        proof = json.loads(proof_path.read_text())
        assert proof['passed'] and proof['statistics']['train']['classification'] == 'integer_float_0_255'
        assert proof['archive_sha256'] == digest(self.archive) and proof['data_manifest_sha256'] == digest(self.manifest_path)
        self.split,self.windows = catalog(self.manifest)
        self.ends = self.manifest['episode_ends']
        self.membership = {ep:name for name,episodes in self.split.items() for ep in episodes}
        self.reference_bindings = self._references()
        self.input_bindings = {str(self.archive.relative_to(ROOT)):digest(self.archive),
            str(self.manifest_path.relative_to(ROOT)):digest(self.manifest_path),
            str(proof_path.relative_to(ROOT)):digest(proof_path),**self.reference_bindings}
        self.package = zipfile.ZipFile(self.archive)
        members = self.package.infolist();self.names = set();total = 0
        assert len(members) == 829
        for info in members:
            path = PurePosixPath(info.filename)
            assert info.filename and info.filename not in self.names and not path.is_absolute()
            assert '..' not in path.parts and '\\' not in info.filename and '\x00' not in info.filename
            assert not stat.S_ISLNK(info.external_attr >> 16) and not info.flag_bits & 1
            self.names.add(info.filename);total += info.file_size
            assert total <= 1024**3 and info.file_size <= 256*1024**2
        self.prefix = self.manifest['zip_prefix'];assert self.prefix == 'pusht/pusht_cchi_v7_replay.zarr/'
        self.metadata = {}
        for name,(shape,chunks,dtype) in ARRAYS.items():
            member = self.prefix+name+'/.zarray'
            assert self.package.getinfo(member).file_size <= 65536
            value = json.loads(self.package.read(member))
            assert value['zarr_format'] == 2 and value['shape'] == shape and value['chunks'] == chunks
            assert value['dtype'] == dtype and value['order'] == 'C' and value['filters'] is None
            assert value.get('dimension_separator','.') == '.'
            assert value['compressor'] == {'id':'blosc','cname':'zstd','clevel':5,'shuffle':2,'blocksize':0}
            self.metadata[name] = value

    def _references(self):
        folder = BASE/'torchvision_reference'
        path = folder/'REFERENCE_MANIFEST.json';manifest = json.loads(path.read_text())
        assert manifest['passed'] and manifest['commit'] == TV_COMMIT
        bindings = {str(path.relative_to(ROOT)):digest(path)}
        records = {row['name']:row for row in manifest['files']}
        assert set(records) == set(TV_FILES)
        for name,identity in TV_FILES.items():
            source = folder/name
            assert digest(source) == identity == records[name]['raw']['sha256']
            bindings[str(source.relative_to(ROOT))] = identity
        for folder,revision,names,record_key in (
            (OLD/'official_data_reference','8edfeb336732b5f3ce7b8b210d0ba370a09e2cac',['utils.py'],'records'),
            (OLD/'official_preprocessing_reference','ab836bf699a2be712dfcf980a5eb70d36391e876',
             ['stable_pretraining/data/transforms.py','stable_pretraining/data/dataset_stats.py'],'source_records')):
            path = folder/'REFERENCE_MANIFEST.json';reference = json.loads(path.read_text())
            assert reference['passed'] and reference['revision'] == revision
            bindings[str(path.relative_to(ROOT))] = digest(path)
            for name in names:
                path = folder/'raw'/name;relative = str(path.relative_to(ROOT))
                matches = [row for row in reference[record_key] if row['path'] == relative]
                assert len(matches) == 1 and digest(path) == matches[0]['actual_sha256']
                bindings[relative] = digest(path)
        stats_path = OLD/'official_preprocessing_reference/raw/stable_pretraining/data/dataset_stats.py'
        assignments = [node for node in ast.parse(stats_path.read_text()).body if isinstance(node,ast.Assign)
            and any(isinstance(target,ast.Name) and target.id == 'ImageNet' for target in node.targets)]
        assert len(assignments) == 1 and isinstance(assignments[0].value,ast.Call)
        stats = {item.arg:ast.literal_eval(item.value) for item in assignments[0].value.keywords}
        assert stats == {'mean':IMAGENET_MEAN,'std':IMAGENET_STD}
        return bindings

    def _episode(self,row):
        import bisect
        return bisect.bisect_right(self.ends,row)

    def _consume(self,name,rows,split,purpose,callback):
        np = self.np
        assert name in ARRAYS and split in ('train','val')
        rows = [int(v) for v in rows]
        assert all(0 <= row < 25650 and self.membership[self._episode(row)] == split for row in rows)
        grouped = defaultdict(dict)
        for position,row in enumerate(rows):
            grouped[row//161].setdefault(row,[]).append(position)
        meta = self.metadata[name];chunks = meta['chunks']
        expected = math.prod(chunks)*4;assert expected <= 18*1024**2
        codec = self.numcodecs.get_codec(meta['compressor'])
        for index,row_positions in sorted(grouped.items()):
            cpu,wall = time.process_time(),time.perf_counter()
            suffix = '.'.join([str(index)]+['0']*(len(chunks)-1))
            member = self.prefix+name+'/'+suffix
            info = self.package.getinfo(member);assert info.file_size <= 18*1024**2
            encoded = self.package.read(member)
            start,stop = index*161,min((index+1)*161,25650)
            exposures = []
            for ep,end in enumerate(self.ends):
                begin = 0 if ep == 0 else self.ends[ep-1]
                if begin < stop and end > start:
                    exposures.append({'episode':ep,'split':self.membership[ep],
                        'start':max(start,begin),'stop':min(stop,end)})
            record = {'array':name,'member':member,'purpose':purpose,'split_used':split,
                'encoded_bytes':len(encoded),'encoded_sha256':sha256(encoded).hexdigest(),
                'valid_raw_range':[start,stop],'passive_raw_episode_split_ranges':exposures,
                'selected_raw_rows':sorted(row_positions),'requested_positions':sum(map(len,row_positions.values())),
                'expected_decoded_bytes':expected,'decoded_bytes':None,'decode_started':False,'complete':False,
                'heldout_rows_used':0,'mixed_chunk_raw_exposure_is_not_blindness':True}
            self.ledger.append(record)
            assert len(encoded) == info.file_size and len(encoded) >= 16
            raw_bytes,block_bytes,encoded_bytes = struct.unpack_from('<III',encoded,4)
            assert raw_bytes == expected and encoded_bytes == len(encoded) and 0 < block_bytes <= raw_bytes
            record['decode_started'] = True
            decoded = codec.decode(encoded);record['decoded_bytes'] = memoryview(decoded).nbytes
            assert record['decoded_bytes'] == expected
            block = np.frombuffer(decoded,dtype='<f4').reshape(chunks)
            for row,positions in row_positions.items():
                selected = block[row-start]
                assert np.isfinite(selected).all()
                callback(selected,positions)
            record.update({'complete':True,'CPU_s':time.process_time()-cpu,'wall_s':time.perf_counter()-wall})
            del selected,block,decoded,encoded

    def rows(self,name,rows,split,purpose):
        assert name in ('data/action','data/state')
        output = self.np.empty((len(rows),ARRAYS[name][0][1]),dtype=self.np.float32)
        def consume(value,positions):
            for position in positions:
                output[position] = value
        self._consume(name,rows,split,purpose,consume)
        return output

    def fit_action_normalizer(self):
        np = self.np
        indices = [row for ep in sorted(self.split['train']) for row in range(
            0 if ep == 0 else self.ends[ep-1],self.ends[ep])]
        action = self.rows('data/action',indices,'train','train_only_action_normalizer_all_raw_rows')
        assert ((action >= 0) & (action <= 512)).all()
        values = action.astype(np.float64)
        mean,std = values.mean(axis=0),values.std(axis=0,ddof=1)
        scale = np.maximum(std,ACTION_EPSILON)
        self.normalizer = {'mean':mean.tolist(),'sample_std':std.tolist(),'scale':scale.tolist(),
            'ddof':1,'epsilon':ACTION_EPSILON,'epsilon_clamped_dimensions':(std < ACTION_EPSILON).tolist(),
            'raw_rows':len(indices),'raw_row_indices_sha256':json_hash(indices),'split':'train',
            'fit_unique_raw_rows_not_overlapping_window_weights':True,'accumulation_dtype':'float64',
            'departure_from_original_helper':'Only fixed 144 train episodes; not the entire dataset column.'}
        return self.normalizer

    def raw_batch(self,split,window_ids):
        np = self.np
        assert split in ('train','val') and self.normalizer is not None
        window_ids = [int(v) for v in window_ids]
        assert window_ids and all(0 <= v < len(self.windows[split]) for v in window_ids)
        selected = [self.windows[split][v] for v in window_ids]
        pixels = np.empty((len(selected),4,96,96,3),dtype=np.float32)
        frames = [row for window in selected for row in window['raw_frame_indices']]
        def consume(value,positions):
            assert value.min() >= 0 and value.max() <= 255
            for position in positions:
                pixels[position//4,position%4] = value
        self._consume('data/img',frames,split,'raw_batch_pixels_float255_no_resize',consume)
        rows = [row for window in selected for row in range(window['raw_start'],window['raw_start']+20)]
        actions20 = self.rows('data/action',rows,split,'twenty_real_consecutive_controls_per_window').reshape(-1,20,2)
        assert ((actions20 >= 0) & (actions20 <= 512)).all()
        mean,scale = np.asarray(self.normalizer['mean']),np.asarray(self.normalizer['scale'])
        assert self.normalizer['ddof'] == 1 and self.normalizer['split'] == 'train' and (scale >= ACTION_EPSILON).all()
        normalized = ((actions20.astype(np.float64)-mean)/scale).astype(np.float32)
        return {'pixels_raw':pixels,'action':group_controls(np,normalized),
            'window_ids':np.asarray(window_ids,dtype=np.int64),'metadata':selected}

    def batch(self,split,window_ids):
        import torch
        raw = self.raw_batch(split,window_ids)
        return {'pixels':transform_pixels(torch,raw['pixels_raw']),
            'action':torch.from_numpy(raw['action']),'window_ids':raw['window_ids'],'metadata':raw['metadata']}

    def close(self):
        self.package.close()
