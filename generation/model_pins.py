"""Recorded checkpoint pins shared by downloads and uncached backend loading."""
from pathlib import Path
import hashlib
import json
import warnings
import re

ROOT = Path(__file__).resolve().parent.parent


def load_pins(path=None):
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError('PyYAML is required to read recorded model pins') from error
    config = yaml.safe_load(Path(path or ROOT / 'configs/systems.yaml').read_text())
    revisions, upstream = {}, {}
    for row in [*config['systems'], *config.get('auxiliary_checkpoints', [])]:
        if row.get('hf_id') and row.get('revision'):
            if not re.fullmatch(r'[0-9a-f]{40}', str(row['revision'])):
                raise ValueError(f'An immutable HF revision is required for {row["code_name"]}')
            revisions[row['hf_id']] = row['revision']
        if row.get('upstream_commit'):
            upstream[row['code_name']] = row['upstream_commit']
    return revisions, upstream


def revision_for(repo_id, path=None):
    return load_pins(path)[0].get(repo_id)


def checkpoint_path(repo_id, local_path):
    local_path = Path(local_path)
    revision = revision_for(repo_id)
    if local_path.exists():
        receipt = local_path / '.itts_download.json'
        if receipt.is_file():
            recorded = json.loads(receipt.read_text())
            if not recorded.get('complete') or recorded.get('repo_id') != repo_id or recorded.get('revision') != revision:
                raise ValueError('Local model download is incomplete or has a different recorded identity')
            files = recorded.get('files', {})
            if not files or any(Path(name).is_absolute() or '..' in Path(name).parts or not (local_path / name).is_file()
                                or (local_path / name).stat().st_size != size for name, size in files.items()):
                raise ValueError('Recorded local model files are missing/changed')
        else:
            warnings.warn(f'Using explicit local weights for {repo_id}; their revision is not independently verified by a download receipt', RuntimeWarning)
        return str(local_path)
    if revision:
        from huggingface_hub import snapshot_download
        return snapshot_download(repo_id=repo_id, revision=revision,
                                 ignore_patterns=['*.msgpack', '*.h5', 'flax_model*', 'tf_model*', 'onnx*'])
    return repo_id


def verify_voxinstruct_files(directory, pins=None):
    pins = pins or json.loads((ROOT / 'configs/manual_checkpoints.json').read_text())['voxinstruct']
    for name, record in pins.items():
        path = Path(directory) / name
        if not path.is_file() or path.stat().st_size != record['bytes']:
            raise ValueError(f'Missing/different-size recorded VoxInstruct checkpoint: {name}')
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
                digest.update(block)
        if digest.hexdigest() != record['sha256']:
            raise ValueError(f'Recorded VoxInstruct checkpoint hash differs: {name}; preserve it and select the intended model explicitly')
    return True
