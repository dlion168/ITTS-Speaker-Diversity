"""Pure-stdlib input/run/shard validation; never imports a classifier or audio model."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import platform
import os
from importlib.metadata import version, PackageNotFoundError

HERE = Path(__file__).resolve().parent
ATTRS = {'accent': ['accent'], 'age': ['age'], 'gender': ['gender'],
         'gender_x_age': ['age', 'gender'], 'emotion': ['emotion']}
VALUES = {'gender': ['female', 'male'], 'age': ['child', 'young_adult', 'middle_aged', 'elderly'],
          'accent': ['north_american', 'british_isles', 'oceanian', 'south_asian', 'south_african', 'germanic'],
          'emotion': ['angry', 'happy', 'neutral', 'sad']}
VALUES['gender_x_age'] = [f'{g}_{a}' for g in VALUES['gender'] for a in VALUES['age']]
ALIASES = {'older': 'elderly', 'English': 'british_isles', 'North America': 'north_american',
           'South Asia': 'south_asian', 'Oceania': 'oceanian', 'South African': 'south_african', 'Germanic': 'germanic'}
MODELS = {'moss-voicegen', 'parler-mini', 'parler-large', 'capspeech', 'qwen3-voicedesign', 'voxinstruct', 'voxcpm2'}


def canonical(value):
    return ALIASES.get(value, value)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def require_accent_checkpoints(root, mode):
    names = ['accent_wavlm', 'accent_whisper'] if mode == 'ensemble' else ['accent_' + mode]
    for name in names:
        path = Path(root) / name / 'best.pt'
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f'Requested accent mode {mode} requires {path}; no silent member fallback')


def read_rows(path):
    with Path(path).open(newline='') as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or 'file' not in reader.fieldnames:
            raise ValueError(f'Missing file column: {path}')
        rows = list(reader)
    if any(None in row or any(v is None for v in row.values()) for row in rows):
        raise ValueError(f'Malformed CSV record: {path}')
    names = [row['file'] for row in rows]
    if any(not name or name != name.strip() for name in names) or len(set(names)) != len(names):
        raise ValueError(f'Empty/duplicate file IDs: {path}')
    return rows


def output_stem(label, shard, nproc):
    stem = 'controllability.' + label.replace('/', '__')
    return stem + (f'.shard{shard}of{nproc}' if nproc > 1 else '')


def source_identity():
    return {name: sha256(HERE / name) for name in
            ['score_controllability.py', 'merge_shards.py', 'scoring_run.py', 'run_scoring.sh']}


def runtime_identity(groups):
    packages = {}
    for name in ['torch', 'torchaudio', 'transformers', 'speechbrain', 'funasr', 'numpy']:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    assets = {}
    if 'accent' in groups:
        root = Path(os.environ.get('ACCENT_CHECKPOINT_ROOT', HERE / 'checkpoint'))
        for name in ['accent_wavlm', 'accent_whisper']:
            path = root / name / 'best.pt'
            assets[name] = {'path': str(path.resolve()), 'bytes': path.stat().st_size,
                            'mtime_ns': path.stat().st_mtime_ns} if path.is_file() else None
    return {'package_versions': packages, 'accent_file_signatures': assets,
            'classifiers_config_sha256': sha256(HERE.parent / 'configs/classifiers.yaml'),
            'release_assets_sha256': sha256(HERE.parent / 'configs/release_assets.json'),
            'scope': 'Package versions and checkpoint file signatures'}


def prepare(run, wav_root, models, groups, nproc, batch, workers, allow_partial=False, resume=False):
    run, wav_root = Path(run).resolve(), Path(wav_root).resolve()
    if nproc < 1 or batch < 1 or workers < 1:
        raise ValueError('NPROC, BATCH and WORKERS must be positive')
    if not models or set(models) - MODELS or len(set(models)) != len(models):
        raise ValueError('Select unique paper models')
    if not groups or set(groups) - set(ATTRS) or len(set(groups)) != len(groups):
        raise ValueError('Select unique scoreable groups')
    tasks = []
    for group in groups:
        for model in models:
            folder = wav_root / model / group
            path = folder / 'controllability_manifest.csv'
            rows = read_rows(path)
            if not rows:
                raise ValueError(f'Empty selected manifest: {path}')
            counts = Counter()
            for row in rows:
                targets = {a: canonical(row.get('target_' + a, '')) for a in ATTRS[group]}
                value = f"{targets['gender']}_{targets['age']}" if group == 'gender_x_age' else targets[group]
                if value not in VALUES[group]:
                    raise ValueError(f'Unsupported/empty target in {path}')
                counts[value] += 1
                audio = folder / row['file']
                if Path(row['file']).is_absolute() or '..' in Path(row['file']).parts or not audio.is_file() or audio.stat().st_size == 0:
                    raise ValueError(f'Missing/unsafe selected audio ID in {path}')
            each = 6000 if group == 'gender_x_age' else 3000
            if not allow_partial and counts != Counter({value: each for value in VALUES[group]}):
                raise ValueError(f'Paper class counts differ in {path}; use explicit partial-input mode only for labelled smoke checks')
            tasks.append({'label': model + '/' + group, 'input_csv': str(path), 'input_sha256': sha256(path),
                          'rows': len(rows), 'attrs': ATTRS[group]})
    plan = {'schema_version': 1, 'new_execution': True, 'paper_counts_for_selected_groups': not allow_partial,
            'models': models, 'groups': groups, 'nproc': nproc, 'batch': batch, 'workers': workers,
            'max_s': 20, 'python': platform.python_version(), 'code_sha256': source_identity(), 'tasks': tasks,
            'runtime_identity': runtime_identity(groups),
            'scope': 'Input IDs, settings and output coverage'}
    destination = run / 'scoring_run.json'
    if resume:
        if not destination.is_file() or json.loads(destination.read_text()) != plan:
            raise ValueError('Resume run settings differ or are missing; choose a fresh SCORING_RUN_DIR')
    else:
        if run.exists():
            raise FileExistsError('Refusing to adopt/overwrite an existing scoring directory; use a fresh SCORING_RUN_DIR')
        run.mkdir(parents=True)
        with destination.open('x') as handle:
            json.dump(plan, handle, indent=2)
            handle.write('\n')
    return plan


def validate_pair(path, metadata_path, expected, attrs, *, input_hash=None, batch=None, label=None, shard=None, max_s=20):
    rows = read_rows(path)
    if [row['file'] for row in rows] != [row['file'] for row in expected]:
        raise ValueError(f'Shard file IDs/order differ from selected input: {path}')
    for row, original in zip(rows, expected):
        for attr in attrs:
            target = canonical(row.get('target_' + attr, ''))
            prediction = canonical(row.get('pred_' + attr, ''))
            if target != canonical(original.get('target_' + attr, '')) or prediction not in VALUES[attr]:
                raise ValueError(f'Missing/conflicting target/prediction in {path}')
    metadata = json.loads(Path(metadata_path).read_text())
    if metadata.get('n_rows') != len(expected) or metadata.get('missing_wav', 0):
        raise ValueError(f'Incomplete shard metadata: {metadata_path}')
    if (label is not None and metadata.get('label') != label) or (shard is not None and metadata.get('shard') != shard):
        raise ValueError(f'Shard identity differs: {metadata_path}')
    if input_hash is not None:
        provenance = metadata.get('input_provenance', {})
        if provenance.get('csv_sha256') != input_hash or provenance.get('batch') != batch or provenance.get('max_s') != max_s or provenance.get('scorer_sha256') != sha256(HERE / 'score_controllability.py'):
            raise ValueError(f'Missing/different shard input metadata: {metadata_path}')
    return rows


def validated_files(run_manifest):
    path = Path(run_manifest)
    plan = json.loads(path.read_text())
    if plan['code_sha256'] != source_identity():
        raise ValueError('Scoring code changed; do not silently reuse the old run')
    if plan.get('runtime_identity') != runtime_identity(plan['groups']):
        raise ValueError('Scoring runtime/model file signatures changed; no summary was replaced')
    files = []
    for task in plan['tasks']:
        source = Path(task['input_csv'])
        if sha256(source) != task['input_sha256']:
            raise ValueError('Scoring input changed after the run was planned')
        rows = read_rows(source)
        for shard in range(plan['nproc']):
            stem = output_stem(task['label'], shard, plan['nproc'])
            csv_path = path.parent / (stem + '_per_utt.csv')
            validate_pair(csv_path, path.parent / (stem + '.json'), rows[shard::plan['nproc']], task['attrs'],
                          input_hash=task['input_sha256'], batch=plan['batch'], label=task['label'], shard=f'{shard}of{plan["nproc"]}')
            files.append(csv_path)
    found = set(path.parent.glob('controllability.*_per_utt.csv'))
    if found != set(files):
        raise ValueError('Unexpected/mixed scoring CSVs in this run directory')
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--wav-root', type=Path, required=True)
    parser.add_argument('--models', nargs='+', required=True)
    parser.add_argument('--groups', nargs='+', required=True)
    parser.add_argument('--nproc', type=int, required=True)
    parser.add_argument('--batch', type=int, required=True)
    parser.add_argument('--workers', type=int, required=True)
    parser.add_argument('--allow-partial-inputs', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    try:
        prepare(args.run_dir, args.wav_root, args.models, args.groups, args.nproc, args.batch, args.workers,
                args.allow_partial_inputs, args.resume)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print('Scoring input plan validated:', args.run_dir / 'scoring_run.json')


if __name__ == '__main__':
    main()
