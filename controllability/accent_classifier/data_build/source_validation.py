"""Fail-fast coverage/schema checks for the declared accent-training sources."""
import hashlib
import json
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq


def required_sources(mirror, regions, extras):
    result = {}
    for region in regions:
        slug = region.lower().replace(' ', '_')
        result['cv17/' + slug] = Path(mirror) / slug
        for name, path in extras.get(region, []):
            if name in result:
                raise ValueError(f'Duplicate source key: {name}')
            result[name] = Path(path)
    return result


def validate_sources(sources, expected_manifest=None):
    expected = None
    if expected_manifest:
        expected = json.loads(Path(expected_manifest).read_text()).get('sources')
        if not isinstance(expected, dict) or set(expected) != set(sources):
            raise ValueError('The expected source manifest must cover exactly every requested source')
    inventory, problems = {}, []
    for name, directory in sources.items():
        paths = sorted(directory.glob('*.parquet')) if directory.is_dir() else []
        if not paths:
            problems.append(f'{name}: missing nonempty Parquet source directory')
            continue
        files = []
        for path in paths:
            parquet = pq.ParquetFile(path)
            schema = parquet.schema_arrow
            needed = {'audio_bytes', 'client_id', 'dur'}
            if needed - set(schema.names):
                problems.append(f'{name}/{path.name}: missing columns {sorted(needed - set(schema.names))}')
                continue
            if not (pa.types.is_binary(schema.field('audio_bytes').type) or pa.types.is_large_binary(schema.field('audio_bytes').type)):
                problems.append(f'{name}/{path.name}: audio_bytes must be binary')
                continue
            if parquet.metadata.num_rows <= 0:
                problems.append(f'{name}/{path.name}: source has zero rows')
                continue
            files.append({'file': path.name, 'rows': parquet.metadata.num_rows, 'bytes': path.stat().st_size})
        count = sum(item['rows'] for item in files)
        inventory[name] = {'rows': count, 'files': files}
        if expected is not None:
            record = expected[name]
            if not isinstance(record, dict) or record.get('rows') != count:
                problems.append(f'{name}: row count differs from the provided manifest')
            hashes = record.get('files', {}) if isinstance(record, dict) else {}
            if hashes:
                if set(hashes) != {path.name for path in paths}:
                    problems.append(f'{name}: file set differs from the expected source hashes')
                for path in paths:
                    if path.name in hashes:
                        h = hashlib.sha256()
                        with path.open('rb') as handle:
                            for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
                                h.update(block)
                        if h.hexdigest() != hashes[path.name]:
                            problems.append(f'{name}/{path.name}: source SHA-256 mismatch')
    if problems:
        raise ValueError('Incomplete or incompatible accent sources; no split output was written:\n' + '\n'.join(problems))
    return inventory


def assert_speaker_disjoint(pooled):
    # source is part of the namespace; identifiers from different corpora must
    # not be falsely treated as the same person without an identity mapping.
    identities = {split: {(row[5], row[3]) for row in rows} for split, rows in pooled.items()}
    for left, right in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        if identities[left] & identities[right]:
            raise ValueError(f'Speaker overlap between {left} and {right}; refusing to write split shards')


def validate_training_splits(root, expected_labels, original_manifests=None):
    """Reject partial or unvalidated split trees before loading classifier weights."""
    if original_manifests is not None:
        if __package__:
            from .verify_original_shards import verify
        else:
            from verify_original_shards import verify
        return verify(root, expected_labels, original_manifests)['totals']
    root = Path(root)
    path = root / 'manifest_summary.json'
    if not path.is_file():
        raise ValueError('Missing split manifest_summary.json; build and validate the declared sources before training')
    manifest = json.loads(path.read_text())
    if manifest.get('_labels') != list(expected_labels):
        raise ValueError('Training label order differs from the split manifest')
    inventory = manifest.get('_source_inventory', {})
    base = {'cv17/' + value for value in ['north_america', 'english', 'south_asia', 'germanic', 'oceania', 'south_african']}
    needed = base | {'afrispeech', 'voxpopuli', 'l2arctic'}
    if set(inventory) != needed or any(not item.get('rows', 0) for item in inventory.values()):
        raise ValueError('Split source coverage is unrecorded or incomplete; partial shards cannot stand in for the paper dataset')
    pooled, totals = {}, {}
    for split in ['train', 'val', 'test']:
        files = sorted((root / split).glob('*.parquet'))
        if not files:
            raise ValueError(f'Missing training split: {split}')
        counts, identities = {}, []
        for file in files:
            table = pq.ParquetFile(file).read(columns=['cls', 'label', 'client_id', 'source'])
            for row in table.to_pylist():
                cls, label = row['cls'], row['label']
                if cls not in expected_labels or label != expected_labels.index(cls):
                    raise ValueError(f'Class/label mismatch in {split}')
                if not row['client_id'] or not row['source']:
                    raise ValueError(f'Missing speaker/source identity in {split}')
                counts[cls] = counts.get(cls, 0) + 1
                identities.append((cls, None, None, row['client_id'], None, row['source']))
        for cls in expected_labels:
            if not counts.get(cls) or counts[cls] != manifest.get(cls, {}).get(split):
                raise ValueError(f'Observed class counts differ from the split manifest: {split}/{cls}')
        totals[split] = sum(counts.values())
        if totals[split] != manifest.get('_' + split + '_total'):
            raise ValueError(f'Observed row count differs from the split manifest: {split}')
        pooled[split] = identities
    assert_speaker_disjoint(pooled)
    return totals
