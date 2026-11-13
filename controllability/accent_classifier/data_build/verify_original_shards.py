"""Check a copy of the original split shards against the released clip lists (source_manifests/splits_6cls)."""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import pyarrow.parquet as pq

BUNDLED_ORIGINAL = Path(__file__).resolve().parent / 'source_manifests/splits_6cls'


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(root, labels, manifests=None, reference=None):
    root = Path(root)
    manifests = Path(manifests or BUNDLED_ORIGINAL)
    reference = Path(reference or BUNDLED_ORIGINAL)
    totals, speakers, hashes = {}, {}, {}
    for split in ['train', 'val', 'test']:
        path = manifests / (split + '.jsonl.gz')
        hashes[split] = file_hash(path)
        if hashes[split] != file_hash(reference / path.name):
            raise ValueError('Supplied manifest differs from the bundled released split list')
        expected = defaultdict(dict)
        with gzip.open(path, 'rt') as handle:
            for line in handle:
                row = json.loads(line)
                name, index = row['final_shard'], row['final_shard_row']
                if row['split'] != split or Path(name).name != name or index in expected[name]:
                    raise ValueError('Invalid/duplicate original final-shard reference')
                expected[name][index] = row
        files = sorted((root / split).glob('*.parquet'))
        if {file.name for file in files} != set(expected):
            raise ValueError(f'Original shard file set differs for {split}')
        total, identities = 0, set()
        for file in files:
            records = expected[file.name]
            parquet = pq.ParquetFile(file)
            if set(records) != set(range(parquet.metadata.num_rows)):
                raise ValueError(f'Original row coverage differs: {split}/{file.name}')
            index = 0
            for batch in parquet.iter_batches(batch_size=128, columns=['audio_bytes', 'cls', 'label', 'client_id', 'dur', 'source']):
                for row in batch.to_pylist():
                    original = records[index]
                    for field, recorded in [('cls', 'class'), ('label', 'label'), ('client_id', 'client_id'), ('source', 'source')]:
                        if row[field] != original[recorded]:
                            raise ValueError(f'Original label/identity mismatch: {split}/{file.name}/{index}')
                    if row['cls'] not in labels or row['label'] != labels.index(row['cls']):
                        raise ValueError('Original shard class order differs from training labels')
                    if not math.isclose(row['dur'], original['dur'], rel_tol=1e-6, abs_tol=1e-6):
                        raise ValueError('Original duration metadata differs')
                    audio = row['audio_bytes']
                    if not audio or hashlib.sha256(audio).hexdigest() != original['audio_sha256']:
                        raise ValueError(f'Original audio hash differs: {split}/{file.name}/{index}')
                    identities.add((row['source'], row['client_id']))
                    index += 1
            total += index
        if not total:
            raise ValueError('Original split is empty')
        totals[split], speakers[split] = total, identities
    if speakers['train'] & speakers['val'] or speakers['train'] & speakers['test'] or speakers['val'] & speakers['test']:
        raise ValueError('Speaker overlap across original splits')
    return {'totals': totals, 'manifest_sha256': hashes, 'matched_bundled_original_lists': True,
            'scope': 'Original final-shard bytes, labels and membership verified'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--splits', type=Path, required=True)
    parser.add_argument('--manifests', type=Path, default=BUNDLED_ORIGINAL)
    args = parser.parse_args()
    labels = ['North America', 'English', 'South Asia', 'Germanic', 'Oceania', 'South African']
    try:
        result = verify(args.splits, labels, args.manifests)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
