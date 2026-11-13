"""Original-shard path works without raw-mirror inventory; tiny synthetic bytes only."""
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import pyarrow as pa
import pyarrow.parquet as pq
from controllability.accent_classifier.data_build.verify_original_shards import verify


class OriginalShardTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.manifests = self.root / 'manifests';self.manifests.mkdir()
        self.labels = ['English']
        for split in ['train', 'val', 'test']:
            folder = self.root / 'shards' / split;folder.mkdir(parents=True)
            row = {'audio_bytes': b'synthetic-' + split.encode(), 'cls': 'English', 'label': 0,
                   'client_id': split + '-speaker', 'dur': .1, 'source': 'cv17'}
            pq.write_table(pa.Table.from_pylist([row]), folder / 'shard-0000.parquet')
            original = {'split': split, 'class': 'English', 'label': 0, 'client_id': row['client_id'],
                        'dur': .1, 'source': 'cv17', 'final_shard': 'shard-0000.parquet', 'final_shard_row': 0,
                        'audio_sha256': hashlib.sha256(row['audio_bytes']).hexdigest(), 'source_identifier': None}
            with gzip.open(self.manifests / (split + '.jsonl.gz'), 'wt') as handle:handle.write(json.dumps(original) + '\n')

    def test_exact_shards_can_be_verified_without_source_inventory(self):
        result = verify(self.root / 'shards', self.labels, self.manifests, self.manifests)
        self.assertEqual(result['totals'], {'train': 1, 'val': 1, 'test': 1})
        self.assertTrue(result['matched_bundled_original_lists'])

    def test_changed_audio_cannot_be_claimed_as_original(self):
        path = self.root / 'shards/train/shard-0000.parquet'
        row = pq.read_table(path).to_pylist()[0];row['audio_bytes'] = b'changed'
        pq.write_table(pa.Table.from_pylist([row]), path)
        with self.assertRaisesRegex(ValueError, 'audio hash differs'):
            verify(self.root / 'shards', self.labels, self.manifests, self.manifests)

    def test_untrusted_manifest_does_not_match_released_reference(self):
        reference = self.root / 'reference';reference.mkdir()
        for file in self.manifests.iterdir():(reference / file.name).write_bytes(file.read_bytes())
        (self.manifests / 'train.jsonl.gz').write_bytes(b'not the released list')
        with self.assertRaisesRegex(ValueError, 'differs from the bundled'):
            verify(self.root / 'shards', self.labels, self.manifests, reference)
