"""Complete delivered CV17 metadata linkage; no source audio or original mirror needed."""
import csv
import gzip
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class CV17SidecarTests(unittest.TestCase):
    def test_all_final_cv17_hashes_have_consistent_sidecar_metadata(self):
        folder = ROOT / 'controllability/accent_classifier/data_build/source_manifests'
        with (folder / 'cv17_utt_map.csv').open() as handle:rows = list(csv.DictReader(handle))
        mapping = {row['audio_sha256']: row for row in rows}
        self.assertEqual(len(mapping), len(rows))
        self.assertTrue(all(row['cv_sentence'] for row in rows))
        count = 0
        for split in ['train', 'val', 'test']:
            with gzip.open(folder / 'splits_6cls' / (split + '.jsonl.gz'), 'rt') as handle:
                for line in handle:
                    original = json.loads(line)
                    if original['source'] != 'cv17':
                        continue
                    count += 1
                    row = mapping[original['audio_sha256']]
                    self.assertEqual(row['client_id'], original['client_id'])
                    self.assertEqual(row['cls'], original['class'])
        self.assertEqual(count, 115443)
        self.assertEqual(count, len(mapping))
