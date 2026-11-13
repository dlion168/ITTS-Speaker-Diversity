"""Coverage and split-write safeguards tested with small local Parquet fixtures."""
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import wave
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from controllability.accent_classifier.data_build.source_validation import validate_sources, assert_speaker_disjoint, validate_training_splits


class AccentSourceValidationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output = self.root / 'splits'

    def execute(self, *extra):
        return subprocess.run([sys.executable, str(ROOT / 'controllability/accent_classifier/data_build/make_splits.py'), *extra],
                              env={**os.environ, 'ACC_DATA_ROOT': str(self.root / 'data'), 'ACC_SPLITS': str(self.output)},
                              capture_output=True, text=True, timeout=60)

    def write_source(self, relative, invalid=False):
        folder = self.root / 'data' / relative;folder.mkdir(parents=True)
        records = []
        for speaker in range(5):
            for take in range(2):
                buffer = io.BytesIO()
                with wave.open(buffer, 'wb') as audio:
                    audio.setnchannels(1);audio.setsampwidth(2);audio.setframerate(16000)
                    audio.writeframes(b'\x01\x00' * 1600)
                records.append({'audio_bytes': buffer.getvalue(), 'client_id': relative + str(speaker), 'dur': .1,
                                'id': f'{relative}-{speaker}-{take}'})
        table = pa.Table.from_pylist(records)
        if invalid:table = table.drop(['audio_bytes'])
        pq.write_table(table, folder / 'source.parquet')
        return folder

    def complete_sources(self):
        for name in ['north_america', 'english', 'south_asia', 'germanic', 'oceania', 'south_african']:
            self.write_source('cv17_6class/' + name)
        for name in ['extra_sa/afrispeech', 'extra_germanic/voxpopuli_de_nl', 'extra_southasia/l2arctic']:
            self.write_source(name)

    def test_missing_sources_fail_without_output(self):
        self.write_source('cv17_6class/north_america')
        result = self.execute()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Incomplete or incompatible accent sources', result.stderr)
        self.assertFalse(self.output.exists())

    def test_source_schema_is_required(self):
        path = self.write_source('bad', invalid=True)
        with self.assertRaisesRegex(ValueError, 'missing columns'):
            validate_sources({'fixture': path})

    def test_check_only_defaults_to_paper_six_classes(self):
        self.complete_sources()
        result = self.execute('--check-sources')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(json.loads(result.stdout)['sources']), 9)
        self.assertFalse(self.output.exists())

    def test_complete_build_records_exact_source_references(self):
        self.complete_sources()
        result = self.execute()
        self.assertEqual(result.returncode, 0, result.stderr)
        summary = json.loads((self.output / 'manifest_summary.json').read_text())
        self.assertEqual(len(summary['_labels']), 6)
        self.assertTrue(summary['_speaker_disjoint'])
        totals = validate_training_splits(self.output, summary['_labels'])
        self.assertEqual(sum(totals.values()), 90)
        rows = []
        for split in ['train', 'val', 'test']:
            items = [json.loads(line) for line in (self.output / f'manifests/{split}.jsonl').read_text().splitlines()]
            self.assertTrue(items)
            self.assertTrue(all(len(item['audio_sha256']) == 64 and item['source_identifier_field'] == 'id' for item in items))
            rows.extend(items)
        self.assertEqual(len(rows), 90)

    def test_existing_shards_are_preserved(self):
        self.complete_sources()
        target = self.output / 'train/shard-0000.parquet';target.parent.mkdir(parents=True)
        target.write_bytes(b'do not overwrite this existing output')
        result = self.execute()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(target.read_bytes(), b'do not overwrite this existing output')

    def test_speakers_cannot_cross_splits_within_a_source(self):
        row = ('English', 'fixture.parquet', 0, 'speaker1', .1, 'cv17')
        with self.assertRaisesRegex(ValueError, 'Speaker overlap'):
            assert_speaker_disjoint({'train': [row], 'val': [row], 'test': []})
        other = ('English', 'fixture.parquet', 0, 'speaker1', .1, 'different_corpus')
        assert_speaker_disjoint({'train': [row], 'val': [other], 'test': []})

    def test_expected_manifest_counts_must_match(self):
        folder = self.write_source('fixture')
        expected = self.root / 'expected.json';expected.write_text(json.dumps({'sources': {'fixture': {'rows': 99}}}))
        with self.assertRaisesRegex(ValueError, 'row count differs'):
            validate_sources({'fixture': folder}, expected)

    def test_training_rejects_missing_source_inventory(self):
        self.output.mkdir()
        (self.output / 'manifest_summary.json').write_text(json.dumps({'_labels': ['English']}))
        with self.assertRaisesRegex(ValueError, 'coverage'):
            validate_training_splits(self.output, ['English'])

    def test_training_rejects_changed_split_counts(self):
        self.complete_sources()
        result = self.execute()
        self.assertEqual(result.returncode, 0, result.stderr)
        path = self.output / 'manifest_summary.json'
        manifest = json.loads(path.read_text());manifest['_train_total'] += 1
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'row count differs'):
            validate_training_splits(self.output, manifest['_labels'])


if __name__ == '__main__':
    unittest.main()
