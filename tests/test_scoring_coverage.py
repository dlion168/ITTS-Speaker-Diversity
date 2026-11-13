"""Shard completeness, uniqueness and input binding without classifier execution."""
import ast
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from controllability.scoring_run import prepare, output_stem, sha256, validated_files, require_accent_checkpoints


class ScoringCoverageTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.run = self.root / 'run'
        folder = self.root / 'audio/capspeech/gender'
        folder.mkdir(parents=True)
        self.source = folder / 'controllability_manifest.csv'
        self.source.write_text('file,target_gender\na.wav,female\nb.wav,male\n')
        for name in ['a.wav', 'b.wav']:(folder / name).write_bytes(b'presence fixture')
        self.plan = prepare(self.run, self.root / 'audio', ['capspeech'], ['gender'], 2, 8, 1, allow_partial=True)

    def write_shard(self, shard):
        stem = output_stem('capspeech/gender', shard, 2)
        name, gender = [('a.wav', 'female'), ('b.wav', 'male')][shard]
        (self.run / (stem + '_per_utt.csv')).write_text(f'file,target_gender,pred_gender\n{name},{gender},{gender}\n')
        metadata = {'label': 'capspeech/gender', 'shard': f'{shard}of2', 'n_rows': 1, 'missing_wav': 0,
                    'input_provenance': {'csv_sha256': sha256(self.source), 'batch': 8, 'max_s': 20,
                                         'scorer_sha256': sha256(ROOT / 'controllability/score_controllability.py')}}
        (self.run / (stem + '.json')).write_text(json.dumps(metadata))

    def merge(self, *extra):
        return subprocess.run([sys.executable, str(ROOT / 'controllability/merge_shards.py'),
                               '--results_dir', str(self.run), '--out', str(self.root / 'summary.json'), *extra],
                              capture_output=True, text=True)

    def test_complete_plan_and_merge_preserve_expected_counts(self):
        self.write_shard(0);self.write_shard(1)
        self.assertEqual(len(validated_files(self.run / 'scoring_run.json')), 2)
        result = self.merge('--run-manifest', str(self.run / 'scoring_run.json'))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads((self.root / 'summary.json').read_text())['capspeech/gender']['n_rows'], 2)

    def test_incomplete_legacy_shards_are_rejected(self):
        self.write_shard(0)
        result = self.merge()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Incomplete/invalid shard', result.stderr)
        self.assertFalse((self.root / 'summary.json').exists())

    def test_complete_plus_sharded_results_are_rejected(self):
        self.write_shard(0);self.write_shard(1)
        (self.run / 'controllability.capspeech__gender_per_utt.csv').write_text('file,target_gender,pred_gender\na.wav,female,female\n')
        result = self.merge()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Mixed complete/sharded', result.stderr)

    def test_duplicate_sample_in_two_shards_is_rejected(self):
        self.write_shard(0);self.write_shard(1)
        path = self.run / (output_stem('capspeech/gender', 1, 2) + '_per_utt.csv')
        path.write_text('file,target_gender,pred_gender\na.wav,female,female\n')
        result = self.merge()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('duplicate sample', result.stderr)

    def test_changed_input_cannot_resume_or_merge(self):
        self.write_shard(0);self.write_shard(1)
        self.source.write_text(self.source.read_text().replace('b.wav,male', 'b.wav,female'))
        with self.assertRaisesRegex(ValueError, 'input changed'):
            validated_files(self.run / 'scoring_run.json')
        with self.assertRaisesRegex(ValueError, 'settings differ'):
            prepare(self.run, self.root / 'audio', ['capspeech'], ['gender'], 2, 8, 1, allow_partial=True, resume=True)

    def test_changed_batch_cannot_resume(self):
        with self.assertRaisesRegex(ValueError, 'settings differ'):
            prepare(self.run, self.root / 'audio', ['capspeech'], ['gender'], 2, 16, 1, allow_partial=True, resume=True)

    def test_stale_extra_group_is_not_merged_into_plan(self):
        self.write_shard(0);self.write_shard(1)
        (self.run / 'controllability.voxcpm2__age_per_utt.csv').write_text('file,target_age,pred_age\nx.wav,child,child\n')
        with self.assertRaisesRegex(ValueError, 'Unexpected/mixed'):
            validated_files(self.run / 'scoring_run.json')

    def test_paper_counts_are_required_without_explicit_partial_mode(self):
        with self.assertRaisesRegex(ValueError, 'Paper class counts'):
            prepare(self.root / 'strict', self.root / 'audio', ['capspeech'], ['gender'], 2, 8, 1)

    def test_malformed_output_keeps_previous_summary(self):
        self.write_shard(0);self.write_shard(1)
        output = self.root / 'summary.json';output.write_text('preserve previous\n')
        (self.run / (output_stem('capspeech/gender', 1, 2) + '_per_utt.csv')).write_text('file,target_gender,pred_gender\nb.wav,male\n')
        result = self.merge()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(output.read_text(), 'preserve previous\n')

    def test_ensemble_cannot_silently_fall_back_to_one_member(self):
        root = self.root / 'checkpoints'
        (root / 'accent_wavlm').mkdir(parents=True)
        (root / 'accent_wavlm/best.pt').write_bytes(b'presence-only weights')
        require_accent_checkpoints(root, 'wavlm')
        with self.assertRaisesRegex(FileNotFoundError, 'no silent member fallback'):
            require_accent_checkpoints(root, 'ensemble')
        (root / 'accent_whisper').mkdir()
        (root / 'accent_whisper/best.pt').write_bytes(b'presence-only weights')
        require_accent_checkpoints(root, 'ensemble')

    def test_invalid_prediction_is_not_accepted_as_completed(self):
        self.write_shard(0);self.write_shard(1)
        path = self.run / (output_stem('capspeech/gender', 1, 2) + '_per_utt.csv')
        path.write_text('file,target_gender,pred_gender\nb.wav,male,unsupported\n')
        with self.assertRaisesRegex(ValueError, 'target/prediction'):
            validated_files(self.run / 'scoring_run.json')

    def test_run_manifest_cannot_be_narrowed_by_legacy_filter(self):
        self.write_shard(0);self.write_shard(1)
        result = self.merge('--run-manifest', str(self.run / 'scoring_run.json'), '--groups', 'age')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('exact model/group scope', result.stderr)

    def test_scorer_resume_checks_original_column_aliases_without_importing_models(self):
        # Execute only the scorer's resume branch. Classifier/torch imports,
        # model construction and the scoring loop are not available here.
        scorer = ROOT / 'controllability/score_controllability.py'
        definition = next(node for node in ast.parse(scorer.read_text()).body
                          if isinstance(node, ast.FunctionDef) and node.name == 'score_folder')
        metadata = {'label': 'capspeech/age', 'shard': '0of1', 'n_rows': 1, 'missing_wav': 0,
                    'accent_mode': 'ensemble', 'age_mode': 'voxprofile',
                    'input_provenance': {'csv_sha256': sha256(self.source), 'batch': 8, 'max_s': 20,
                                         'scorer_sha256': sha256(scorer)}}
        stem = output_stem('capspeech/age', 0, 1)
        (self.run / (stem + '.json')).write_text(json.dumps(metadata))
        csv_path = self.run / (stem + '_per_utt.csv')
        csv_path.write_text('file,target_age,pred_age\na.wav,older,older\n')
        namespace = {'__package__': 'controllability', '__file__': str(scorer), 'Path': Path, 'json': json,
                     'NORMALISERS': {'age': lambda value: 'older' if value == 'elderly' else value},
                     'read_manifest': lambda *args: (self.source, 'filename', {'age': 'age'}, [{'filename': 'a.wav', 'age': 'elderly'}])}
        exec(compile(ast.Module(body=[definition], type_ignores=[]), 'resume branch fixture', 'exec'), namespace)
        args = (self.root / 'capspeech/age', {}, 'ensemble', 'manifest.csv', 20, 8, 'cpu')
        self.assertEqual(namespace['score_folder'](*args, only_attrs=['age'], out_dir=self.run, resume=True), metadata)
        csv_path.write_text('file,target_age,pred_age\nb.wav,older,older\n')
        with self.assertRaisesRegex(ValueError, 'file IDs/order'):
            namespace['score_folder'](*args, only_attrs=['age'], out_dir=self.run, resume=True)
