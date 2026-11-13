"""Regression coverage for 04/05 seed/category leaves and fail-closed scoring."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from scoring_fixture import runtime

ROOT = Path(__file__).resolve().parents[1]


class EntrypointCheckTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.calls = self.root / 'calls.jsonl'
        prefix = self.root / 'runtime'
        runtime(prefix / 'bin/python')
        # Stub only the downstream 04 extraction worker. The entrypoint's real
        # category preflight still executes, without creating 96,000 fake WAVs.
        bin_dir = self.root / 'bin';bin_dir.mkdir()
        bash = bin_dir / 'bash'
        bash.write_text('#!/bin/bash\nif [ "${1##*/}" = run_embedding_subset.sh ]; then exit 0; fi\nexec /bin/bash "$@"\n')
        bash.chmod(0o755)
        self.env = {**os.environ, 'ITTS_DATA_ROOT': str(self.root / 'data'), 'RESULTS_DIR': str(self.root / 'results'),
                    'EMBED_ENV': str(prefix), 'NEMO_ENV': str(prefix), 'VOX_ENV': str(prefix),
                    'VOX_PROFILE_DIR': str(self.root / 'profile'), 'SYSTEMS': 'capspeech', 'NPROC': '1',
                    'SCORING_ALLOW_PARTIAL_INPUTS': '1', 'FIXTURE_CALLS': str(self.calls),
                    'SCORING_RUN_DIR': str(self.root / 'run'), 'SHARDS': '', 'RESUME': '0'}
        self.env['PATH'] = str(bin_dir) + ':' + os.environ['PATH']
        self.py = prefix / 'bin/python'

    def audio(self):
        for category, value in [('gender', 'female'), ('age', 'child'), ('accent', 'north_american'),
                                ('emotion', 'neutral'), ('gender_x_age', 'female_child')]:
            path = self.root / f'data/outputs/attributes/capspeech/{category}'
            path.mkdir(parents=True)
            prefix = 'genderage' if category == 'gender_x_age' else category
            (path / f'{prefix}-{value}-s1-0001-T1.wav').write_bytes(b'presence-only fixture')

    def execute(self, experiment):
        return subprocess.run(['/bin/bash', str(ROOT / 'experiments' / experiment / 'run.sh'), '--from', 'generated'],
                              env=self.env, cwd='/', capture_output=True, text=True, timeout=30)

    def test_04_accepts_real_category_layout(self):
        self.audio()
        result = self.execute('04_attribute_diversity')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        calls = [json.loads(line) for line in self.calls.read_text().splitlines()]
        self.assertTrue(any(call[0].endswith('half_sampling_ci.py') for call in calls))

    def test_05_accepts_layout_and_merges_only_this_run(self):
        self.audio()
        result = self.execute('05_controllability')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        summary = json.loads((self.root / 'results/controllability/summary.json').read_text())
        self.assertEqual(len(summary), 5)
        self.assertTrue(all(entry['n_rows'] == 1 for entry in summary.values()))
        self.assertTrue((self.root / 'run/scoring_run.json').is_file())

    def test_missing_category_stops_both_entrypoints_before_work(self):
        self.audio()
        (self.root / 'data/outputs/attributes/capspeech/emotion/emotion-neutral-s1-0001-T1.wav').unlink()
        for name in ['04_attribute_diversity', '05_controllability']:
            result = self.execute(name)
            self.assertEqual(result.returncode, 2)
            self.assertIn('no generated audio', result.stderr)
        self.assertFalse(self.calls.exists())

    def test_worker_failure_does_not_replace_summary_or_merge_stale_csv(self):
        self.audio()
        self.env['FIXTURE_WORKER_FAIL'] = '1'
        output = self.root / 'results/controllability/summary.json'
        output.parent.mkdir(parents=True)
        output.write_text('{"previous": "preserve"}\n')
        result = self.execute('05_controllability')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('no summary was replaced', result.stderr)
        self.assertEqual(output.read_text(), '{"previous": "preserve"}\n')
        calls = [json.loads(line) for line in self.calls.read_text().splitlines()]
        self.assertFalse(any(call[0].endswith('merge_shards.py') for call in calls))

    def test_run_directory_is_not_adopted_without_explicit_resume(self):
        self.audio()
        first = self.execute('05_controllability')
        self.assertEqual(first.returncode, 0, first.stderr)
        second = self.execute('05_controllability')
        self.assertNotEqual(second.returncode, 0)
        self.assertIn('existing scoring directory', second.stderr)

    def test_direct_scoring_default_summary_is_not_a_reference_table(self):
        self.audio()
        result = self.execute('05_controllability')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.env['RESUME'] = '1'
        self.env.update({'PY': str(self.py), 'MODELS': 'capspeech'})
        self.env.pop('SUMMARY', None)
        resumed = subprocess.run(['/bin/bash', str(ROOT / 'controllability/run_scoring.sh')], env=self.env,
                                 cwd='/', capture_output=True, text=True, timeout=30)
        self.assertEqual(resumed.returncode, 0, resumed.stdout + resumed.stderr)
        self.assertTrue((self.root / 'run/summary.json').is_file())
