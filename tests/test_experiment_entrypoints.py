"""Shell entrypoint errors checked without models, downloads or generation."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ExperimentEntrypointTests(unittest.TestCase):
    def test_scratch_explains_separate_generation(self):
        result = subprocess.run(['bash', str(ROOT / 'experiments/02_overall_diversity/run.sh'), '--from', 'scratch'],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('generate.sh', result.stderr)
        self.assertIn('--from generated', result.stderr)

    def test_metadata_only_directory_is_not_generated_audio(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            (path / 'generation_run.json').write_text('{}')
            result = subprocess.run(['bash', '-c', 'source "$1"; need_generated fixture-generate.sh "$2"',
                                     'fixture', str(ROOT / 'experiments/common.sh'), str(path)],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn('no generated audio', result.stderr)
            # The helper only checks for a nonempty WAV; full payload integrity
            # is checked by the generation driver, not certified by this helper.
            (path / 'fixture.wav').write_bytes(b'not a real WAV; presence-only fixture')
            result = subprocess.run(['bash', '-c', 'source "$1"; need_generated fixture-generate.sh "$2"',
                                     'fixture', str(ROOT / 'experiments/common.sh'), str(path)],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
