"""Immutable release defaults and preservation of existing outputs, using toy arrays/audio."""
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from embeddings.download_released import write_set
from embeddings.download_released_audio import write_wav, target_dir
from embeddings.release_assets import revision_for


class ReleaseAssetTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.pins = json.loads((ROOT / 'configs/release_assets.json').read_text())

    def test_default_release_and_classifier_revisions_are_immutable(self):
        for key in ['dataset', 'accent_classifiers']:
            record = self.pins[key]
            self.assertRegex(record['revision'], r'^[0-9a-f]{40}$')
            self.assertEqual(revision_for(record['repo_id']), record['revision'])
        self.assertEqual(revision_for(self.pins['dataset']['repo_id'], 'explicit-ref'), 'explicit-ref')
        self.assertIsNone(revision_for('different/repository'))

    def test_common_shell_uses_same_release_pin(self):
        result = subprocess.run(['bash', '-c', 'source "$1"; printf "%s" "$HF_REVISION"',
                                 'fixture', str(ROOT / 'experiments/common.sh')],
                                env={key: value for key, value in os.environ.items() if key not in ['HF_REVISION', 'HF_REPO', 'HF_LOCAL_DIR']},
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, self.pins['dataset']['revision'])

    def test_default_results_are_isolated_from_reference_tables(self):
        result = subprocess.run(['bash', '-c', 'source "$1"; printf "%s" "$RESULTS_DIR"',
                                 'fixture', str(ROOT / 'experiments/common.sh')],
                                env={key: value for key, value in os.environ.items() if key != 'RESULTS_DIR'},
                                capture_output=True, text=True)
        self.assertIn('/results/reproductions/run-', result.stdout)
        self.assertNotEqual(Path(result.stdout), ROOT / 'results')

    def test_equal_embeddings_are_preserved_and_different_ones_rejected(self):
        folder = self.root / 'emb'
        write_set(folder, ['b', 'a'], np.array([[2.], [1.]], dtype=np.float32))
        original = (folder / 'embeddings.npy').read_bytes()
        write_set(folder, ['a', 'b'], np.array([[1.], [2.]], dtype=np.float32))
        self.assertEqual((folder / 'embeddings.npy').read_bytes(), original)
        with self.assertRaises(FileExistsError):
            write_set(folder, ['a', 'b'], np.array([[3.], [2.]], dtype=np.float32))
        self.assertEqual((folder / 'embeddings.npy').read_bytes(), original)

    def test_duplicate_or_nonfinite_embeddings_fail_before_write(self):
        for names, X in [(['a', 'a'], np.ones((2, 2))), (['a'], np.array([[np.nan]]))]:
            folder = self.root / 'not-created'
            with self.assertRaises(ValueError):write_set(folder, names, X)
            self.assertFalse(folder.exists())

    def test_existing_audio_is_checked_not_overwritten(self):
        buffer = io.BytesIO()
        sf.write(buffer, np.array([100, -100, 50, -50], dtype=np.int16), 16000, format='FLAC', subtype='PCM_16')
        output = self.root / 'fixture.wav'
        self.assertTrue(write_wav(output, buffer.getvalue()))
        original = output.read_bytes()
        self.assertFalse(write_wav(output, buffer.getvalue()))
        self.assertEqual(output.read_bytes(), original)
        other = io.BytesIO();sf.write(other, np.array([200, -200], dtype=np.int16), 16000, format='FLAC', subtype='PCM_16')
        with self.assertRaises(FileExistsError):write_wav(output, other.getvalue())
        self.assertEqual(output.read_bytes(), original)

    def test_unsafe_release_metadata_cannot_escape_output_layout(self):
        with self.assertRaises(ValueError):target_dir('attribute', 'capspeech', {'attribute': '..'})
        with self.assertRaises(ValueError):target_dir('overall', 'capspeech', {'partition': True})
        with self.assertRaises(ValueError):target_dir('sampling_sweep', 'voxcpm2', {'sweep_param': 'cfg', 'sweep_value': float('nan'), 'seed': 17})

    def test_ci_is_offline_cpu_only_and_read_only(self):
        text = (ROOT / '.github/workflows/lightweight-tests.yml').read_text()
        self.assertIn('contents: read', text)
        self.assertIn('HF_HUB_OFFLINE: "1"', text)
        self.assertIn('CUDA_VISIBLE_DEVICES: ""', text)
        self.assertNotIn('secrets.', text)


if __name__ == '__main__':
    unittest.main()
