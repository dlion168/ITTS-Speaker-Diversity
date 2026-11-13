"""Pinned model/path contracts with a mocked hub; never download weights."""
import json
import hashlib
import ast
import os
import io
from contextlib import redirect_stdout
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from generation import download_weights
from generation.model_pins import load_pins, checkpoint_path, verify_voxinstruct_files


class WeightDownloadContractTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_default_root_matches_generation_driver(self):
        self.assertEqual(download_weights.ROOT, ROOT.resolve())
        self.assertEqual(download_weights.build_parser().parse_args(['--weights-dir', str(self.root)]).weights_dir, self.root)

    def test_direct_infer_root_is_independent_of_working_directory(self):
        source = ROOT / 'generation/infer.py'
        tree = ast.parse(source.read_text())
        argument = next(node for node in ast.walk(tree) if isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute) and node.func.attr == 'add_argument'
                        and node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == '--weights_dir')
        default = next(keyword.value for keyword in argument.keywords if keyword.arg == 'default')
        for environment, expected in [({}, ROOT / 'weights'), ({'WEIGHTS_DIR': '/fixture/models'}, Path('/fixture/models'))]:
            with patch.dict(os.environ, environment, clear=True):
                result = eval(compile(ast.Expression(default), 'inference default expression', 'eval'),
                              {'os': os, '_HERE': source.parent})
            self.assertEqual(Path(result), expected)

    def test_inspection_does_not_create_weight_directory(self):
        destination = self.root / 'not-created'
        for flag in ['--list', '--check', '--manual', '--help']:
            result = subprocess.run([sys.executable, str(ROOT / 'generation/download_weights.py'), flag,
                                     '--weights-dir', str(destination)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(destination.exists())

    def test_all_hf_assets_have_immutable_recorded_pins(self):
        pins, _ = load_pins()
        for repo, _, _ in download_weights.HF_MODELS.values():
            self.assertRegex(pins[repo], r'^[0-9a-f]{40}$')
        self.assertNotIn('refs/pr/9', pins.values())

    def test_download_uses_config_pin_and_success_receipt(self):
        repo, local, _ = download_weights.HF_MODELS['parler-large']
        with patch.object(download_weights, 'WEIGHTS_DIR', self.root), patch('huggingface_hub.snapshot_download') as hub:
            hub.side_effect = lambda **kwargs: (Path(kwargs['local_dir']) / 'model.bin').write_bytes(b'synthetic checkpoint fixture')
            self.assertTrue(download_weights.download_hf('parler-large', repo, local, None))
            self.assertEqual(hub.call_args.kwargs['revision'], load_pins()[0][repo])
            self.assertTrue(json.loads((self.root / local / '.itts_download.json').read_text())['complete'])
            self.assertTrue(download_weights.download_hf('parler-large', repo, local, None))
            self.assertEqual(hub.call_count, 1)

    def test_unknown_existing_weights_are_not_overwritten(self):
        repo, local, _ = download_weights.HF_MODELS['parler-mini']
        folder = self.root / local;folder.mkdir()
        original = folder / 'model.bin';original.write_bytes(b'preserve local weights')
        with patch.object(download_weights, 'WEIGHTS_DIR', self.root), patch('huggingface_hub.snapshot_download') as hub:
            with self.assertRaisesRegex(ValueError, 'no matching download receipt'):
                download_weights.download_hf('parler-mini', repo, local, None)
            hub.assert_not_called()
        self.assertEqual(original.read_bytes(), b'preserve local weights')

    def test_model_and_auxiliary_fallbacks_use_pins_without_network(self):
        pins, _ = load_pins()
        for repo in ['OpenMOSS-Team/MOSS-VoiceGenerator', 'OpenMOSS-Team/MOSS-Audio-Tokenizer',
                     'nvidia/bigvgan_v2_24khz_100band_256x', 'google/flan-t5-large']:
            with patch('huggingface_hub.snapshot_download', return_value='/fixture/pinned-snapshot') as hub:
                self.assertEqual(checkpoint_path(repo, self.root / 'missing'), '/fixture/pinned-snapshot')
                self.assertEqual(hub.call_args.kwargs['revision'], pins[repo])

    def test_existing_local_checkpoint_is_not_replaced(self):
        self.root.mkdir(exist_ok=True)
        with patch('huggingface_hub.snapshot_download') as hub:
            self.assertEqual(checkpoint_path('OpenMOSS-Team/MOSS-VoiceGenerator', self.root), str(self.root))
            hub.assert_not_called()

    def test_manual_checkpoint_hashes_are_checked_without_deserialization(self):
        file = self.root / 'ar_1800k.pyt';file.write_bytes(b'synthetic bytes')
        pins = {'ar_1800k.pyt': {'bytes': len(file.read_bytes()), 'sha256': hashlib.sha256(file.read_bytes()).hexdigest()}}
        self.assertTrue(verify_voxinstruct_files(self.root, pins))
        file.write_bytes(b'different bytes')
        with self.assertRaises(ValueError):verify_voxinstruct_files(self.root, pins)
        self.assertEqual(file.read_bytes(), b'different bytes')

    def test_recorded_download_with_missing_weights_is_not_skipped(self):
        repo, local, _ = download_weights.HF_MODELS['parler-large']
        folder = self.root / local;folder.mkdir()
        (folder / '.itts_download.json').write_text(json.dumps({'repo_id': repo, 'revision': load_pins()[0][repo], 'complete': True, 'files': {'model.bin': 5}}))
        with patch.object(download_weights, 'WEIGHTS_DIR', self.root), patch('huggingface_hub.snapshot_download') as hub:
            with self.assertRaisesRegex(ValueError, 'missing/changed'):
                download_weights.download_hf('parler-large', repo, local, None)
            hub.assert_not_called()

    def test_single_manual_tensor_does_not_look_complete_in_status(self):
        folder = self.root / 'voxinstruct/pretrained/voxinstruct-sft-checkpoint';folder.mkdir(parents=True)
        (folder / 'ar_1800k.pyt').write_bytes(b'partial toy file')
        (self.root / 'voxinstruct/repo').mkdir()
        output = io.StringIO()
        with patch.object(download_weights, 'WEIGHTS_DIR', self.root), redirect_stdout(output):
            download_weights.check_status()
        self.assertIn('AR/NAR sizes=missing/different', output.getvalue())
