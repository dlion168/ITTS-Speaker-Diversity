"""Sweep selection/layout regressions; never load an encoder or TTS model."""
import ast
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('sweep_summary', ROOT / 'experiments/03_sampling_sweep/summarize.py')
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


class SweepLayoutTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def leaf(self, system='voxcpm2', param='cfg_1.0', seed='seed_17', legacy=False):
        leaf = self.root / system / param / seed
        for encoder in summary.ENCODERS:
            path = (leaf / seed if legacy else leaf) / encoder
            path.mkdir(parents=True)
            # Only presence is tested here; these are not real embeddings.
            (path / 'embeddings.npy').write_bytes(b'synthetic presence-only fixture')
        return leaf

    def test_cli_selection_overrides_environment(self):
        with patch.dict(os.environ, {'SYSTEMS': 'moss-voicegen'}):
            self.assertEqual(summary.selected_systems(['voxcpm2']), ['voxcpm2'])

    def test_environment_selection_ignores_known_non_sweep_systems(self):
        with patch.dict(os.environ, {'SYSTEMS': 'parler-mini voxcpm2 voxcpm2'}):
            self.assertEqual(summary.selected_systems(), ['voxcpm2'])

    def test_no_environment_defaults_to_three_sweep_systems(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(summary.selected_systems(), summary.SYSTEMS)

    def test_invalid_or_empty_selection_is_rejected(self):
        for value in ['', 'parler-mini', 'voxcpm2 typo']:
            with patch.dict(os.environ, {'SYSTEMS': value}), self.assertRaises(ValueError):
                summary.selected_systems()

    def test_unselected_missing_systems_are_not_inspected(self):
        leaf = self.leaf()
        result = summary.discover_leaves(self.root, ['voxcpm2'])
        self.assertEqual(result, [('voxcpm2', 'cfg', 1.0, 17, leaf)])

    def test_selected_missing_system_fails_clearly(self):
        with self.assertRaisesRegex(ValueError, 'selected system voxcpm2'):
            summary.discover_leaves(self.root, ['voxcpm2'])

    def test_empty_system_and_seedless_parameter_are_rejected(self):
        (self.root / 'voxcpm2').mkdir()
        with self.assertRaisesRegex(ValueError, 'no cfg parameter'):
            summary.discover_leaves(self.root, ['voxcpm2'])
        (self.root / 'voxcpm2/cfg_1.0').mkdir()
        with self.assertRaisesRegex(ValueError, 'no seed directories'):
            summary.discover_leaves(self.root, ['voxcpm2'])

    def test_missing_encoder_and_double_seed_layout_are_rejected(self):
        leaf = self.leaf()
        (leaf / summary.ENCODERS[0] / 'embeddings.npy').unlink()
        with self.assertRaisesRegex(ValueError, 'missing selected embedding'):
            summary.discover_leaves(self.root, ['voxcpm2'])
        self.leaf('voxinstruct', legacy=True)
        with self.assertRaisesRegex(ValueError, 'one seed level too deep'):
            summary.discover_leaves(self.root, ['voxinstruct'])

    def test_auxiliary_directories_do_not_become_parameters_or_seeds(self):
        leaf = self.leaf()
        (self.root / 'voxcpm2/logs').mkdir()
        (leaf.parent / 'logs').mkdir()
        self.assertEqual(len(summary.discover_leaves(self.root, ['voxcpm2'])), 1)

    def test_extractor_path_expression_preserves_all_experiment_layouts(self):
        # Evaluate only the writer's path expression, not main() or any model.
        tree = ast.parse((ROOT / 'embeddings/extract_embeddings.py').read_text())
        assignment = next(node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                          and any(isinstance(target, ast.Name) and target.id == 'model_output_dir' for target in node.targets))
        expression = compile(ast.Expression(assignment.value), 'extractor path contract', 'eval')
        for model, folder in [('voxcpm2/cfg_1.0', 'seed_17'), ('capspeech', 'sample1'), ('voxcpm2', 'age')]:
            output = eval(expression, {'__builtins__': {}}, {'output_dir': self.root,
                          'args': SimpleNamespace(tts_model=model), 'dataset_name': folder, 'model_name': 'encoder'})
            self.assertEqual(output, self.root / model / folder / 'encoder')
            self.assertNotIn('seed_17/seed_17', str(output))

    def test_summary_on_toy_arrays_keeps_metric_values_and_columns(self):
        leaf = self.leaf()
        for encoder in summary.ENCODERS:
            summary.np.save(leaf / encoder / 'embeddings.npy', summary.np.eye(4))
            (leaf / encoder / 'filenames.txt').write_text('a.wav\nb.wav\nc.wav\nd.wav\n')
        output = self.root / 'results'
        result = subprocess.run([sys.executable, str(ROOT / 'experiments/03_sampling_sweep/summarize.py'),
                                 '--sweep_root', str(self.root), '--systems', 'voxcpm2', '--out_dir', str(output)],
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        table = summary.pd.read_csv(output / 'sampling_sweep/param_sweep_seeded_diversity.csv')
        self.assertEqual(list(table.columns), ['system', 'param_kind', 'param', 'encoder', 'seed', 'n', 'vendi', 'mpcd'])
        self.assertEqual(len(table), 8)
        self.assertTrue((table['system'] == 'voxcpm2').all())
        self.assertTrue((table['n'] == 4).all())
        self.assertTrue(summary.np.allclose(table['vendi'], 4))
        self.assertTrue(summary.np.allclose(table['mpcd'], 1))


class SweepShellTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.data = self.root / 'data'
        self.log = self.root / 'calls.jsonl'
        prefix = self.root / 'runtime'
        (prefix / 'bin').mkdir(parents=True)
        runtime = prefix / 'bin/python'
        # A stand-in records argv and materializes presence-only output files.
        # It never executes the script passed to it or imports model code.
        runtime.write_text('#!' + sys.executable + '\n' + '''import json,os,pathlib,sys
args=sys.argv[1:]
if args[:1]==['-u']:args=args[1:]
with open(os.environ['FIXTURE_CALLS'],'a') as handle:handle.write(json.dumps(args)+'\\n')
if args and args[0].endswith('extract_embeddings.py'):
 if os.environ.get('FIXTURE_EXTRACT_FAILURE')=='1':sys.exit(7)
 if os.environ.get('FIXTURE_NO_OUTPUT')=='1':sys.exit(0)
 options=dict(zip(args[1::2],args[2::2]))
 out=pathlib.Path(options['--output_dir'])/options['--tts_model']/pathlib.Path(options['--wav_dir']).name/options['--models']
 out.mkdir(parents=True,exist_ok=True)
 (out/'embeddings.npy').write_bytes(b'presence-only fixture, not actual embeddings')
''')
        runtime.chmod(0o755)
        self.env = {**os.environ, 'ITTS_DATA_ROOT': str(self.data), 'RESULTS_DIR': str(self.root / 'results'),
                    'EMBED_ENV': str(prefix), 'NEMO_ENV': str(prefix), 'SYSTEMS': 'voxcpm2',
                    'FIXTURE_CALLS': str(self.log)}
        (self.data / 'embeddings_overall/real/set_1').mkdir(parents=True)

    def wav_leaf(self, system='voxcpm2', nonempty=True):
        param = 'temp_1.0' if system == 'moss-voicegen' else 'cfg_1.0'
        path = self.data / 'outputs/param_sweep' / system / param / 'seed_17'
        path.mkdir(parents=True)
        if nonempty:
            (path / 'fixture.wav').write_bytes(b'presence-only WAV fixture, not real speech')
        return path

    def run_driver(self, stage='generated'):
        return subprocess.run(['bash', str(ROOT / 'experiments/03_sampling_sweep/run.sh'), '--from', stage],
                              capture_output=True, text=True, env=self.env, cwd='/', timeout=30)

    def calls(self):
        return [json.loads(line) for line in self.log.read_text().splitlines()] if self.log.exists() else []

    def test_generated_checks_seed_leaf_and_extracts_selected_system_only(self):
        self.wav_leaf()
        self.wav_leaf('moss-voicegen')
        result = self.run_driver()
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.calls()
        extracts = [call for call in calls if call[0].endswith('extract_embeddings.py')]
        self.assertEqual(len(extracts), 8)
        for call in extracts:
            self.assertEqual(call[call.index('--tts_model') + 1], 'voxcpm2/cfg_1.0')
            self.assertIn('/voxcpm2/cfg_1.0/seed_17', call[call.index('--wav_dir') + 1])
        self.assertEqual(calls[-1][-2:], ['--systems', 'voxcpm2'])
        self.assertTrue((self.data / 'embeddings_param_sweep/voxcpm2/cfg_1.0/seed_17/speechbrain_ecapa/embeddings.npy').is_file())
        self.assertFalse((self.data / 'embeddings_param_sweep/moss-voicegen').exists())
        self.assertFalse(list(self.data.rglob('seed_17/seed_17')))

    def test_missing_selected_system_stops_before_extraction(self):
        self.wav_leaf('moss-voicegen')
        result = self.run_driver()
        self.assertEqual(result.returncode, 2)
        self.assertIn('no sweep seed directories', result.stderr)
        self.assertEqual(self.calls(), [])
        self.assertFalse((self.data / 'embeddings_param_sweep').exists())

    def test_empty_seed_leaf_stops_before_extraction(self):
        self.wav_leaf(nonempty=False)
        result = self.run_driver()
        self.assertEqual(result.returncode, 2)
        self.assertIn('no generated audio', result.stderr)
        self.assertEqual(self.calls(), [])

    def test_unknown_system_is_not_silently_ignored(self):
        self.env['SYSTEMS'] = 'voxcpm2 typo'
        result = self.run_driver()
        self.assertEqual(result.returncode, 2)
        self.assertIn('unknown paper system', result.stderr)
        self.assertEqual(self.calls(), [])

    def test_audio_stage_passes_system_filter_to_download(self):
        self.wav_leaf()
        result = self.run_driver('audio')
        self.assertEqual(result.returncode, 0, result.stderr)
        download = next(call for call in self.calls() if call[0].endswith('download_released_audio.py'))
        self.assertEqual(download[download.index('--systems') + 1], 'voxcpm2')

    def test_embeddings_stage_passes_system_filter_to_summary(self):
        result = self.run_driver('embeddings')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(any(call[0].endswith('extract_embeddings.py') for call in self.calls()))
        self.assertEqual(self.calls()[-1][-2:], ['--systems', 'voxcpm2'])

    def test_worker_failure_status_is_preserved_and_owned_lock_is_cleaned(self):
        self.wav_leaf()
        self.env['FIXTURE_EXTRACT_FAILURE'] = '1'
        result = self.run_driver()
        self.assertEqual(result.returncode, 7)
        self.assertFalse(any(call[0].endswith('summarize.py') for call in self.calls()))
        self.assertEqual(list((self.data / 'embeddings_param_sweep/.locks').iterdir()), [])

    def test_worker_help_creates_no_embedding_output(self):
        output = self.data / 'embeddings_param_sweep'
        self.env['PARAM_SWEEP_EMBEDDINGS_ROOT'] = str(output)
        result = subprocess.run(['bash', str(ROOT / 'embeddings/run_embedding_param_sweep.sh'), '--help'],
                                capture_output=True, text=True, env=self.env, cwd='/', timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(output.exists())

    def test_successful_child_without_array_is_not_a_complete_worker(self):
        self.wav_leaf()
        self.env['FIXTURE_NO_OUTPUT'] = '1'
        result = self.run_driver()
        self.assertEqual(result.returncode, 1)
        self.assertIn('output is incomplete', result.stderr)
        self.assertFalse(any(call[0].endswith('summarize.py') for call in self.calls()))
        self.assertEqual(list((self.data / 'embeddings_param_sweep/.locks').iterdir()), [])


if __name__ == '__main__':
    unittest.main()
