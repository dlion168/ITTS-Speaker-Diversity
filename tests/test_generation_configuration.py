"""Configuration/metadata contracts tested without loading any TTS model."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from generation.settings import load_settings
from generation.preflight import attribute_payloads


class GenerationConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.input = self.root / 'partition_1.jsonl'
        self.input.write_text(''.join(json.dumps({'id': f'fixture{i}', 'text': 'Synthetic test only.'}) + '\n' for i in range(2)))
        self.config = {'instruction': 'A test neutral instruction.', 'generation_seed': 11,
                       'partitions': [1], 'texts_per_partition': 2,
                       'text_source': str(self.root / 'partition_{partition}.jsonl')}
        self.path = self.root / 'overall.yaml'
        self.write_config()

    def write_config(self):
        self.path.write_text(json.dumps(self.config))

    def command(self, *extra, kind='overall'):
        return [sys.executable, str(ROOT / 'generation/run_generation.py'), kind,
                '--models', 'capspeech' if kind == 'overall' else 'voxinstruct',
                '--config', str(self.path), '--data-root', str(self.root / 'output'), *extra]

    def run_plan(self, *extra, kind='overall'):
        result = subprocess.run(self.command('--dry-run', *extra, kind=kind), capture_output=True, text=True, cwd='/')
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_yaml_instruction_reaches_infer_command(self):
        result = self.run_plan()
        argv = result['tasks'][0]['command']
        self.assertEqual(argv[argv.index('--default_style_prompt') + 1], self.config['instruction'])
        self.assertEqual(result['tasks'][0]['seed'], 11)
        self.assertFalse((self.root / 'output').exists())

    def test_cli_instruction_and_seed_override(self):
        result = self.run_plan('--instruction', 'Explicit CLI instruction.', '--seed', '42')
        argv = result['tasks'][0]['command']
        self.assertEqual(argv[argv.index('--default_style_prompt') + 1], 'Explicit CLI instruction.')
        self.assertEqual(result['tasks'][0]['seed'], 42)

    def test_partition_changes_affect_tasks(self):
        self.config['partitions'] = [2, 3]
        self.write_config()
        result = self.run_plan()
        self.assertEqual(len(result['tasks']), 2)
        self.assertTrue(result['tasks'][0]['output'].endswith('sample2'))

    def test_custom_sweep_uses_config_flag_values_and_seeds(self):
        self.config = {'instruction': 'Sweep fixture.', 'values': [.51, .54], 'seeds': [7],
                       'utterances_per_config': 2, 'text_source': str(self.input),
                       'systems': {'voxinstruct': {'param': 'cfg', 'flag': '--vox_cfg_at_on_text'}}}
        self.write_config()
        result = self.run_plan(kind='sampling_sweep')
        self.assertEqual(len(result['tasks']), 2)
        self.assertEqual({row['seed'] for row in result['tasks']}, {7})
        self.assertEqual(len({row['output'] for row in result['tasks']}), 2)
        self.assertTrue(all('--vox_cfg_at_on_text' in row['command'] for row in result['tasks']))

    def test_unsafe_or_unresolved_sweep_flag_is_rejected(self):
        for flag in ['TODO', '--output_dir']:
            self.config = {'instruction': 'Fixture.', 'values': [.5], 'seeds': [17],
                           'utterances_per_config': 2, 'systems': {'voxinstruct': {'param': 'cfg', 'flag': flag}}}
            self.write_config()
            result = subprocess.run(self.command('--dry-run', kind='sampling_sweep'), capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.root / 'output').exists())

    def test_invalid_configuration_is_rejected(self):
        for key, value in [('instruction', ''), ('partitions', [True]), ('partitions', [1.0]), ('partitions', [1, 1]), ('generation_seed', True)]:
            changed = dict(self.config);changed[key] = value
            self.path.write_text(json.dumps(changed))
            with self.assertRaises(ValueError):
                load_settings('overall', self.path)

    def runtime_map(self, emit=True):
        # A stand-in emitter tests orchestration only, not a generation model.
        script = """import json,pathlib,sys,wave
options=dict(zip(sys.argv[2::2],sys.argv[3::2]))
out=pathlib.Path(options['--output_dir'])
rows=[json.loads(line) for line in pathlib.Path(options['--input']).read_text().splitlines() if line.strip()]
for row in rows:
 with wave.open(str(out/(row['id']+'.wav')),'wb') as audio:
  audio.setnchannels(1);audio.setsampwidth(2);audio.setframerate(16000);audio.writeframes(b'\\x01\\x00'*1600)
""" if emit else 'pass'
        path = self.root / 'runtimes.json'
        path.write_text(json.dumps({'capspeech': [sys.executable, '-c', script]}))
        return path

    def test_complete_run_records_effective_instruction_and_config(self):
        result = subprocess.run(self.command('--python-map', str(self.runtime_map())), capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        out = self.root / 'output/outputs/overall/capspeech/sample1'
        metadata = json.loads((out / 'generation_run.json').read_text())
        self.assertEqual(metadata['instruction'], self.config['instruction'])
        self.assertEqual(metadata['seed'], 11)
        self.assertEqual(len(metadata['experiment_config']['sha256']), 64)
        self.assertEqual(json.loads((out / 'generation_complete.json').read_text())['validated_pcm_wavs'], 2)

    def test_successful_child_without_wavs_is_not_success(self):
        result = subprocess.run(self.command('--python-map', str(self.runtime_map(emit=False))), capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        out = self.root / 'output/outputs/overall/capspeech/sample1'
        self.assertFalse((out / 'generation_complete.json').exists())

    def test_failed_repeat_archives_previous_completion_receipt(self):
        runtime = self.runtime_map(emit=False)
        result = subprocess.run(self.command('--python-map', str(runtime)), capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        out = self.root / 'output/outputs/overall/capspeech/sample1'
        receipt = out / 'generation_complete.json'
        receipt.write_text('{"previous": "synthetic test receipt"}\n')
        result = subprocess.run(self.command('--python-map', str(runtime)), capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(receipt.exists())
        previous = list(out.glob('generation_complete.previous-*.json'))
        self.assertEqual(len(previous), 1)
        self.assertEqual(json.loads(previous[0].read_text())['previous'], 'synthetic test receipt')

    def test_unknown_existing_audio_is_never_certified_or_overwritten(self):
        out = self.root / 'output/outputs/overall/capspeech/sample1'
        out.mkdir(parents=True)
        original = out / 'fixture0.wav';original.write_bytes(b'preserve this original fixture')
        result = subprocess.run(self.command('--python-map', str(self.runtime_map())), capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(original.read_bytes(), b'preserve this original fixture')
        self.assertFalse((out / 'generation_run.json').exists())

    def test_input_count_mismatch_fails_before_output_writes(self):
        self.config['texts_per_partition'] = 3;self.write_config()
        result = subprocess.run(self.command('--python-map', str(self.runtime_map())), capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.root / 'output').exists())

    def test_conflicting_input_instruction_fails_before_output_writes(self):
        rows = [json.loads(line) for line in self.input.read_text().splitlines()]
        rows[0]['instruction'] = 'A different voice description.'
        self.input.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        result = subprocess.run(self.command('--python-map', str(self.runtime_map())), capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Per-row instruction conflicts', result.stderr)
        self.assertFalse((self.root / 'output').exists())

    def test_attribute_values_and_counts_are_selected_from_yaml(self):
        prompts = self.root / 'prompts.json'
        prompts.write_text(json.dumps([{'id': f'gender-{gender}-s1-0001-T1', 'text': 'Fixture.', 'style_prompt': 'Fixture.'} for gender in ['female', 'male']]))
        self.config = {'single_attribute': {'gender': ['female'], 'utterances_per_value': 1},
                       'prompt_sources': {'single_attribute': str(prompts)}}
        self.write_config()
        settings = load_settings('attributes', self.path)
        payloads = attribute_payloads(settings, self.root / 'new-inputs')
        rows = json.loads(next(iter(payloads.values())))
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]['id'].startswith('gender-female-'))
        self.assertFalse((self.root / 'new-inputs').exists())


if __name__ == '__main__':
    unittest.main()
