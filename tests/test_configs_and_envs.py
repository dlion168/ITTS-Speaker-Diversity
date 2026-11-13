"""Checks of configs/systems.yaml, the reference-audio default path and envs/setup_generation_envs.sh (installs nothing)."""
import ast
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import yaml

ROOT = Path(__file__).resolve().parents[1]


class SystemsConfigTests(unittest.TestCase):
    def setUp(self):
        self.systems = yaml.safe_load((ROOT / 'configs/systems.yaml').read_text())['systems']

    def test_seven_systems_and_capspeech_checkpoint_per_category(self):
        self.assertEqual(len(self.systems), 7)
        cap = next(row for row in self.systems if row['code_name'] == 'capspeech')
        self.assertEqual(cap['checkpoint_per_category'], {
            'accent': 'AccCapTTS', 'emotion': 'EmoCapTTS', 'age': 'CapTTS',
            'gender': 'CapTTS', 'gender_x_age': 'CapTTS', 'overall': 'CapTTS'})

    def test_revisions_and_upstream_commits_are_full_hashes(self):
        for system in self.systems:
            for key in ['revision', 'upstream_commit']:
                if key in system:
                    self.assertRegex(system[key], r'^[0-9a-f]{40}$')

    def test_reference_audio_default_aligns_with_data_root_without_download(self):
        # Evaluate only the default path expressions, never the downloader.
        source = ROOT / 'data/emilia/download_reference_audio.py'
        tree = ast.parse(source.read_text())
        data_root = next(node.value for node in ast.walk(tree) if isinstance(node, ast.Assign)
                         and any(isinstance(target, ast.Name) and target.id == 'data_root' for target in node.targets))
        out_argument = next(node for node in ast.walk(tree) if isinstance(node, ast.Call)
                            and isinstance(node.func, ast.Attribute) and node.func.attr == 'add_argument'
                            and node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == '--out_dir')
        default = next(keyword.value for keyword in out_argument.keywords if keyword.arg == 'default')
        for environment, expected in [({}, ROOT / 'data'), ({'ITTS_DATA_ROOT': ''}, ROOT / 'data'),
                                      ({'ITTS_DATA_ROOT': '/fixture/external-data'}, Path('/fixture/external-data'))]:
            with patch.dict(os.environ, environment, clear=True):
                root = eval(compile(ast.Expression(data_root), 'data root expression', 'eval'),
                            {'Path': Path, 'os': os, '__file__': str(source)})
                output = eval(compile(ast.Expression(default), 'output default expression', 'eval'), {'data_root': root})
                self.assertEqual(output, expected / 'real_world_data')


class GenerationEnvironmentHelperTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.log = self.root / 'conda_calls'
        (self.root / 'bin').mkdir()
        conda = self.root / 'bin/conda'
        conda.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$FIXTURE_CONDA_CALLS"\nexit 0\n')
        conda.chmod(0o755)
        self.env = {**os.environ, 'PATH': str(self.root / 'bin') + ':' + os.environ['PATH'],
                    'FIXTURE_CONDA_CALLS': str(self.log)}

    def run_helper(self, *targets):
        return subprocess.run(['bash', str(ROOT / 'envs/setup_generation_envs.sh'), *targets],
                              env=self.env, capture_output=True, text=True, timeout=10)

    def test_unknown_environment_is_rejected_before_conda(self):
        for target in ['typo', 'itts_moss']:
            result = self.run_helper(target)
            self.assertEqual(result.returncode, 2)
            self.assertIn('Unknown generation environment', result.stderr)
        self.assertFalse(self.log.exists())

    def test_all_creates_the_six_generation_environments(self):
        names = ['moss', 'parler', 'qwen', 'vox', 'voxcpm2', 'capspeech']
        result = self.run_helper('all')
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.log.read_text().splitlines()
        self.assertEqual([call.split()[-1].split('/')[-1] for call in calls], [f'itts_{n}.yml' for n in names])
        for name in names:
            env = yaml.safe_load((ROOT / f'envs/itts_{name}.yml').read_text())
            self.assertEqual(env['name'], f'itts_{name}')


if __name__ == '__main__':
    unittest.main()
