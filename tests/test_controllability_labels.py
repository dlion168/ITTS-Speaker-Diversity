"""merge_shards.py writes the prompt names, not the classifier's label names."""
from pathlib import Path
import json
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ControllabilityLabelTests(unittest.TestCase):
    def test_summary_uses_prompt_names(self):
        rows = ['file,target_accent,pred_accent,target_age,pred_age,target_gender,pred_gender',
                'a.wav,North America,English,,,,',
                'b.wav,South Asia,South Asia,,,,',
                'c.wav,,,older,older,female,female']
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            (path / 'controllability.sys__mixed_per_utt.csv').write_text('\n'.join(rows) + '\n')
            subprocess.run([sys.executable, str(ROOT / 'controllability/merge_shards.py'),
                            '--results_dir', str(path)], check=True, capture_output=True)
            entry = json.loads((path / 'summary.json').read_text())['sys/mixed']
        accent = entry['attributes']['accent']
        self.assertEqual(accent['confusion'], {'north_american': {'british_isles': 1},
                                               'south_asian': {'south_asian': 1}})
        self.assertEqual(entry['attributes']['age']['per_class_recall'], {'elderly': 1.0})
        self.assertEqual(list(entry['per_cell']), ['female_elderly'])

    def test_committed_summary_has_no_classifier_names(self):
        text = (ROOT / 'results/controllability/summary.json').read_text()
        for name in ('North America', 'English', 'South Asia', 'Oceania', 'South African', 'Germanic', '"older"', '_older'):
            self.assertNotIn(name, text)

    def test_all_requested_aliases_preserve_perfect_classification_counts(self):
        aliases = {'English': 'british_isles', 'North America': 'north_american',
                   'South Asia': 'south_asian', 'Oceania': 'oceanian',
                   'South African': 'south_african', 'Germanic': 'germanic'}
        rows = ['file,target_accent,pred_accent,target_age,pred_age']
        rows += [f'{i}.wav,{old},{old},,' for i, old in enumerate(aliases)]
        rows.append('age.wav,,,older,older')
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            (path / 'controllability.sys__mixed_per_utt.csv').write_text('\n'.join(rows) + '\n')
            subprocess.run([sys.executable, str(ROOT / 'controllability/merge_shards.py'),
                            '--results_dir', str(path)], check=True, capture_output=True)
            entry = json.loads((path / 'summary.json').read_text())['sys/mixed']
        self.assertEqual(entry['n_rows'], 7)
        accent = entry['attributes']['accent']
        self.assertEqual(accent['confusion'], {name: {name: 1} for name in aliases.values()})
        self.assertEqual(accent['n'], 6)
        self.assertEqual(accent['controllability_acc'], 1.0)
        self.assertEqual(accent['uar'], 1.0)
        self.assertEqual(entry['attributes']['age']['confusion'], {'elderly': {'elderly': 1}})


if __name__ == '__main__':
    unittest.main()
