"""Non-generating checks for the newly added current-root generation stages."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('generation_plan',ROOT/'generation/run_generation.py')
plan=importlib.util.module_from_spec(spec);spec.loader.exec_module(plan)


class GenerationPlanTests(unittest.TestCase):
    def test_full_paper_plan_sizes(self):
        for kind,count in [('overall',21),('sampling_sweep',45),('attributes',35)]:
            self.assertEqual(len(plan.tasks(kind,plan.SYSTEMS,Path('fixture'))),count)

    def test_seeded_sweep_parameters(self):
        rows=plan.tasks('sampling_sweep',plan.SYSTEMS,Path('fixture'))
        self.assertEqual({row['seed'] for row in rows},{17,42,99})
        self.assertEqual({row['model'] for row in rows},{'moss-voicegen','voxinstruct','voxcpm2'})
        self.assertTrue(all('seed_' in str(row['output']) for row in rows))

    def test_dry_run_never_creates_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            output=Path(temporary)/'not-created'
            run=subprocess.run([sys.executable,str(ROOT/'generation/run_generation.py'),'overall','--models','capspeech','--data-root',str(output),'--dry-run'],capture_output=True,text=True,cwd='/')
            self.assertEqual(run.returncode,0,run.stderr)
            self.assertEqual(len(json.loads(run.stdout)['tasks']),3)
            self.assertFalse(output.exists())

    def test_subprocess_seed_hook_is_deterministic(self):
        environment={**os.environ,'ITTS_EXPERIMENT_SEED':'17',
                     'PYTHONPATH':str(ROOT/'generation/seed_hook')+os.pathsep+os.environ.get('PYTHONPATH','')}
        script='import random,numpy as np,torch,json; print(json.dumps([random.random(),float(np.random.rand()),float(torch.rand(1))]))'
        results=[subprocess.run([sys.executable,'-c',script],env=environment,capture_output=True,text=True,timeout=60) for _ in range(2)]
        self.assertTrue(all(result.returncode==0 for result in results))
        self.assertEqual(json.loads(results[0].stdout),json.loads(results[1].stdout))


if __name__=='__main__':unittest.main()
