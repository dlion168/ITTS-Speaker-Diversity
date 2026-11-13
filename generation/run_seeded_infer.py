#!/usr/bin/env python3
"""Seed an explicitly selected runtime, then execute the unchanged generation entry."""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import random
import runpy
import sys

if __name__=='__main__':
    import numpy as np
    import torch
    seed=int(os.environ['ITTS_EXPERIMENT_SEED'])
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    here=Path(__file__).resolve().parent
    os.environ['PYTHONPATH']=str(here/'seed_hook')+os.pathsep+os.environ.get('PYTHONPATH','')
    metadata=os.environ.get('ITTS_INVOCATION_METADATA')
    if metadata:
        # Runtime of this invocation (WAVs skipped on resume are not covered).
        with Path(metadata).open('a') as handle:
            handle.write(json.dumps({'time':datetime.now(timezone.utc).isoformat(),'seed':seed,
                                     'python':sys.version.split()[0],'numpy':np.__version__,'torch':torch.__version__,
                                     'record_type':'invocation'})+'\n')
    sys.path.insert(0,str(here))
    sys.argv=[str(here/'infer.py'),*sys.argv[1:]]
    runpy.run_path(str(here/'infer.py'),run_name='__main__')
