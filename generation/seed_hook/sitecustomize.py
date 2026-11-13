"""Opt-in seed propagation to generation subprocesses; no credentials are touched."""
import os
if os.environ.get('ITTS_EXPERIMENT_SEED'):
    import random
    import numpy as np
    import torch
    seed=int(os.environ['ITTS_EXPERIMENT_SEED'])
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
