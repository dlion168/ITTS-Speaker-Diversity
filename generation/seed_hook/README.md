# Generation subprocess seeding

`sitecustomize.py` seeds Python, NumPy and PyTorch only when ITTS_EXPERIMENT_SEED is explicitly set. `run_seeded_infer.py` adds this directory to PYTHONPATH after entering the selected generation runtime, so outer environment managers are not forced to import that runtime's libraries. VoxInstruct subprocesses inherit the setting through their existing environment propagation.

Input: the explicit seed for a new run. Output: seeded RNG state, with no audio edits. Bitwise equality across different software or hardware is not guaranteed.
