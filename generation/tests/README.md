# Generation-stage tests

`test_generation_plan.py` checks paper task counts, sweep seeds/parameters, side-effect-free dry runs, and deterministic seed propagation into independent Python processes. It does not load generation models or synthesize audio.

Run `python -m unittest discover -s generation/tests -v` in an environment with NumPy and PyTorch.
