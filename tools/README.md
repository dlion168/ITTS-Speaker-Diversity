# tools/

| File | What it does | Status |
|---|---|---|
| `migrate_from_old_repo.sh` | Copies every file this repo needs from the old `ITTS_diversity` repo into the new layout, runs `filter_results.py`, and lists files it could not find. Run from the repo root: `bash tools/migrate_from_old_repo.sh /path/to/ITTS_diversity` | in repo |
| `filter_results.py` | Removes what the paper does not use: `non_binary` prompts, and systems / categories / values outside the paper from `results/attributes/raw_*.csv`. Prints how many rows were kept. | in repo |
