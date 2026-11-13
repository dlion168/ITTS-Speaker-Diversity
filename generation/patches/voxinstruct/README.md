# VoxInstruct batch-inference patch

These changes add **cross-sample batched inference** to the upstream
[VoxInstruct](https://github.com/thuhcsi/VoxInstruct) repo, which the
generation pipeline clones into `weights/voxinstruct/repo/` (a gitignored
path). Because that clone is not tracked here, the modifications are versioned
in this folder instead.

## What it does

- **`inference.py`** — adds `generate_batched()`: a lockstep cross-sample AR
  loop (no reference-audio case) that advances the whole batch one token per
  step, tracking per-sample ST→AT phase switches and EOS with boolean masks.
  Also adds a gated batched-NAR path (see below). Controlled by `--batch_size`.
- **`model/ar.py`** — AR `predict` adjustments needed for the batched KV-cache
  path used by `generate_batched`.
- **`model/nar.py`** — adds `predict_batched()`: batches the NAR transformer
  forward across samples via padding + an audio attention mask (upstream
  `predict` hardcodes batch=1). Only the cheap confidence selection/scatter
  stays per-sample.

## Environment flags (read by the ITTS backend / inference.py)

- `VOX_AR_BATCH_SIZE` (default 16) — cross-sample AR batch size.
- `VOX_NAR_BATCH=1` — enable batched NAR (default off = original per-sample NAR).

The paper's audio uses CFG 1 / 1 / 1 and 1 NAR step (`configs/systems.yaml`), the defaults of
`generation/infer.py` (`--vox_cfg_st_on_text`, `--vox_cfg_at_on_text`, `--vox_cfg_at_on_st`,
`--vox_nar_iter_steps`).

## Upstream base commit

The patch applies on top of VoxInstruct commit `e953295485bd88a293b37735c98b362554f4635b`.

## How to apply

After `download_weights.py` clones VoxInstruct into `weights/voxinstruct/repo/`:

```bash
cp generation/patches/voxinstruct/files/inference.py weights/voxinstruct/repo/inference.py
cp generation/patches/voxinstruct/files/model/ar.py weights/voxinstruct/repo/model/ar.py
cp generation/patches/voxinstruct/files/model/nar.py weights/voxinstruct/repo/model/nar.py
```

Use the files under `files/`. `batch_inference.patch` is a diff of the main changes against the
upstream commit, for reading; it does not include two speed-ups that `files/` has (NAR
text-encoding cache, vectorized AR sampling).
