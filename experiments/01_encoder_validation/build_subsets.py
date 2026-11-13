"""Step 1 of the encoder validation (paper §3.2): controlled LibriSpeech subsets.

For each k in configs/experiments/encoder_validation.yaml, draw 100 subsets of N = 240
utterances: k distinct speakers of train-clean-100, 240/k utterances from each (without
replacement). Speakers are drawn from those with at least 240/min(k) utterances.
The reference pool R for Chamfer and MMD is one utterance of every speaker, drawn first and
never used in a subset.

    python experiments/01_encoder_validation/build_subsets.py --librispeech_dir <.../LibriSpeech/train-clean-100>

Writes data/librispeech_validation/subsets.jsonl (one subset per line: k, subset_id, utts),
reference_pool.txt (R) and utterances.txt (every utterance id the subsets and R use, sorted).
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]


def speaker_utterances(libri_dir: Path):
    """{speaker id: sorted utterance ids} from <speaker>/<chapter>/<spk>-<chapter>-<utt>.flac."""
    spk = defaultdict(list)
    for f in libri_dir.glob("*/*/*.flac"):
        spk[f.parent.parent.name].append(f.stem)
    return {s: sorted(u) for s, u in sorted(spk.items())}


def main():
    cfg = yaml.safe_load((ROOT / "configs/experiments/encoder_validation.yaml").read_text())
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--librispeech_dir", required=True, type=Path)
    ap.add_argument("--out_dir", type=Path, default=ROOT / "data/librispeech_validation")
    ap.add_argument("--seed", type=int, default=cfg["seed"])
    a = ap.parse_args()

    ks, N, n_sub = cfg["k"], cfg["subset_size"], cfg["subsets_per_k"]
    assert all(N % k == 0 for k in ks), "240/k must be an integer"
    spk = speaker_utterances(a.librispeech_dir)
    rng = np.random.default_rng(a.seed)
    reference = [u[rng.integers(len(u))] for u in spk.values()]
    held = set(reference)
    spk = {s: [x for x in u if x not in held] for s, u in spk.items()}
    need = N // min(ks)
    pool = [s for s, u in spk.items() if len(u) >= need]
    print(f"{len(spk)} speakers, {sum(map(len, spk.values()))} utterances; {len(pool)} with >= {need} utterances")
    assert len(pool) >= max(ks)

    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / "reference_pool.txt").write_text("\n".join(reference) + "\n")
    used = set(reference)
    with open(a.out_dir / "subsets.jsonl", "w") as f:
        for k in ks:
            per = N // k
            for sid in range(n_sub):
                speakers = rng.choice(len(pool), size=k, replace=False)
                utts = []
                for i in sorted(speakers):
                    u = spk[pool[i]]
                    utts += [u[j] for j in sorted(rng.choice(len(u), size=per, replace=False))]
                used.update(utts)
                f.write(json.dumps({"k": k, "subset_id": sid, "utts_per_speaker": per, "utts": utts}) + "\n")
    (a.out_dir / "utterances.txt").write_text("\n".join(sorted(used)) + "\n")
    print(f"{len(ks) * n_sub} subsets -> {a.out_dir / 'subsets.jsonl'}; {len(reference)} reference utterances; {len(used)} distinct utterances")


if __name__ == "__main__":
    main()
