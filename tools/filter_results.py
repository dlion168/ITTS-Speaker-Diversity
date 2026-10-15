"""Keep only what the paper uses.

- prompts: data/prompts/raw/*.json -> data/prompts/{single_attribute,gender_x_age}.json,
  dropping non_binary entries.
- results/attributes/raw_mpcd_ci*.csv -> mpcd_ci*.csv, dropping systems, categories,
  and values not in the paper.
Run after tools/migrate_from_old_repo.sh. Prints what was dropped.
"""
import json, os, glob
import pandas as pd

PAPER_SYSTEMS = ["moss-voicegen", "parler-large", "parler-mini", "capspeech",
                 "qwen3-voicedesign", "voxinstruct", "voxcpm2"]
DROP_CATEGORIES = ["l2_accent", "native_accent"]
DROP_VALUES = ["non_binary"]

def filter_prompts():
    raw = "data/prompts/raw"
    if not os.path.isdir(raw):
        print("skip prompts: data/prompts/raw not found"); return
    single, gxa = [], []
    for path in sorted(glob.glob(f"{raw}/*.json")):
        cat = os.path.splitext(os.path.basename(path))[0]
        entries = json.load(open(path))
        kept = [e for e in entries if not any(f"-{v}-" in str(e.get("id", "")) for v in DROP_VALUES)]
        print(f"prompts {cat}: kept {len(kept)} / {len(entries)}")
        (gxa if cat == "gender_x_age" else single).extend(kept)
    json.dump(single, open("data/prompts/single_attribute.json", "w"), indent=1, ensure_ascii=False)
    json.dump(gxa, open("data/prompts/gender_x_age.json", "w"), indent=1, ensure_ascii=False)

def filter_csv(src, dst):
    d = pd.read_csv(src)
    n0 = len(d)
    sys_col = "model" if "model" in d.columns else "system"
    d = d[d[sys_col].isin(PAPER_SYSTEMS)]
    if "category" in d.columns: d = d[~d["category"].isin(DROP_CATEGORIES)]
    if "value" in d.columns: d = d[~d["value"].isin(DROP_VALUES)]
    d = d.drop(columns=[c for c in ["vendi_all", "raw_lo", "raw_hi"] if c in d.columns])
    d = d.rename(columns={"model": "system"})
    d.to_csv(dst, index=False)
    print(f"{dst}: kept {len(d)} / {n0}")

if __name__ == "__main__":
    filter_prompts()
    for src in glob.glob("results/attributes/raw_*.csv"):
        filter_csv(src, src.replace("raw_", ""))
