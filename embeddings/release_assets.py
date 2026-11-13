"""Immutable defaults for published assets; no network access during import."""
import json
from pathlib import Path


def revision_for(repo_id, explicit=None):
    if explicit:
        return explicit
    record = json.loads((Path(__file__).resolve().parents[1] / 'configs/release_assets.json').read_text())
    for kind in ['dataset', 'accent_classifiers']:
        if record[kind]['repo_id'] == repo_id:
            return record[kind]['revision']
    return None
