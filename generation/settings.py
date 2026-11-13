"""Validated generation settings from experiment YAML, without importing models."""
import hashlib
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent.parent
VALUES = {
    'gender': {'female', 'male'},
    'age': {'child', 'young_adult', 'middle_aged', 'elderly'},
    'accent': {'north_american', 'british_isles', 'oceanian', 'south_asian', 'south_african', 'germanic'},
    'emotion': {'angry', 'happy', 'neutral', 'sad'},
}
CELLS = {f'{gender}_{age}' for gender in VALUES['gender'] for age in VALUES['age']}
SWEEP_FLAGS = {
    'moss-voicegen': {'--moss_audio_temperature'},
    'voxcpm2': {'--voxcpm_cfg_value'},
    'voxinstruct': {'--vox_cfg_st_on_text', '--vox_cfg_at_on_text', '--vox_cfg_at_on_st'},
}


def load_yaml(path):
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError('PyYAML is required in GENERATION_DRIVER_PYTHON to read experiment settings') from error
    path = Path(path).resolve()
    raw = path.read_bytes()
    value = yaml.safe_load(raw)
    if not isinstance(value, dict):
        raise ValueError(f'Experiment settings must be a YAML mapping: {path.name}')
    return value, {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest()}


def positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return value


def sequence(values, name):
    if not isinstance(values, list) or not values:
        raise ValueError(f'{name} must be a nonempty list')
    try:
        unique = set(values)
    except TypeError as error:
        raise ValueError(f'{name} must contain scalar values') from error
    if len(unique) != len(values):
        raise ValueError(f'{name} contains duplicates')
    return values


def instruction(value):
    if not isinstance(value, str) or not value.strip() or value == 'TODO':
        raise ValueError('A nonempty neutral instruction is required')
    if re.search(r'\b(?:hf_|gh[pousr]_)[A-Za-z0-9]{20,}', value):
        raise ValueError('Instruction contains credential-like material; value withheld')
    return value


def source_path(value, partition=None):
    if not isinstance(value, str) or not value:
        raise ValueError('text_source must be a nonempty local path template')
    # Accept {1,2,3} as a placeholder for {partition}.
    value = value.replace('{1,2,3}', '{partition}')
    if partition is not None:
        try:
            value = value.format(partition=partition)
        except (KeyError, ValueError) as error:
            raise ValueError('text_source only supports the {partition} placeholder') from error
    result = Path(value)
    return result if result.is_absolute() else ROOT / result


def load_settings(kind, path=None):
    filename = {'overall': 'overall', 'sampling_sweep': 'sampling_sweep', 'attributes': 'attributes'}[kind]
    config, origin = load_yaml(path or ROOT / f'configs/experiments/{filename}.yaml')
    result = {'kind': kind, 'config': origin}
    seed = config.get('generation_seed', 0)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError('generation_seed must be an integer in [0, 2**32)')
    result['generation_seed'] = seed
    if kind in {'overall', 'sampling_sweep'}:
        if kind == 'sampling_sweep' and 'instruction' not in config:
            overall, neutral_origin = load_yaml(ROOT / 'configs/experiments/overall.yaml')
            result['instruction_config'] = neutral_origin
            result['instruction'] = instruction(overall.get('instruction'))
        else:
            result['instruction'] = instruction(config.get('instruction'))
    if kind == 'overall':
        parts = sequence(config.get('partitions'), 'partitions')
        if any(isinstance(part, bool) or not isinstance(part, int) or part not in (1, 2, 3) for part in parts):
            raise ValueError('partitions must select the available source partitions 1, 2 or 3')
        result.update(partitions=parts, text_source=config.get('text_source'),
                      expected_rows=positive_integer(config.get('texts_per_partition'), 'texts_per_partition'))
    elif kind == 'sampling_sweep':
        values = sequence(config.get('values'), 'values')
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0 for value in values):
            raise ValueError('Sweep values must be finite positive numbers')
        seeds = sequence(config.get('seeds'), 'seeds')
        if any(isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32 for seed in seeds):
            raise ValueError('Sweep seeds must be integers in [0, 2**32)')
        systems = config.get('systems')
        if not isinstance(systems, dict) or not systems:
            raise ValueError('Sweep systems must be a nonempty mapping')
        for system, setting in systems.items():
            if system not in SWEEP_FLAGS or not isinstance(setting, dict) or setting.get('flag') not in SWEEP_FLAGS[system]:
                raise ValueError(f'Unsupported or unresolved sweep flag for {system}')
            expected = 'temperature' if system == 'moss-voicegen' else 'cfg'
            if setting.get('param') != expected:
                raise ValueError(f'Incorrect sweep parameter kind for {system}')
        result.update(values=values, seeds=seeds, systems=systems,
                      text_source=config.get('text_source', 'data/emilia/overall_partition_1.jsonl'),
                      expected_rows=positive_integer(config.get('utterances_per_config'), 'utterances_per_config'))
    else:
        single = config.get('single_attribute', {})
        if not isinstance(single, dict):
            raise ValueError('single_attribute must be a mapping')
        unknown = set(single) - set(VALUES) - {'utterances_per_value', 'templates_per_value'}
        if unknown:
            raise ValueError(f'Unsupported single-attribute settings: {sorted(unknown)}')
        categories = {}
        for category, allowed in VALUES.items():
            if category in single:
                selected = sequence(single[category], category)
                if set(selected) - allowed:
                    raise ValueError(f'Unsupported values for {category}')
                categories[category] = selected
        cross = config.get('gender_x_age', {})
        if cross:
            cells = sequence(cross.get('cells'), 'gender_x_age.cells')
            if set(cells) - CELLS:
                raise ValueError('Unsupported gender-age cells')
            categories['gender_x_age'] = cells
        if not categories:
            raise ValueError('No paper attribute categories selected')
        sources = config.get('prompt_sources', {})
        if not isinstance(sources, dict):
            raise ValueError('prompt_sources must be a mapping')
        result.update(categories=categories,
                      rows_per_value=positive_integer(single.get('utterances_per_value', 3000), 'utterances_per_value'),
                      rows_per_cell=positive_integer(cross.get('utterances_per_cell', 6000), 'utterances_per_cell'),
                      single_source=source_path(sources.get('single_attribute', 'data/prompts/single_attribute.json')),
                      cross_source=source_path(sources.get('gender_x_age', 'data/prompts/gender_x_age.json')))
    return result
