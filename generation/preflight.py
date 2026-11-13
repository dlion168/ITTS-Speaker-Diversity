"""Validate the generation inputs and output folders before writing anything."""
from collections import Counter
import csv
import hashlib
import json
import re
import wave


def attribute_payloads(settings, data):
    payloads = {}
    sources = {}
    for category, allowed in settings['categories'].items():
        source = settings['cross_source'] if category == 'gender_x_age' else settings['single_source']
        if source not in sources:
            value = json.loads(source.read_text())
            if not isinstance(value, list):
                raise ValueError(f'Prompt input must be a JSON list: {source}')
            sources[source] = value
        prefix = 'genderage' if category == 'gender_x_age' else category
        rows, counts = [], Counter()
        for row in sources[source]:
            if not isinstance(row, dict) or not isinstance(row.get('id'), str):
                raise ValueError(f'Prompt records must contain string IDs: {source}')
            identifier = row.get('id', '')
            match = re.fullmatch(re.escape(prefix) + r'-(.+)-s\d+-\d+-.+', identifier)
            if match and match[1] in allowed:
                style = row.get('style_prompt')
                if not isinstance(style, str) or not style.strip():
                    raise ValueError(f'Missing source instruction: {identifier}')
                rows.append(row)
                counts[match[1]] += 1
        count = settings['rows_per_cell'] if category == 'gender_x_age' else settings['rows_per_value']
        if counts != Counter({value: count for value in allowed}):
            raise ValueError(f'Prompt counts do not match the YAML for {category}')
        payloads[data / f'generation_inputs/{category}.json'] = (json.dumps(rows, ensure_ascii=False) + '\n').encode()
    return payloads


def input_info(path, expected_rows, limit=None, payload=None, neutral_instruction=None):
    raw = path.read_bytes() if payload is None else payload
    text = raw.decode('utf-8')
    if path.suffix == '.json':
        rows = json.loads(text)
    elif path.suffix == '.jsonl':
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    elif path.suffix == '.csv':
        rows = list(csv.DictReader(text.splitlines()))
    else:
        raise ValueError(f'Unsupported generation input format: {path.name}')
    if not isinstance(rows, list) or len(rows) != expected_rows:
        raise ValueError(f'Input row count does not match the YAML: {path.name}; expected {expected_rows}')
    names = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f'Input rows must be objects: {path.name}')
        identifier, spoken = row.get('id'), row.get('text')
        if not isinstance(identifier, str) or not identifier.strip() or '/' in identifier or '\\' in identifier:
            raise ValueError(f'Paper generation requires safe, nonempty original IDs: {path.name}')
        if not isinstance(spoken, str) or not spoken.strip():
            raise ValueError(f'Missing spoken text for {identifier}')
        if neutral_instruction is not None:
            aliases = {'style_prompt', 'description', 'prompt', 'style', 'instruction', 'voice_prompt'}
            for key, value in row.items():
                if isinstance(key, str) and key.lower().strip() in aliases and value:
                    if not isinstance(value, str) or value.strip() != neutral_instruction:
                        raise ValueError(f'Per-row instruction conflicts with the configured neutral instruction: {identifier}')
        names.append(identifier.strip())
    if len(set(names)) != len(names):
        raise ValueError(f'Duplicate generation input IDs: {path.name}')
    return {'sha256': hashlib.sha256(raw).hexdigest(), 'ids': names[:limit] if limit else names}


def prepare_runs(planned, commands, settings, payloads, limit):
    cache, prepared = {}, []
    for task, item in zip(planned, commands):
        key = (task['input'], task['expected_rows'])
        if key not in cache:
            cache[key] = input_info(task['input'], task['expected_rows'], limit, payloads.get(task['input']), settings.get('instruction'))
        info = cache[key]
        output = task['output']
        metadata = {
            'schema_version': 2, 'new_execution': True,
            'model': item['model'], 'seed': item['seed'], 'input_sha256': info['sha256'],
            'selected_rows': len(info['ids']), 'command': item['command'],
            'experiment_config': settings['config'],
            'instruction': settings.get('instruction'),
            'instruction_config': settings.get('instruction_config'),
            'nominal_seeded_sweep': settings['kind'] == 'sampling_sweep',
        }
        path = output / 'generation_run.json'
        if path.exists():
            if json.loads(path.read_text()) != metadata:
                raise ValueError(f'Existing generation directory has different generation settings: {output}; use a new data root')
        elif output.exists() and any(output.glob('*.wav')):
            raise ValueError(f'Existing audio has no generation_run.json: {output}; evaluate it with --from generated or choose a fresh generation root')
        allowed = {identifier + '.wav' for identifier in info['ids']}
        if output.exists() and any(path.name not in allowed for path in output.glob('*.wav')):
            raise ValueError(f'Unexpected WAV IDs already exist in {output}')
        prepared.append((task, item, metadata, info['ids']))
    # Check every derived input before writing any file; never replace a
    # different cached input just because a configuration has changed.
    for path, raw in payloads.items():
        if path.exists() and path.read_bytes() != raw:
            raise FileExistsError(f'Refusing to replace different generation input: {path}; use a fresh data root')
    return prepared


def materialize_payloads(payloads):
    for path, raw in payloads.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            with path.open('xb') as handle:
                handle.write(raw)


def verify_wavs(output, identifiers):
    for identifier in identifiers:
        path = output / (identifier + '.wav')
        try:
            with wave.open(str(path), 'rb') as audio:
                expected = audio.getnframes() * audio.getnchannels() * audio.getsampwidth()
                if expected <= 0 or audio.getframerate() <= 0:
                    raise ValueError('Empty or invalid PCM WAV')
                actual = 0
                while True:
                    chunk = audio.readframes(65536)
                    if not chunk:
                        break
                    actual += len(chunk)
                if actual != expected:
                    raise ValueError('Truncated PCM payload')
        except (OSError, EOFError, wave.Error, ValueError) as error:
            raise ValueError(f'Missing or invalid generated WAV: {path}') from error
