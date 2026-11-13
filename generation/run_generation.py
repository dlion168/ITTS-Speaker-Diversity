#!/usr/bin/env python3
"""Portable generation stages for the current experiment drivers.

Only the seven paper systems and their prompts are selected. Use --dry-run before a
long generation run.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from generation.settings import load_settings, source_path, instruction as validate_instruction
from generation.preflight import attribute_payloads, prepare_runs, materialize_payloads, verify_wavs
SYSTEMS = ('moss-voicegen', 'parler-large', 'parler-mini', 'capspeech',
           'qwen3-voicedesign', 'voxinstruct', 'voxcpm2')
ENVIRONMENTS = {'moss-voicegen':'itts_moss', 'parler-large':'itts_parler', 'parler-mini':'itts_parler',
                'capspeech':'itts_capspeech', 'qwen3-voicedesign':'itts_qwen',
                'voxinstruct':'itts_vox', 'voxcpm2':'itts_voxcpm2'}


def tasks(kind, models, data, settings=None):
    settings = settings or load_settings(kind)
    result=[]
    for model in models:
        if kind=='sampling_sweep':
            if model not in settings['systems']:continue
            for value in settings['values']:
                for seed in settings['seeds']:
                    prefix='temp' if model=='moss-voicegen' else 'cfg'
                    option=settings['systems'][model]['flag']
                    label=str(float(value))
                    result.append({'model':model,'input':source_path(settings['text_source']),
                                   'output':data/f'outputs/param_sweep/{model}/{prefix}_{label}/seed_{seed}',
                                   'seed':seed,'extra':[option,str(value)],'expected_rows':settings['expected_rows']})
        elif kind=='overall':
            for part in settings['partitions']:
                result.append({'model':model,'input':source_path(settings['text_source'],part),
                               'output':data/f'outputs/overall/{model}/sample{part}','extra':[],
                               'expected_rows':settings['expected_rows']})
        else:
            for category,values in settings['categories'].items():
                per_value=settings['rows_per_cell'] if category=='gender_x_age' else settings['rows_per_value']
                result.append({'model':model,'input':data/f'generation_inputs/{category}.json',
                               'output':data/f'outputs/attributes/{model}/{category}',
                               'extra':['--attribute_category',category], 'expected_rows':per_value*len(values)})
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind',choices=['overall','sampling_sweep','attributes'])
    parser.add_argument('--models',nargs='+',choices=SYSTEMS,default=list(SYSTEMS))
    parser.add_argument('--data-root',type=Path,default=Path(os.environ.get('ITTS_DATA_ROOT',ROOT/'data')))
    parser.add_argument('--python-map',type=Path,default=os.environ.get('GENERATION_PYTHON_MAP'))
    parser.add_argument('--env-root',type=Path,default=Path(os.environ.get('GENERATION_ENV_ROOT',ROOT/'envs')))
    parser.add_argument('--weights-dir',type=Path,default=Path(os.environ.get('WEIGHTS_DIR',ROOT/'weights')))
    parser.add_argument('--config',type=Path,help='Experiment YAML; defaults to configs/experiments/<kind>.yaml')
    parser.add_argument('--instruction','--default-style-prompt','--default_style_prompt',dest='instruction',
                        help='Explicit neutral instruction override for overall/sampling_sweep')
    parser.add_argument('--batch-size',type=int,default=None)
    parser.add_argument('--seed',type=int,default=None,
                        help='Override the new overall/attribute run seed; sweep seeds are selected in YAML')
    parser.add_argument('--limit',type=int,default=None)
    parser.add_argument('--device',default=os.environ.get('GENERATION_DEVICE','cuda'))
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    if args.limit is not None and args.limit<1:parser.error('--limit must be positive')
    if args.batch_size is not None and args.batch_size<1:parser.error('--batch-size must be positive')
    try:
        settings=load_settings(args.kind,args.config)
        if args.kind=='sampling_sweep' and args.seed is not None:
            raise ValueError('Sweep seeds come from YAML seeds; use --config instead of --seed')
        seed=settings['generation_seed'] if args.kind=='sampling_sweep' else args.seed if args.seed is not None else int(os.environ.get('GENERATION_SEED',settings['generation_seed']))
        if not 0<=seed<2**32:raise ValueError('New-run seed must be in [0, 2**32)')
        if args.instruction is not None:
            if args.kind=='attributes':raise ValueError('Attribute instructions come from exact per-row prompts, not a global override')
            settings['instruction']=validate_instruction(args.instruction)
            settings['instruction_config']=None
    except (ValueError,OSError) as error:parser.error(str(error))
    data=args.data_root.resolve()
    try:
        mapping=json.loads(Path(args.python_map).read_text()) if args.python_map else {}
        if not isinstance(mapping,dict):raise ValueError('--python-map must contain a JSON object')
        planned=tasks(args.kind,args.models,data,settings)
    except (ValueError,OSError) as error:parser.error(str(error))
    if not planned:parser.error('No paper generation tasks selected')
    commands=[]
    for task in planned:
        value=mapping.get(task['model'],str(args.env_root/ENVIRONMENTS[task['model']]/'bin/python'))
        runtime=value if isinstance(value,list) else [value]
        if not runtime or any(not isinstance(part,str) or not part or re.search(r'\b(?:hf_|gh[pousr]_)[A-Za-z0-9]{20,}',part) for part in runtime):
            parser.error('Runtime argv must be nonempty and must not contain credentials; use environment variables')
        command=runtime+[str(ROOT/'generation/run_seeded_infer.py'),'--model',task['model'],
                         '--input',str(task['input']),'--output_dir',str(task['output']),
                          '--weights_dir',str(args.weights_dir),'--device',args.device,*task['extra']]
        if args.kind in {'overall','sampling_sweep'}:
            command+=['--default_style_prompt',settings['instruction']]
        if args.batch_size is not None:command+=['--batch_size',str(args.batch_size)]
        if args.limit is not None:command+=['--limit',str(args.limit)]
        task_seed=task.get('seed',seed)
        commands.append({'model':task['model'],'seed':task_seed,'command':command,'output':str(task['output'])})
    if args.dry_run:
        print(json.dumps({'kind':args.kind,'config':settings['config'],'instruction':settings.get('instruction'),'tasks':commands},indent=2));return 0
    # Validate every runtime before creating inputs or generating any audio.
    for item in commands:
        executable=item['command'][0]
        if not shutil.which(executable):
            parser.error(f'Missing generation interpreter for {item["model"]}; supply GENERATION_PYTHON_MAP or GENERATION_ENV_ROOT')
    payloads=attribute_payloads(settings,data) if args.kind=='attributes' else {}
    prepared=prepare_runs(planned,commands,settings,payloads,args.limit)
    materialize_payloads(payloads)
    for task,item,metadata,identifiers in prepared:
        output=task['output'];output.mkdir(parents=True,exist_ok=True)
        path=output/'generation_run.json'
        if path.exists() and json.loads(path.read_text())!=metadata:
            raise ValueError(f'Existing generation directory has different generation settings: {output}; use a new data root')
        if not path.exists():
            with path.open('x') as handle:handle.write(json.dumps(metadata,indent=2)+'\n')
        receipt=output/'generation_complete.json'
        if receipt.exists():
            # Keep the previous receipt, but do not leave a stale success marker
            # in place if this invocation or its resumed output check fails.
            previous=output/('generation_complete.previous-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')+'.json')
            receipt.rename(previous)
        env={**os.environ,'ITTS_EXPERIMENT_SEED':str(item['seed']),'PYTHONHASHSEED':str(item['seed']),
             'ITTS_INVOCATION_METADATA':str(output/'runtime_invocations.jsonl')}
        subprocess.run(item['command'],env=env,check=True,cwd=ROOT)
        verify_wavs(output,identifiers)
        receipt.write_text(json.dumps({'status':'complete','validated_pcm_wavs':len(identifiers),
                                      'checked_at':datetime.now(timezone.utc).isoformat(),
                                      'input_sha256':metadata['input_sha256'],
                                      'config_sha256':settings['config']['sha256']},indent=2)+'\n')
    return 0


if __name__=='__main__':raise SystemExit(main())
