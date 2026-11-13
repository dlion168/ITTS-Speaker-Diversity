"""Stand-in interpreter for shell entrypoints; never executes model code."""
import sys


def runtime(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('#!' + sys.executable + '\n' + r'''
import csv,hashlib,json,os,pathlib,sys
args=sys.argv[1:]
if args[:1]==['-u']:args=args[1:]
with open(os.environ['FIXTURE_CALLS'],'a') as handle:handle.write(json.dumps(args)+'\n')
if args[:1]==['-c']:
 print('/fixture/ca.pem');sys.exit(0)
if args[:1]==['-']:sys.exit(0)
name=pathlib.Path(args[0]).name
if name in ['download_checkpoints.py','extract_embeddings.py','half_sampling_ci.py']:sys.exit(0)
if name=='score_controllability.py':
 if os.environ.get('FIXTURE_WORKER_FAIL')=='1':sys.exit(7)
 sys.path.insert(0,str(pathlib.Path(args[0]).parent))
 from scoring_run import output_stem,sha256
 start=args.index('--csv');folders=args[1:start];options={}
 i=start
 while i<len(args):
  if args[i]=='--resume':i+=1;continue
  options[args[i]]=args[i+1];i+=2
 shard,nproc=map(int,options['--shard'].split('/'))
 for folder in folders:
  folder=pathlib.Path(folder);source=folder/options['--csv']
  with source.open() as handle:rows=list(csv.DictReader(handle))
  rows=rows[shard::nproc];attrs=options['--attrs'].split(',')
  label=folder.parent.name+'/'+folder.name;stem=output_stem(label,shard,nproc)
  out=pathlib.Path(options['--out_dir']);out.mkdir(parents=True,exist_ok=True)
  with (out/(stem+'_per_utt.csv')).open('w',newline='') as handle:
   fields=['file']+[prefix+a for a in attrs for prefix in ['target_','pred_']]
   writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
   for row in rows:
    result={'file':row['file']}
    for a in attrs:result['target_'+a]=row['target_'+a];result['pred_'+a]=row['target_'+a]
    writer.writerow(result)
  (out/(stem+'.json')).write_text(json.dumps({'label':label,'shard':str(shard)+'of'+str(nproc),'n_rows':len(rows),'missing_wav':0,
   'input_provenance':{'csv_sha256':sha256(source),'batch':int(options['--batch']),'max_s':20,'scorer_sha256':sha256(args[0])}}))
 sys.exit(0)
if name in ['make_manifests.py','scoring_run.py','merge_shards.py']:
 os.execv(sys.executable,[sys.executable,*args])
raise SystemExit('Fixture blocked unexpected script: '+name)
''')
    path.chmod(0o755)
