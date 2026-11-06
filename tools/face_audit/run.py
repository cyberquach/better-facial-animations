#!/usr/bin/env python3
"""Read-only local NPC regression audit. Requires numpy, lz4, a C++17 compiler.
No download, game launch or installation. Missing coverage is never a pass.
"""
import argparse, concurrent.futures, hashlib, json, os, subprocess, sys, shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--game', type=Path, required=True)
    ap.add_argument('--profile', required=True)
    ap.add_argument('--baseline', type=Path, required=True)
    ap.add_argument('--baseline-include', type=Path, help='matching baseline include tree when its Head layout differs')
    ap.add_argument('--stimuli', type=Path, required=True, help='JSON model number -> sex (0/1) -> coefficient file')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--jobs', type=int, default=4)
    ap.add_argument('--reuse-inventory', action='store_true')
    ap.add_argument('--installed', action='store_true')
    ap.add_argument('--full', action='store_true')
    ap.add_argument('--strength-sweep', action='store_true')
    ap.add_argument('--limit', type=int)
    args = ap.parse_args()
    if args.jobs < 1: ap.error('--jobs must be positive')
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    if any(out.glob('*-v?-*.jsonl')):
        ap.error('output already contains audit results; choose a fresh directory')
    required=[args.game/'MO2/profiles'/args.profile, args.baseline, args.stimuli]
    if args.baseline_include:required.append(args.baseline_include/'a2f/animation/direct.h')
    if args.reuse_inventory:required.append(out/'records.json')
    missing=[str(p) for p in required if not p.exists()]
    if missing:
        (out/'summary.json').write_text(json.dumps({'status':'blocked_input','missing':missing},indent=2))
        print('Missing required audit inputs:', ', '.join(missing),flush=True)
        return 2
    tool = out/'tool'
    shutil.copytree(HERE,tool,ignore=shutil.ignore_patterns('__pycache__'),dirs_exist_ok=True)
    shutil.copytree(REPO/'include',out/'include',dirs_exist_ok=True)
    shutil.copy2(REPO/'src/animation/direct.cc',out/'candidate.cc')
    shutil.copy2(args.baseline.resolve(),out/'baseline.cc')
    if args.baseline_include:
        shutil.copytree(args.baseline_include,out/'baseline-include',dirs_exist_ok=True)
    (out/'models').mkdir(exist_ok=True)
    for name in ['A2FDirect.bin','A2FDirect3.bin']:shutil.copy2(REPO/'dist'/name,out/'models'/name)
    files=[*sorted((out/'models').glob('*.bin')),out/'candidate.cc',out/'baseline.cc',*sorted((out/'include').rglob('*.h')),*sorted(tool.rglob('*.py')),*sorted(tool.rglob('*.cc')),*sorted((out/'baseline-include').rglob('*.h'))]
    (out/'source-manifest.json').write_text(json.dumps({str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2))
    env = dict(os.environ, BFA_AUDIT_GAME=str(args.game.resolve()), BFA_AUDIT_PROFILE=args.profile,
               BFA_AUDIT_OUTPUT=str(out), BFA_AUDIT_BASELINE=str(out/'baseline.cc'),
               BFA_AUDIT_STIMULI=str(args.stimuli.resolve()), BFA_AUDIT_SOURCE=str(out/'candidate.cc'))
    def run(label, command, allowed=(0,)):
        print(label, flush=True)
        with (out / (label+'.log')).open('w') as log:
            result=subprocess.run(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=7200)
        if result.returncode not in allowed:raise subprocess.CalledProcessError(result.returncode,command)
        return result.returncode
    for name, source in [('old', out/'baseline.cc'), ('new', out/'candidate.cc')]:
        includes = out/'baseline-include' if name == 'old' and args.baseline_include else out/'include'
        defines = ['-DBASELINE'] if name == 'old' else []
        header = (includes/'a2f/animation/direct.h').read_text()
        if 'lip_neutral_w' in header:
            defines.append('-DBFA_AUDIT_NEUTRAL_WEIGHTS')
        if 'calibrated_lips' in header:
            defines.append('-DBFA_AUDIT_CALIBRATION')
        elif 'lip_repairs' not in header:
            defines.append('-DBFA_AUDIT_LEGACY_HEAD')
        run('compile-'+name, ['g++','-std=c++17','-O2','-shared','-fPIC','-I'+str(includes)]+defines+[str(source),str(tool/'bridge.cc'),'-o',str(out/(name+'.so'))])
    if not args.reuse_inventory: run('inventory', [sys.executable,str(tool/'assets.py')])
    if args.installed and not (args.reuse_inventory and (out/'installed.json').exists()):
        run('installed-inventory',[sys.executable,str(tool/'installed.py')])
    tasks = [(model, scope, shard) for model in [2,3] for scope in (['active','installed'] if args.installed else ['active']) for shard in range(args.jobs)]
    def worker(task):
        model,scope,shard=task; label=f'{scope}-v{model}-{shard}'
        cmd=[sys.executable,str(tool/'audit.py'),'--model',str(model),'--out',label+'.jsonl','--shard',str(shard),'--shards',str(args.jobs)]
        if scope=='installed':cmd+=['--installed']
        if args.full:cmd+=['--full']
        if args.strength_sweep:cmd+=['--strength-sweep']
        if args.limit:cmd+=['--limit',str(args.limit)]
        run(label,cmd)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        list(pool.map(worker,tasks))
    code=run('summary',[sys.executable,str(tool/'summarize.py')],allowed=(0,1,2))
    print(out/'summary.json',flush=True)
    return code

if __name__=='__main__':sys.exit(main())
