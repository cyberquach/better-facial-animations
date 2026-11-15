#!/usr/bin/env python3
"""Run the pinned beard clearance matrix; local captures only, no downloads."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixtures', type=Path, help='directory of pinned captures')
    parser.add_argument('--output', type=Path, help='new evidence directory')
    args = parser.parse_args()
    out = (args.output or ROOT / 'build/beard-clearance' /
           time.strftime('%Y%m%d-%H%M%S')).resolve()
    out.mkdir(parents=True, exist_ok=False)
    manifest_path = ROOT / 'tests/beard_clearance.json'
    manifest = json.loads(manifest_path.read_text())
    report = {'status': 'blocked', 'results': [], 'source_sha256': {}}
    code = [ROOT / 'src/animation/direct.cc', ROOT / 'include/a2f/animation/direct.h',
            ROOT / 'include/a2f/inference/live_core.h',
            ROOT / 'tests/beard_clearance_test.cc', Path(__file__), manifest_path]
    report['source_sha256'] = {str(p.relative_to(ROOT)): digest(p) for p in code}
    try:
        if args.fixtures is None:
            raise FileNotFoundError('supply --fixtures with the local capture directory')
        # Validate the complete matrix before compiling or executing any case.
        pinned = manifest['heads'] + list(manifest['speech'].values())
        for case in pinned:
            path = args.fixtures / case['file']
            if digest(path) != case['sha256']:
                raise ValueError('fixture SHA-256 mismatch: ' + str(path))
        model = ROOT / 'dist' / manifest['model']['file']
        if digest(model) != manifest['model']['sha256']:
            raise ValueError('model SHA-256 mismatch: ' + str(model))
        exe = out / 'beard_clearance_test'
        command = shlex.split(os.environ.get('HOST_CXX', 'g++')) + [
            '-std=c++17', '-O2', '-Wall', '-Wextra', '-I', str(ROOT / 'include'),
            str(ROOT / 'src/animation/direct.cc'),
            str(ROOT / 'tests/beard_clearance_test.cc'), '-o', str(exe)]
        report['compile_command'] = command
        with (out / 'compile.log').open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=180)
        report['test_sha256'] = digest(exe)
        for case in manifest['heads']:
            speech = manifest['speech']['female' if case['female'] else 'male']
            command = [str(exe), str((args.fixtures / case['file']).resolve()),
                       str(model), str((args.fixtures / speech['file']).resolve()),
                       str(case['min_visible']), str(case['max_clipped']),
                       str(case['max_stretched']), str(case['min_withdrawn_clipped'])]
            result = subprocess.run(command, capture_output=True, text=True,
                                    timeout=300)
            report['results'].append({
                'fixture': case['file'], 'command': command,
                'returncode': result.returncode, 'stdout': result.stdout,
                'stderr': result.stderr})
            print(case['file'], 'pass' if result.returncode == 0 else 'FAIL',
                  result.stderr.strip().replace('\n', '; '), flush=True)
        report['manifest'] = manifest
        report['status'] = ('fail' if any(r['returncode'] for r in report['results'])
                            else 'pass')
    except FileNotFoundError as exc:
        report['reason'] = str(exc)
    except (ValueError, subprocess.SubprocessError, OSError) as exc:
        report.update(status='fail', reason=str(exc))
    (out / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
    print(report.get('reason', report['status']))
    print(out / 'result.json')
    return {'pass': 0, 'fail': 1, 'blocked': 2}[report['status']]


if __name__ == '__main__':
    raise SystemExit(main())
