#!/usr/bin/env python3
"""Run the pinned follower teeth matrix; local captures only, no downloads."""
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
    out = (args.output or ROOT / 'build/follower-teeth' /
           time.strftime('%Y%m%d-%H%M%S')).resolve()
    out.mkdir(parents=True, exist_ok=False)
    manifest_path = ROOT / 'tests/follower_teeth.json'
    manifest = json.loads(manifest_path.read_text())
    report = {'status': 'blocked', 'results': [], 'source_sha256': {}}
    code = [ROOT / 'src/direct.cc', ROOT / 'src/direct.h', ROOT / 'src/live_core.h',
            ROOT / 'tests/follower_teeth_test.cc', Path(__file__), manifest_path]
    report['source_sha256'] = {str(p.relative_to(ROOT)): digest(p) for p in code}
    try:
        if args.fixtures is None:
            raise FileNotFoundError('supply --fixtures with the local capture directory')
        # Validate the complete matrix before compiling or executing any case.
        for case in manifest['fixtures']:
            path = args.fixtures / case['file']
            if digest(path) != case['sha256']:
                raise ValueError('fixture SHA-256 mismatch: ' + str(path))
        anatomy = manifest['anatomy']
        regions = args.fixtures / anatomy['file']
        if digest(regions) != anatomy['sha256']:
            raise ValueError('anatomy SHA-256 mismatch: ' + str(regions))
        report['anatomy'] = anatomy
        exe = out / 'follower_teeth_test'
        command = shlex.split(os.environ.get('HOST_CXX', 'g++')) + [
            '-std=c++17', '-O2', '-Wall', '-Wextra', '-I', str(ROOT / 'src'),
            str(ROOT / 'src/direct.cc'), str(ROOT / 'tests/follower_teeth_test.cc'),
            '-o', str(exe)]
        report['compile_command'] = command
        with (out / 'compile.log').open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=180)
        report['test_sha256'] = digest(exe)
        for case in manifest['fixtures']:
            for name in ('A2FDirect.bin', 'A2FDirect3.bin'):
                model = ROOT / 'dist' / name
                command = [str(exe), str((args.fixtures / case['file']).resolve()),
                           str(model), str(regions.resolve())]
                result = subprocess.run(command, capture_output=True, text=True,
                                        timeout=90)
                report['results'].append({
                    'fixture': case['file'], 'fixture_sha256': case['sha256'],
                    'model': name, 'model_sha256': digest(model), 'command': command,
                    'returncode': result.returncode, 'stdout': result.stdout,
                    'stderr': result.stderr})
                print(case['file'], name, 'pass' if result.returncode == 0 else 'FAIL',
                      flush=True)
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
