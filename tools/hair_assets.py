"""Replay pinned installed-hair fixtures through the current production filter."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys

from compatibility import run_command
import commonlib_build

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_output(data, expectations, counts):
    if len(data) != expectations['queries'] * 5:
        raise ValueError('truncated or unexpected production probe output')
    assertions = expectations['assertions']
    actual_counts = {'excluded': sum(a['admitted'] is False for a in assertions),
                     'preserved': sum(a['admitted'] is True for a in assertions)}
    if actual_counts != counts:
        raise ValueError('anatomical assertion counts differ from the pinned manifest')
    failures = []
    for assertion in assertions:
        index = assertion['query']
        if not 0 <= index < expectations['queries']:
            raise ValueError('assertion query index outside corpus')
        admitted = bool(data[index * 5 + 4] & 4)
        if admitted != assertion['admitted']:
            failures.append({**assertion, 'actual_admitted': admitted})
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixtures', type=Path, help='directory containing pinned local corpus')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = ROOT / 'tests/hair_assets.json'
    code = [Path(__file__), ROOT / 'tools/compatibility.py', ROOT / 'tools/commonlib_build.py', manifest_path,
            ROOT / 'tests/hair_assets_probe.cc', ROOT / 'tests/head_parts_test.cc',
            ROOT / 'src/animation/head_cache.cc', ROOT / 'src/animation/direct.cc']
    code += sorted((ROOT / 'include').rglob('*.h'))
    report = {'status': 'blocked', 'source_sha256':
              {str(p.relative_to(ROOT)): digest(p) for p in code}}
    try:
        manifest = json.loads(manifest_path.read_text())
        if args.fixtures is None:
            raise FileNotFoundError('supply the pinned installed-hair corpus with --fixtures')
        fixtures = args.fixtures.resolve()
        report['fixture_sha256'] = {}
        for name, expected in manifest['files'].items():
            path = fixtures / name
            actual = digest(path)
            report['fixture_sha256'][name] = actual
            if actual != expected:
                raise ValueError('fixture hash mismatch: ' + name)
        expectations = json.loads((fixtures / 'expectations.json').read_text())
        if expectations['queries'] != manifest['queries']:
            raise ValueError('query count differs from pinned manifest')
        exe = out / 'hair_assets_probe.exe'
        commonlib_build.runtime_files(out)
        compile_cmd = [sys.executable, str(ROOT / 'tools/commonlib_build.py'), '--case',
                       json.dumps({'test': 'hair_assets_probe', 'sources': [
                           'tests/hair_assets_probe.cc', 'src/animation/direct.cc']}), str(exe)]
        report['compile_command'] = compile_cmd
        rc = run_command(compile_cmd, out / 'compile.log', 180)
        if rc:
            raise RuntimeError('probe compilation failed; see compile.log')
        report['test_sha256'] = digest(exe)
        output = out / 'admission.bin'
        command = shlex.split(os.environ.get('WINE', 'wine64')) + [
            str(exe), str(fixtures / 'queries.bin'), str(output)]
        report['run_command'] = command
        rc = run_command(command, out / 'probe.log', 180)
        if rc:
            raise RuntimeError('production probe failed; see probe.log')
        failures = check_output(output.read_bytes(), expectations, manifest['assertions'])
        report.update(status='fail' if failures else 'pass', queries=manifest['queries'],
                      assertions=manifest['assertions'], failure_count=len(failures),
                      failures=failures, output_sha256=digest(output), scope=manifest['scope'])
    except FileNotFoundError as exc:
        report['reason'] = str(exc)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        report.update(status='fail', reason=str(exc))
    (out / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
    print('hair-assets: ' + report['status'])
    if report.get('reason'):
        print(report['reason'])
    return {'pass': 0, 'fail': 1, 'blocked': 2}[report['status']]


if __name__ == '__main__':
    sys.exit(main())
