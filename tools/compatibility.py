#!/usr/bin/env python3
"""Offline BFA compatibility contracts. No game launch, install or downloads."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import signal
import struct
import subprocess
import sys
import time
import zlib

import commonlib_build

ROOT = Path(__file__).resolve().parents[1]


def records(data):
    """Bounded TES4/GRUP reader, including compressed records and XXXX sizes."""
    def walk(start, end, depth=0):
        if depth > 64:
            raise ValueError('excessive GRUP nesting')
        while start < end:
            if start + 24 > end:
                raise ValueError('truncated record header')
            kind, size, flags, form = struct.unpack_from('<4sIII', data, start)
            stop = start + (size if kind == b'GRUP' else size + 24)
            if stop > end or stop < start + 24:
                raise ValueError('record exceeds container')
            if kind == b'GRUP':
                yield from walk(start + 24, stop, depth + 1)
            else:
                body = data[start + 24:stop]
                if flags & 0x40000:
                    if len(body) < 4:
                        raise ValueError('truncated compression header')
                    expected = struct.unpack_from('<I', body)[0]
                    if expected > 64 * 1024 * 1024:
                        raise ValueError('record exceeds decompression limit')
                    decoder = zlib.decompressobj()
                    body = decoder.decompress(body[4:], expected + 1)
                    if len(body) != expected or not decoder.eof or decoder.unused_data:
                        raise ValueError('invalid compressed record')
                fields, pos, extended = {}, 0, None
                while pos < len(body):
                    if pos + 6 > len(body):
                        raise ValueError('truncated subrecord header')
                    tag, length = struct.unpack_from('<4sH', body, pos)
                    pos += 6
                    if extended is not None:
                        length, extended = extended, None
                    if pos + length > len(body):
                        raise ValueError('subrecord exceeds record')
                    value = body[pos:pos + length]
                    pos += length
                    if tag == b'XXXX':
                        if length != 4:
                            raise ValueError('invalid XXXX size')
                        extended = struct.unpack('<I', value)[0]
                    else:
                        fields.setdefault(tag, []).append(value)
                if extended is not None:
                    raise ValueError('orphan XXXX')
                yield kind, form, flags, fields
            start = stop
    yield from walk(0, len(data))


def check_asset(spec, path, expected_sha256=None):
    path = Path(path).resolve()
    if path.name.lower() != spec['file'].lower():
        raise ValueError('expected filename ' + spec['file'])
    blob = path.read_bytes()
    digest = hashlib.sha256(blob).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256.lower():
        raise ValueError('fixture SHA-256 differs from pinned version')
    if blob[:4] != b'TES4':
        raise ValueError('not a TES4 plugin')
    parsed = list(records(blob))
    masters = parsed[0][3].get(b'MAST', [])
    found = []
    for wanted in spec['records']:
        # On disk the upper byte indexes the master's list, NOT the runtime
        # FE light index. Require a form owned by this plugin, not an override.
        matches = [(kind, fid, flags, fields) for kind, fid, flags, fields in parsed
                   if fid >> 24 == len(masters) and fid & 0xFFFFFF == int(wanted['id'], 16)]
        if len(matches) != 1:
            raise ValueError('missing/ambiguous owned record ' + wanted['id'])
        kind, fid, flags, fields = matches[0]
        editor = fields.get(b'EDID', [b''])[0].rstrip(b'\0').decode('utf-8')
        if kind.decode() != wanted['type'] or editor != wanted['editor_id'] or flags & 0x20:
            raise ValueError('record contract mismatch: ' + wanted['editor_id'])
        found.append({'form_id': f'{fid:08X}', 'editor_id': editor})
    return {'path': str(path), 'sha256': digest, 'pinned': expected_sha256 is not None,
            'bytes': len(blob), 'records': found, 'version_label': spec['version']}



def run_command(command, log, timeout):
    with log.open('w') as stream:
        try:
            process = subprocess.Popen(command, cwd=ROOT, stdout=stream,
                                       stderr=subprocess.STDOUT, start_new_session=True,
                                       env={**os.environ, 'WINEDEBUG': '-all'})
        except OSError as exc:
            stream.write(str(exc) + '\n')
            return 127
        try:
            return process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            # Bound our own subprocess group, not unrelated Wine/game jobs.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            stream.write('Timed out after ' + str(timeout) + ' seconds\n')
            return 124


def prepare_case(case, result, fixtures, build_ok, version, out):
    """Settles an asset check, or compiles a contract test and fills
    result['run_command'] for run_case. Safe to call concurrently: each case
    only writes its own files under out."""
    if 'records' in case:
        if case['id'] not in fixtures:
            result['reason'] = 'missing local fixture'
            return
        fixture = fixtures[case['id']]
        if isinstance(fixture, str):
            fixture = {'path': fixture}
        result.update(check_asset(case, fixture['path'], fixture.get('sha256')),
                      status='pass', scope='record identity only')
        return
    if case.get('python'):
        result['run_command'] = [sys.executable, case['python']]
        if case.get('fixture_directory'):
            fixture = fixtures.get(case['id'])
            if fixture:
                path = fixture if isinstance(fixture, str) else fixture['path']
                result['run_command'] += ['--fixtures', path]
            result['run_command'] += ['--output', str(out / case['id'])]
        return
    if not build_ok:
        result['reason'] = 'production build failed'
        return
    exe = out / (case['test'] + '.exe')
    # Tests use the same MSVC/CommonLib ABI, with explicit test hook seams.
    objects = list((ROOT / 'build/test-objects').glob('*.obj'))
    if case.get('plugin_objects'):
        if not objects:
            result['reason'] = 'missing test objects; rerun with --build'
            return
        result['object_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in objects}
    cmd = [sys.executable, str(ROOT / 'tools/commonlib_build.py'), '--case',
           json.dumps(case), str(exe)]
    result['compile_command'] = cmd
    rc = run_command(cmd, out / (case['id'] + '-compile.log'), 600)
    if rc:
        result.update(status='fail', reason='compilation failed', returncode=rc)
        return
    result['test_sha256'] = hashlib.sha256(exe.read_bytes()).hexdigest()
    result['run_command'] = shlex.split(os.environ.get('WINE', 'wine64')) + [str(exe)]


def run_case(case, result, out):
    """Runs a prepared case. Tests share the working directory (fixtures
    under dist, Data/SKSE/Plugins), so cases run one at a time."""
    if 'run_command' not in result:
        return
    timeout = 600 if case.get('fixture_directory') else 90
    rc = run_command(result['run_command'], out / (case['id'] + '.log'), timeout)
    status = 'pass' if rc == 0 else 'fail'
    if rc == 2 and case.get('fixture_directory'):
        status = 'blocked'
    result.update(status=status, returncode=rc)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tier', choices=['contracts', 'assets', 'all'], default='contracts')
    parser.add_argument('--fixtures', type=Path,
                        help='JSON object: asset case id -> extracted plugin path')
    parser.add_argument('--case', action='append', default=[],
                        help='select case id (repeatable)')
    parser.add_argument('--build', action='store_true',
                        help='fresh production build before tests')
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--output', type=Path, help='new evidence directory (must not exist)')
    parser.add_argument('--jobs', type=int, default=int(os.environ.get('JOBS', 0)) or os.cpu_count(),
                        help='parallel test compilations (default: $JOBS or the CPU count)')
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error('--jobs must be positive')
    registry = json.loads((ROOT / 'tests/compatibility.json').read_text())
    if args.list:
        print(json.dumps(registry, indent=2))
        return 0
    selected = [c for tier in ['contracts', 'assets'] if args.tier in [tier, 'all']
                for c in registry[tier] if not args.case or c['id'] in args.case]
    unknown = set(args.case) - {c['id'] for c in selected}
    if unknown or not selected:
        parser.error('no cases selected or ids outside selected tier: '
                     + ', '.join(sorted(unknown)))
    out = args.output or ROOT / 'build/compatibility' / time.strftime('%Y%m%d-%H%M%S')
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    # Source hashes include dirty edits: HEAD alone is insufficient provenance.
    inputs = [ROOT / 'build.sh', ROOT / 'tools/commonlib_build.py', ROOT / 'VERSION',
              ROOT / 'tests/compatibility.json', Path(__file__)]
    inputs += sorted((ROOT / 'src').rglob('*.cc'))
    inputs += sorted((ROOT / 'include').rglob('*.h'))
    inputs += sorted((ROOT / 'tests').glob('*.cc'))
    inputs += sorted((ROOT / 'tests').glob('*.inc'))
    inputs += sorted((ROOT / 'tests').glob('*.h'))
    inputs += sorted((ROOT / 'tests').glob('*.py'))
    inputs += [ROOT / c['python'] for c in selected if c.get('python')]
    inputs += [ROOT / 'tests/follower_teeth.json', ROOT / 'tests/mouth_lining.json', ROOT / 'tests/hair_assets.json',
               ROOT / 'tests/beard_clearance.json',
               ROOT / 'tests/emotion_performance.fixtures', ROOT / 'tests/performance.fixtures',
               ROOT / 'tests/ultra_performance.fixtures',
               ROOT / 'dist/performance.onnx', ROOT / 'dist/performance-model.json',
               ROOT / 'dist/ultra-performance.onnx', ROOT / 'dist/ultra-performance-model.json',
               ROOT / 'dist/emotion_performance.onnx', ROOT / 'dist/emotion_performance.json']
    report = {
        'schema': 1,
        'tier': args.tier,
        'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'version': (ROOT / 'VERSION').read_text().strip(),
        'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                                            text=True).strip(),
        'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in inputs if p.is_file()},
        'results': [],
        'coverage_gaps': registry['coverage_gaps'],
    }
    build_ok = True
    if args.build:
        build_ok = run_command(['sh', 'build.sh'], out / 'build.log', 900) == 0
        report['build_status'] = 'pass' if build_ok else 'fail'
    if build_ok and any(c.get('sources') for c in selected):
        commonlib_build.runtime_files(out)
    fixtures = json.loads(args.fixtures.read_text()) if args.fixtures else {}
    report['results'] = [{'id': case['id'], 'status': 'blocked'} for case in selected]

    def prepare(case, result):
        try:
            prepare_case(case, result, fixtures, build_ok, report['version'], out)
        except (OSError, ValueError, struct.error, zlib.error) as exc:
            result.update(status='fail', reason=str(exc))

    with concurrent.futures.ThreadPoolExecutor(args.jobs) as pool:
        list(pool.map(prepare, selected, report['results']))
    for case, result in zip(selected, report['results']):
        try:
            run_case(case, result, out)
        except OSError as exc:
            result.update(status='fail', reason=str(exc))
        print(case['id'] + ': ' + result['status'], flush=True)
    statuses = {r['status'] for r in report['results']}
    report['status'] = ('fail' if not build_ok or 'fail' in statuses
                        else 'blocked' if 'blocked' in statuses else 'pass')
    (out / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
    print(str(out / 'result.json'))
    return {'pass': 0, 'fail': 1, 'blocked': 2}[report['status']]


if __name__ == '__main__':
    sys.exit(main())
