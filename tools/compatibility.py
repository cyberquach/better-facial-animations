#!/usr/bin/env python3
"""Offline BFA compatibility contracts. No game launch, install or downloads."""
import argparse
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


def build_variable(name):
    """A quoted assignment in build.sh: the plugin link line lives there only."""
    match = re.search('^' + name + '="([^"]*)"', (ROOT / 'build.sh').read_text(), re.M)
    if not match:
        raise ValueError('build.sh does not define ' + name)
    ffmpeg = os.environ.get('FFMPEG', 'deps/ffmpeg/win64')
    return match.group(1).replace('$FFMPEG', ffmpeg).split()


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


def run_case(case, result, fixtures, build_ok, version, out):
    """Fills result for one registry case: an asset check, a Python test or a
    compiled contract test (its sources plus, for one that includes the
    plugin's main source, the objects and libraries build.sh links)."""
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
        rc = run_command(result['run_command'], out / (case['id'] + '.log'), 90)
        result.update(status='pass' if rc == 0 else 'fail', returncode=rc)
        return
    if not build_ok:
        result['reason'] = 'production build failed'
        return
    exe = out / (case['test'] + '.exe')
    cmd = shlex.split(os.environ.get('CXX', 'x86_64-w64-mingw32-g++'))
    cmd += ['-std=c++17', '-O2', '-Wall', '-static', '-DBFA_VERSION="' + version + '"']
    ort = os.environ.get('ORT', 'deps/dml/microsoft.ml.onnxruntime.directml')
    cmd += ['-isystem', ort + '/build/native/include']
    cmd += case['sources']
    if case.get('plugin_objects'):
        objects = [ROOT / o for o in build_variable('PLUGIN_OBJS')]
        if any(not p.exists() for p in objects):
            result['reason'] = 'missing build objects; rerun with --build'
            return
        result['object_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in objects}
        cmd += [str(p) for p in objects] + build_variable('PLUGIN_LIBS')
    cmd += case.get('link', []) + ['-o', str(exe)]
    result['compile_command'] = cmd
    rc = run_command(cmd, out / (case['id'] + '-compile.log'), 180)
    if rc:
        result.update(status='fail', reason='compilation failed', returncode=rc)
        return
    result['test_sha256'] = hashlib.sha256(exe.read_bytes()).hexdigest()
    result['run_command'] = shlex.split(os.environ.get('WINE', 'wine64')) + [str(exe)]
    rc = run_command(result['run_command'], out / (case['id'] + '.log'), 90)
    result.update(status='pass' if rc == 0 else 'fail', returncode=rc)


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
    args = parser.parse_args()
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
    inputs = [ROOT / 'build.sh', ROOT / 'VERSION', ROOT / 'tests/compatibility.json',
              Path(__file__)]
    inputs += sorted((ROOT / 'src').glob('*')) + sorted((ROOT / 'tests').glob('*.cc'))
    inputs += sorted((ROOT / 'tests').glob('*.py'))
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
    fixtures = json.loads(args.fixtures.read_text()) if args.fixtures else {}
    for case in selected:
        result = {'id': case['id'], 'status': 'blocked'}
        report['results'].append(result)
        try:
            run_case(case, result, fixtures, build_ok, report['version'], out)
        except (OSError, ValueError, struct.error, zlib.error) as exc:
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
