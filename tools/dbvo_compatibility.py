#!/usr/bin/env python3
"""Verify a local DBVO 2 DLL against the version used for the speech regression tests."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
DLL_LIMIT = 16 * 1024 * 1024


class Blocked(Exception):
    pass


def inspect(data, baseline):
    digest = hashlib.sha256(data).hexdigest()
    result = {'sha256': digest, 'size': len(data)}
    try:
        pe = struct.unpack_from('<I', data, 0x3c)[0]
        valid = (data[:2] == b'MZ' and data[pe:pe+4] == b'PE\0\0'
                 and struct.unpack_from('<H', data, pe+4)[0] == 0x8664
                 and struct.unpack_from('<H', data, pe+22)[0] & 0x2000
                 and struct.unpack_from('<H', data, pe+24)[0] == 0x20b)
    except struct.error:
        valid = False
    result['x64_pe_dll'] = bool(valid)
    result['status'] = ('reviewed_binary' if valid and digest == baseline['sha256']
                        else 'needs_review')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dll', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args(argv)
    report = {'checked_at': datetime.now(timezone.utc).isoformat(),
              'scope': 'Exact reviewed binary identity, not an in-game compatibility certification'}
    try:
        baseline = json.loads((ROOT / 'tests/dbvo2_baseline.json').read_text())
        with args.dll.open('rb') as source:
            data = source.read(DLL_LIMIT + 1)
        if len(data) > DLL_LIMIT:
            raise Blocked('DLL exceeds size limit')
        report.update(inspect(data, baseline))
        report['reviewed_version'] = baseline['version']
        code = 0 if report['status'] == 'reviewed_binary' else 1
    except (Blocked, OSError, ValueError, KeyError, TypeError) as exc:
        report.update(status='blocked', reason=str(exc) if isinstance(exc, Blocked)
                      else f'Check could not complete ({type(exc).__name__})')
        code = 2
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
