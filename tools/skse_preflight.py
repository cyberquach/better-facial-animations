#!/usr/bin/env python3
"""Read explicit SKSE runtime declarations without loading an untrusted DLL.

0: runtime explicitly listed; 1: explicit list excludes it; 2: no decisive list.
A match is a prerequisite, never proof that a plugin loads or works.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct


def declaration(blob):
    def unpack(fmt, offset):
        if offset < 0 or offset + struct.calcsize(fmt) > len(blob):
            raise ValueError('truncated PE data')
        return struct.unpack_from(fmt, blob, offset)
    if blob[:2] != b'MZ':
        raise ValueError('not a PE file')
    pe, = unpack('<I', 0x3c)
    if blob[pe:pe + 4] != b'PE\0\0':
        raise ValueError('invalid PE signature')
    machine, count = unpack('<HH', pe + 4)
    optional_size, = unpack('<H', pe + 20)
    optional = pe + 24
    if machine != 0x8664 or unpack('<H', optional)[0] != 0x20b:
        raise ValueError('expected x64 PE32+')
    if optional_size < 120 or unpack('<I', optional + 108)[0] < 1:
        raise ValueError('missing export directory')
    sections = []
    for i in range(count):
        size, address, raw_size, raw = unpack('<IIII', optional + optional_size + i * 40 + 8)
        if raw_size and raw + raw_size > len(blob):
            raise ValueError("truncated file-backed section")
        sections.append((address, raw_size, raw))
    def offset(rva, size):
        for address, raw_size, raw in sections:
            if address <= rva and rva + size <= address + raw_size:
                start = raw + rva - address
                if start + size <= len(blob):
                    return start
        raise ValueError('RVA outside file-backed section')
    export, export_size = unpack('<II', optional + 112)
    if not export:
        return None
    table = offset(export, 40)
    functions, names, function_rva, name_rva, ordinal_rva = unpack('<IIIII', table + 20)
    if names > 100000 or functions > 100000:
        raise ValueError('unreasonable export table')
    for i in range(names):
        name, = unpack('<I', offset(name_rva + i * 4, 4))
        start = offset(name, 1)
        end = blob.find(b'\0', start, min(len(blob), start + 512))
        if end < 0:
            raise ValueError('unterminated export name')
        if blob[start:end] != b'SKSEPlugin_Version':
            continue
        ordinal, = unpack('<H', offset(ordinal_rva + i * 2, 2))
        if ordinal >= functions:
            raise ValueError('invalid export ordinal')
        data_rva, = unpack('<I', offset(function_rva + ordinal * 4, 4))
        if export <= data_rva < export + export_size:
            raise ValueError('forwarded version export')
        data = offset(data_rva, 0x350)
        if unpack('<I', data)[0] != 1:
            raise ValueError('unknown SKSE metadata version')
        values = unpack('<16I', data + 0x30c)
        versions = []
        for value in values:
            if value == 0:
                break
            versions.append(f'{value >> 24}.{(value >> 16) & 255}.{(value >> 4) & 4095}.{value & 15}')
        return {'explicit_runtimes': versions,
                'version_independence': unpack('<I', data + 0x308)[0],
                'version_independence_ex': unpack('<I', data + 0x304)[0]}
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dll', type=Path)
    parser.add_argument('--runtime', required=True, help='MAJOR.MINOR.PATCH[.BUILD]')
    args = parser.parse_args()
    parts = args.runtime.split('.')
    if len(parts) not in (3, 4) or not all(p.isdecimal() for p in parts):
        parser.error('runtime must have three or four numeric components')
    runtime = '.'.join(str(int(p)) for p in parts + (['0'] if len(parts) == 3 else []))
    result = {'path': str(args.dll.resolve()), 'runtime': runtime,
              'scope': 'Explicit native runtime declaration only; launch verification required.'}
    code = 2
    try:
        blob = args.dll.read_bytes()
        result['sha256'] = hashlib.sha256(blob).hexdigest()
        data = declaration(blob)
        result['declaration'] = data
        if data and data['explicit_runtimes']:
            if runtime in data['explicit_runtimes']:
                code = 0
            elif data['version_independence'] & 3:
                result['note'] = 'Address-library/signature independence may allow this runtime; verify the loader.'
            else:
                code = 1
    except (OSError, ValueError, struct.error) as exc:
        result['error'] = str(exc)
    result['status'] = ['declared-match', 'declared-mismatch', 'undetermined'][code]
    print(json.dumps(result, indent=2))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
