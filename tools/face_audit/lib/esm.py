"""Minimal TES5/SSE plugin walker."""
import struct, zlib

def subrecords(data):
    i, big = 0, None
    while i < len(data):
        t, n = struct.unpack_from('<4sH', data, i); i += 6
        if t == b'XXXX':
            big = struct.unpack_from('<I', data, i)[0]; i += n; continue
        if big is not None: n, big = big, None
        yield t.decode('ascii', 'replace'), data[i:i + n]; i += n

def walk(path, want, ctx=None):
    """Yield (type, formid, subrecords list, ctx) for record types in `want`.
    ctx tracks enclosing WRLD/CELL formids."""
    buf = open(path, 'rb').read()
    def rec(i, end, ctx):
        while i < end:
            t, size = struct.unpack_from('<4sI', buf, i)
            if t == b'GRUP':
                label, gtype = struct.unpack_from('<Ii', buf, i + 8)
                c = dict(ctx)
                if gtype == 1: c['wrld'] = label
                if gtype in (6, 8, 9, 10): c['cell'] = label
                yield from rec(i + 24, i + size, c)
                i += size; continue
            flags, fid = struct.unpack_from('<II', buf, i + 8)
            ty = t.decode()
            if ty in want:
                data = buf[i + 24:i + 24 + size]
                if flags & 0x40000: data = zlib.decompress(data[4:])
                yield ty, fid, list(subrecords(data)), ctx
            i += 24 + size
    yield from rec(0, len(buf), ctx or {})

def cstr(b): return b.split(b'\0')[0].decode('cp1252', 'replace')
def first(subs, t):
    for k, v in subs:
        if k == t: return v
