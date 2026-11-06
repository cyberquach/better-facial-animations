"""Minimal Skyrim SE nif reader: per-vertex UVs (and positions) of BSTriShape / BSDynamicTriShape blocks."""
import struct, numpy as np

def _header(d):
    o = d.index(b'\n') + 1
    ver, endian, user, nblocks = struct.unpack_from('<IBII', d, o); o += 13
    bsver, = struct.unpack_from('<I', d, o); o += 4
    for _ in range(3): o += 1 + d[o]  # author, process script, export script
    if bsver >= 103: o += 1 + d[o]
    nt, = struct.unpack_from('<H', d, o); o += 2; types = []
    for _ in range(nt): n, = struct.unpack_from('<I', d, o); types.append(d[o + 4:o + 4 + n].decode()); o += 4 + n
    tidx = struct.unpack_from(f'<{nblocks}H', d, o); o += 2 * nblocks
    sizes = struct.unpack_from(f'<{nblocks}I', d, o); o += 4 * nblocks
    ns, _ = struct.unpack_from('<II', d, o); o += 8
    for _ in range(ns): n, = struct.unpack_from('<I', d, o); o += 4 + n
    ng, = struct.unpack_from('<I', d, o); o += 4 + 4 * ng
    return bsver, [types[i] for i in tidx], sizes, o

def shapes(d):
    """-> list of dict(type, verts (N,3), uvs (N,2), tris (M,3))"""
    bsver, types, sizes, o = _header(d); out = []
    for t, sz in zip(types, sizes):
        if t in ('BSTriShape', 'BSDynamicTriShape', 'BSSubIndexTriShape'):
            p = o + 4  # name
            nex, = struct.unpack_from('<I', d, p); p += 4 + 4 * nex + 4  # extra data refs, controller
            p += 4 + 12 + 36 + 4 + 4  # flags, translation, rotation, scale, collision
            p += 16 + 4 + 4 + 4  # bounding sphere, skin, shader, alpha
            desc, = struct.unpack_from('<Q', d, p); p += 8
            ntri, nv = struct.unpack_from('<HH', d, p); p += 4
            dsize, = struct.unpack_from('<I', d, p); p += 4
            vsz = (desc & 0xF) * 4; uvo = ((desc >> 8) & 0xF) * 4; flags = desc >> 44
            full = bool(flags & (1 << 10)) or bsver < 100
            V, UV = np.zeros((nv, 3)), np.zeros((nv, 2))
            if dsize:
                for i in range(nv):
                    b = p + i * vsz
                    V[i] = struct.unpack_from('<3f', d, b) if full else np.frombuffer(d[b:b + 6], '<f2')
                    if flags & 2: UV[i] = np.frombuffer(d[b + uvo:b + uvo + 4], '<f2')
                p += nv * vsz
            T = np.frombuffer(d[p:p + ntri * 6], '<u2').reshape(-1, 3) if dsize else np.zeros((0, 3), int)
            if dsize: p += ntri * 6
            if bsver == 100:
                pds, = struct.unpack_from('<I', d, p); p += 4; assert pds == 0
            if t == 'BSDynamicTriShape':  # positions live here as float4
                dds, = struct.unpack_from('<I', d, p); p += 4
                V = np.frombuffer(d[p:p + dds], '<f4').reshape(-1, 4)[:, :3].astype(float)
            out.append(dict(type=t, verts=V, uvs=UV, tris=T))
        o += sz
    return out

def skin_partition_vertices(d):
    """SSE skinned meshes keep vertex data in NiSkinPartition. -> list of dict(verts, uvs) per partition block."""
    bsver, types, sizes, o = _header(d); out = []
    for t, sz in zip(types, sizes):
        if t == 'NiSkinPartition':
            p = o + 4  # num partitions
            dsize, vsz, desc = struct.unpack_from('<IIQ', d, p); p += 16
            uvo = ((desc >> 8) & 0xF) * 4; flags = desc >> 44; nv = dsize // vsz
            UV = np.array([np.frombuffer(d[p + i * vsz + uvo:p + i * vsz + uvo + 4], '<f2') for i in range(nv)], float)
            out.append(dict(uvs=UV, has_pos=bool(flags & 1)))  # dynamic shapes carry no positions here
        o += sz
    return out
