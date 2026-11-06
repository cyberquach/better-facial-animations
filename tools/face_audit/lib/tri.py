"""FaceGen TRI (FRTRI003) reader: base verts + named differential morphs."""
import struct, numpy as np

def read(path):
    d = open(path, 'rb').read()
    assert d[:8] == b'FRTRI003', d[:8]
    nv, nt, nq, _, _, nuv, flags, nmorph, nmod, nmodv = struct.unpack_from('<10i', d, 8)
    p = 64
    verts = np.frombuffer(d, '<f4', (nv + nmodv) * 3, p).reshape(-1, 3); p += (nv + nmodv) * 12
    tris = np.frombuffer(d, '<i4', nt * 3, p).reshape(-1, 3); p += nt * 12
    p += nq * 16
    if nuv:
        p += nuv * 8 + nt * 12 + nq * 16
    morphs = {}
    for _ in range(nmorph):
        n = struct.unpack_from('<i', d, p)[0]; name = d[p + 4:p + 4 + n].rstrip(b'\0').decode(); p += 4 + n
        scale = struct.unpack_from('<f', d, p)[0]; p += 4
        delta = np.frombuffer(d, '<i2', nv * 3, p).reshape(-1, 3).astype(np.float32) * scale; p += nv * 6
        morphs[name] = delta
    return dict(verts=verts[:nv].copy(), tris=tris.copy(), morphs=morphs)
