#!/usr/bin/env python3
"""Capture installed FaceGen heads for the beard-clearance gate. Local, read-only.

    BFA_AUDIT_OUTPUT=<dir with records.json from assets.py> BFA_AUDIT_GAME=<Skyrim> \\
    [BFA_AUDIT_PROFILE=<MO2 profile name or directory>] \\
    python3 tools/face_audit/beard_capture.py <output dir> <plugin:formid>...

Writes <editor id>.bin per NPC: the winning FaceGen NIF's dynamic shapes with
their NiSkinPartition triangles and the expression TRI of the head part each
shape names (the NPC's own parts and their extras first). A shape is marked as
measured when its head part is facial hair, a scar or of no listed type and it
is not a brow or lash. Captures hold third-party geometry: keep them out of Git.
"""
import hashlib, json, re, struct, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE / 'lib')]
import esm, nif_uv, tri as trilib
from assets import Assets, named_shapes, records

HEAD, MOUTH, EYES, PINNED, NONE = range(5)  # direct::Part
MORPHS = re.findall(r'\{\d+, \d+, "(\w+)"\}', (HERE.parents[1] / 'src/animation/direct.cc').read_text())


def shape_triangles(d):
    """Per tri-shape block: the triangles of its NiSkinPartition."""
    bsver, types, sizes, o0 = nif_uv._header(d)
    offs = np.concatenate([[o0], o0 + np.cumsum(sizes)[:-1]]).astype(int)
    out = []
    for bi, t in enumerate(types):
        if t not in ('BSTriShape', 'BSDynamicTriShape', 'BSSubIndexTriShape'): continue
        p = offs[bi] + 4
        nex, = struct.unpack_from('<I', d, p); p += 4 + 4 * nex + 4
        p += 4 + 12 + 36 + 4 + 4 + 16
        skin, = struct.unpack_from('<i', d, p)
        tris = np.zeros((0, 3), np.uint16)
        if 0 <= skin < len(types):
            part, = struct.unpack_from('<i', d, offs[skin] + 4)
            if 0 <= part < len(types) and types[part] == 'NiSkinPartition':
                q = offs[part]; end = q + sizes[part]
                npart, = struct.unpack_from('<I', d, q); q += 4
                dsize, vsz, desc = struct.unpack_from('<IIQ', d, q); q += 16 + dsize
                acc = []
                for _ in range(npart):
                    nv, nt, nb, nstr, nw = struct.unpack_from('<5H', d, q); q += 10 + 2 * nb
                    if d[q]: q += 2 * nv
                    q += 1
                    if d[q]: q += 4 * nv * nw
                    q += 1 + 2 * nstr
                    has_faces = d[q]; q += 1
                    assert nstr == 0
                    if has_faces: q += 6 * nt
                    if d[q]: q += nv * nw
                    q += 1 + 2 + 8
                    acc.append(np.frombuffer(d, '<u2', nt * 3, q).reshape(-1, 3)); q += 6 * nt
                assert q == end, (q, end)
                tris = np.concatenate(acc) if acc else tris
        out.append(tris)
    return out


def classify(name, typ):
    """direct::Part and whether the shape is the facial skin, from the head part type."""
    n = name.lower()
    if typ == 3: return NONE, False
    skin = (typ == 1) if typ is not None else ('head' in n and 'brow' not in n and 'beard' not in n)
    if typ in (1, 4, 5, 6): return HEAD, skin
    if typ == 2: return EYES, skin
    if typ is not None and typ > 6: return PINNED, skin
    if 'mouth' in n: return MOUTH, False
    if 'eyes' in n: return EYES, skin
    if any(w in n for w in ('head', 'brow', 'beard', 'mustache', 'moustache', 'goatee', 'stubble', 'sideburn', 'scar', 'lash')): return HEAD, skin
    return (PINNED if typ is not None else NONE), skin


class Capture:
    def __init__(self):
        self.a = Assets(cached=True); self.extras = {}; self.walked = set()

    def _extras(self, plugin):
        if plugin in self.walked: return
        self.walked.add(plugin); masters = []
        for ty, fid, flags, subs in records(self.a.files[plugin], (b'TES4', b'HDPT')):
            if ty == 'TES4': masters = [esm.cstr(v).lower() for t, v in subs if t == 'MAST']; continue
            form = lambda f: (masters + [plugin])[min(f >> 24, len(masters))] + ':%06x' % (f & 0xffffff)
            self.extras[form(fid)] = [form(struct.unpack('<I', v)[0]) for t, v in subs if t == 'HNAM' and len(v) == 4]

    def parts(self, npc):
        seen, todo = {}, list(npc['parts'])
        while todo:
            i = todo.pop()
            if i in seen or i not in self.a.parts: continue
            seen[i] = self.a.parts[i]; self._extras(i.split(':')[0])
            todo += self.extras.get(i, [])
        return {p['edid'].lower(): p for p in seen.values()}

    def write(self, npc_id, directory):
        a = self.a; npc = a.npcs[npc_id]; plugin, fid = npc_id.split(':')
        d, src = a.read('meshes\\actors\\character\\facegendata\\facegeom\\%s\\%08x.nif' % (plugin, int(fid, 16)))
        own = self.parts(npc); out = []; sources = {}
        for s, t in zip(named_shapes(d), shape_triangles(d)):
            if s['type'] != 'BSDynamicTriShape': continue
            n = s['name'].lower()
            part = own.get(n) or (a.part_names.get(n) or [None])[-1]
            typ = part['type'] if part else None
            cls, skin = classify(s['name'], typ)
            if cls == NONE: continue
            verts = s['verts'].astype('<f4'); morphs = None
            if part and part['tri']:
                data, where = a.read(part['tri'])
                tmp = directory / '.capture.tri'; tmp.write_bytes(data)
                try:
                    tr = trilib.read(str(tmp))
                    if len(tr['verts']) == len(verts):
                        morphs = [np.asarray(tr['morphs'].get(m, np.zeros_like(tr['verts'])), '<f4') for m in MORPHS]
                        sources[part['tri']] = hashlib.sha256(data).hexdigest()
                finally:
                    tmp.unlink()
            measured = (cls == HEAD and not skin and typ in (4, 0, None, 5) and 'lash' not in n and 'brow' not in n
                        and len(t) and np.ptp(verts, axis=0).max() > .5)
            name = s['name'].encode()
            blob = struct.pack('<I', len(name)) + name + struct.pack('<6I', len(verts), cls, int(skin), 0, 0, int(measured))
            blob += verts.tobytes() + struct.pack('<I', len(MORPHS) if morphs is not None else 0)
            for m, values in zip(MORPHS, morphs or []):
                blob += struct.pack('<I', len(m)) + m.encode() + values.tobytes()
            blob += struct.pack('<I', len(t)) + t.astype('<u4').tobytes()
            out.append(blob)
        path = directory / (re.sub(r'[^A-Za-z0-9_.-]', '_', npc['edid']) + '.bin')
        path.write_bytes(b'BFABEARD' + struct.pack('<3I', 1, int(npc['female']), len(out)) + b''.join(out))
        return dict(file=path.name, npc=npc_id, editor_id=npc['edid'], female=int(npc['female']),
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    facegen_sha256=hashlib.sha256(d).hexdigest(), facegen=str(src.get('loose') or src.get('archive')),
                    tri_sha256=sources)


if __name__ == '__main__':
    if len(sys.argv) < 3: sys.exit(__doc__)
    directory = Path(sys.argv[1]); directory.mkdir(parents=True, exist_ok=True)
    capture = Capture()
    print(json.dumps([capture.write(n, directory) for n in sys.argv[2:]], indent=2))
