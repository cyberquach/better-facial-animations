"""Write A2FDirect3.bin version 9: the faces the folded Audio2Face-3D v3.0 network (tools/a2f3_fold.py) animates, which
the plugin maps onto every head in game (src/direct.cc): Claire for women and Mark for men, each as mark_export.py
writes Mark for v2.3 (skin, jaw, tongue). Nothing in it is about Skyrim's heads.

Run: venv/bin/python tools/mark_export3.py <out.bin> ['<latent glob with {identity}>']
  e.g. venv/bin/python tools/mark_export3.py dist/A2FDirect3.bin
Inputs, under $A2F (default: the repository): sdk/_data/audio2face-models/audio2face-3d-v3.0/{model_data,bs_skin,
model_config}_<Identity>, models/v3/{heads.npz, rest_<Identity>.npy} (a2f3_fold.py; a2f3_ref.rest_latent) and the folded
network's output for real voice lines (tools/a2f3_bake.py: frames x 1043, default glob
work/RiverwoodLocation/**/*.a2f3_{identity}.npy), whose skin and tongue deltas the bases are a PCA of. Lines run with
an emotion belong in it: only emotion moves the upper face.

Layout (little-endian; Skyrim axes: x right, y forward, z up; the SDK's units):
  'A2FD', u32 version = 9, u32 faces = 2 (female, male), u32 n_skin = 512, u32 n_tongue = 512, u32 n_out = 1043
  per face:
    u32 nv ; f32 verts[nv*3] (the skin at silence) ; f16 jaw[nv] (z of jawOpen: how far each vertex follows the jaw)
    f32 landmarks[8*3] (centroids of the blendshapes in mark_export.LANDMARKS, weighted by |delta|^4)
    u32 k ; u32 nm ; u32 midx[nm] (ascending: the vertices that move) ; f16 basis[nm*k*3] (vertex-major)
    f32 proj[k*n_skin] ; f32 rest[n_out] (the network's output for silence; skin coefficient
        k = (raw[0:n_skin] - rest) . proj[k])
    f32 jaw_points[5*3] (the jaw's points at rest) ; f32 jaw_strength (they move by it times
        raw[n_skin+n_tongue : +15] - rest, in the SDK's axes; the lower teeth move with them rigidly)
    u32 nt ; f32 tongue_verts[nt*3] (at rest) ; f32 tongue_tip[3] ; f16 tongue_basis[nt*10*3] (vertex-major: motion per
        tongue coefficient, on top of the jaw's)
    f32 tongue_proj[10*n_tongue] (tongue coefficient j = (raw[n_skin : n_skin+n_tongue] - rest) . tongue_proj[j])
"""
import glob, json, os, struct, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mark_export import AX, K, LANDMARKS, MOVING, SIDE, centroid  # noqa: E402

A2F = os.environ.get('A2F', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SDK = os.path.join(A2F, 'sdk/_data/audio2face-models/audio2face-3d-v3.0')
FACES = (('Claire', 0), ('Mark', 2))  # female, male: the network's identity index
N_SKIN = N_TONGUE = 512
N_OUT = N_SKIN + N_TONGUE + 19  # then the jaw's 5 points and the eyes (not used)
KT = 10  # tongue coefficients (a2f::kNumTongue)


def pca(D, B, k):
    """Principal directions of the deltas D @ B (frames x n) through the latents' Gram matrix: U (k x n, orthonormal
    rows), proj (k x latent) with coefficient = d . proj, the variance per direction and the share explained."""
    lam, Q = np.linalg.eigh(D.T @ D)
    H = np.sqrt(np.clip(lam, 0, None))[:, None] * Q.T
    e, u = np.linalg.eigh(H @ (B @ B.T) @ H.T)
    o = np.argsort(e)[::-1][:k]
    U = (u[:, o] / np.sqrt(e[o])).T @ H @ B
    return U, (B @ U.T).T, e[o] / len(D), float(e[o].sum() / e.sum())


def face(name, identity, files):
    md = np.load(f'{SDK}/model_data_{name}.npz'); bs = np.load(f'{SDK}/bs_skin_{name}.npz')
    cfg = json.load(open(f'{SDK}/model_config_{name}.json'))['config']
    heads = np.load(os.path.join(A2F, 'models/v3/heads.npz'))
    rest = np.load(os.path.join(A2F, f'models/v3/rest_{name}.npy')).astype(np.float64)
    assert rest.shape == (N_OUT,), rest.shape
    lat = np.concatenate([np.load(f).astype(np.float64) for f in files])
    # the SDK skin animator's strengths, folded in: skin strength x the upper / lower face blend of the face mask
    y = md['neutral_skin'][:, 1]; lo, hi = y.min(), y.max()
    fm = 1 / (1 + np.exp(-(cfg['face_mask_level'] - (y - lo) / (hi - lo)) / cfg['face_mask_softness']))
    w = cfg['skin_strength'] * (cfg['upper_face_strength'] * (1 - fm) + cfg['lower_face_strength'] * fm)
    W = heads['skin_w'].reshape(N_SKIN, -1, 3).astype(np.float64) * w[None, :, None]
    B = (W @ AX.T).reshape(N_SKIN, -1)
    U, proj, var, explained = pca(lat[:, :N_SKIN] - rest[:N_SKIN], B, K)
    basis = U.reshape(K, -1, 3).transpose(1, 0, 2)  # nv x K x 3
    std = np.linalg.norm(basis * np.sqrt(var)[None, :, None], axis=2).max(1)
    midx = np.where(std > MOVING * std.max())[0]
    # the skin at silence: what a head at rest is matched with
    silent = (rest[:N_SKIN] @ heads['skin_w'] + heads['skin_const'][identity]).reshape(-1, 3) * w[:, None]
    verts = (md['neutral_skin'] + silent) @ AX.T
    jaw = (bs['jawOpen'] @ AX.T)[:, 2]
    blinks = [centroid(verts, bs[n] @ AX.T) for n in ('eyeBlinkLeft', 'eyeBlinkRight')]
    mid = 0.5 * (blinks[0][0] + blinks[1][0])
    landmarks = np.array([centroid(verts, sum(bs[n] @ AX.T for n in names) * (((verts[:, 0] - mid) * side > 0) if side
                                   else np.ones(len(verts)))[:, None]) for names, side in zip(LANDMARKS, SIDE)])
    # the jaw's points and the tongue at rest, with the SDK animators' strengths and offsets (height: y, depth: z)
    jr = rest[N_SKIN + N_TONGUE:N_SKIN + N_TONGUE + 15].reshape(5, 3)
    jaw_points = (md['neutral_jaw'] + cfg['lower_teeth_strength'] * jr +
                  [0, cfg['lower_teeth_height_offset'], cfg['lower_teeth_depth_offset']]) @ AX.T
    TW = cfg['tongue_strength'] * heads['tongue_w'].reshape(N_TONGUE, -1, 3).astype(np.float64)
    tr = rest[N_SKIN:N_SKIN + N_TONGUE]
    TU, tproj, _, texplained = pca(lat[:, N_SKIN:N_SKIN + N_TONGUE] - tr, (TW @ AX.T).reshape(N_TONGUE, -1), KT)
    tsilent = cfg['tongue_strength'] * (tr @ heads['tongue_w'] + heads['tongue_const'][identity]).reshape(-1, 3)
    tongue = (md['neutral_tongue'] + tsilent + [0, cfg['tongue_height_offset'], cfg['tongue_depth_offset']]) @ AX.T
    front = tongue[tongue[:, 1] > tongue[:, 1].max() - 0.5]  # the tip: the front end, at the middle
    tip = np.array([tongue[:, 0].mean(), tongue[:, 1].max(), front[:, 2].mean()])
    tongue_basis = TU.reshape(KT, -1, 3).transpose(1, 0, 2)  # nt x KT x 3
    data = b''.join([struct.pack('<I', len(verts)), verts.astype('<f4').tobytes(), jaw.astype('<f2').tobytes(),
                     landmarks.astype('<f4').tobytes(), struct.pack('<II', K, len(midx)), midx.astype('<u4').tobytes(),
                     basis[midx].astype('<f2').tobytes(), proj.astype('<f4').tobytes(), rest.astype('<f4').tobytes(),
                     jaw_points.astype('<f4').tobytes(), struct.pack('<fI', cfg['lower_teeth_strength'], len(tongue)),
                     tongue.astype('<f4').tobytes(), tip.astype('<f4').tobytes(), tongue_basis.astype('<f2').tobytes(),
                     tproj.astype('<f4').tobytes()])
    info = dict(lines=len(files), frames=len(lat), k=K, explained=round(explained, 5),
                tongue_explained=round(texplained, 5), nv=len(verts), moving=len(midx), bytes=len(data))
    return data, info


def export(out, pattern):
    data, info = [b'A2FD', struct.pack('<IIIII', 9, len(FACES), N_SKIN, N_TONGUE, N_OUT)], {}
    for name, identity in FACES:
        files = sorted(glob.glob(pattern.format(identity=name), recursive=True))
        assert files, 'no latents for ' + name
        d, info[name] = face(name, identity, files)
        data.append(d)
    data = b''.join(data)
    open(out, 'wb').write(data)
    info['bytes'] = len(data)
    return info


if __name__ == '__main__':
    print(export(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else
                 os.path.join(A2F, 'work/RiverwoodLocation/**/*.a2f3_{identity}.npy')))
