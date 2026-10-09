"""Write A2FDirect.bin version 8: Mark, the face Audio2Face-3D v2.3 animates, which the plugin maps onto every head in
game (src/animation/direct.cc): his skin, jaw and tongue. Nothing in it is about Skyrim's heads: each head's mapping is built in
game.

Run: venv/bin/python tools/mark_export.py <out.bin> '<raw output glob>'
  e.g. MARK=models/mark venv/bin/python tools/mark_export.py build/A2FDirect.bin 'work/**/*.a2fraw.npy'
Inputs: $MARK (default models/mark)/{model_data.npz, bs_skin.npz, model_config.json, network.onnx} (the Audio2Face-3D
v2.3 Mark package; needs onnxruntime) and raw network output of real voice lines (frames x >= 272), whose skin deltas
the basis is a PCA of. Lines run with an emotion belong in it, strong ones too: only emotion moves the upper face.

Layout (little-endian; Skyrim axes: x right, y forward, z up; Mark's units):
  'A2FD', u32 version = 8
  u32 nv ; f32 verts[nv*3] (neutral skin) ; f16 jaw[nv] (z of jawOpen: how far each vertex follows the jaw)
  f32 landmarks[8*3] (centroids of the blendshapes in LANDMARKS, weighted by |delta|^4)
  u32 k ; u32 nm ; u32 midx[nm] (ascending: the vertices that move) ; f16 basis[nm*k*3] (vertex-major)
  f32 proj[k*272] ; f32 rest[301] (the network's output for silence; skin coefficient k = (raw[0:272] - rest) . proj[k])
  f32 jaw_points[5*3] (the jaw's points at rest) ; f32 jaw_strength (they move by it times raw[282:297] - rest, which
      is in Mark's own axes; the lower teeth move with them rigidly)
  u32 nt ; f32 tongue_verts[nt*3] (at rest) ; f32 tongue_tip[3] ; f16 tongue_basis[nt*10*3] (vertex-major: motion per
      raw[272:282] - rest, on top of the jaw's)
"""
import glob, json, os, re, struct, sys
import numpy as np

MARK = os.environ.get('MARK', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models/mark'))
K = 48  # Number of principal components in the skin-motion basis.
AX = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]], float)  # Mark (x, y, z) -> Skyrim (-x, z, y)
# Mark blendshapes per landmark (summed; browInnerUp only on the brow's side of the face); the plugin pairs them with
# the head's engine morphs in this order (kLandmarkMorph): Skyrim's BrowUp raises the whole brow
LANDMARKS = [['eyeBlinkLeft'], ['eyeBlinkRight'], ['browOuterUpLeft', 'browInnerUp'], ['browOuterUpRight', 'browInnerUp'],
             ['jawOpen'], ['mouthPucker'], ['mouthSmileLeft'], ['mouthSmileRight']]
SIDE = [0, 0, -1, 1, 0, 0, 0, 0]  # -1: x < the eyes' midline, 1: x > it
MOVING = 0.01  # a vertex is in the basis when its largest component std is above this fraction of the face's


def centroid(v, d):
    w = np.linalg.norm(d, axis=1) ** 4
    return (v * w[:, None]).sum(0) / w.sum()


def read_rest():
    """The network's output for silence, with Mark's implicit emotion (src/inference/live_core.cc kImplicit) and no explicit one."""
    import onnxruntime
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src', 'live_core.cc')).read()
    body = re.search(r'kImplicit\[kNumImplicit\] = \{(.*?)\}', src, re.S).group(1)
    implicit = [float(x.strip().rstrip('f')) for x in body.split(',')]
    assert len(implicit) == 16, len(implicit)
    net = onnxruntime.InferenceSession(MARK + '/network.onnx', providers=['CPUExecutionProvider'])
    emo = np.array(implicit + [0] * 10, np.float32)[None, None]
    return net.run(None, {'input': np.zeros((1, 1, 8320), np.float32), 'emotion': emo})[0].reshape(-1).astype(np.float64)


def mouth(rest, md, cfg):
    """The jaw's points and the tongue at rest, and the tongue's motion per coefficient, with the SDK animators'
    strengths and offsets (height: Mark's y, depth: his z)."""
    jaw = md['neutral_jaw'] + cfg['lower_teeth_strength'] * rest[282:297].reshape(5, 3) + \
        [0, cfg['lower_teeth_height_offset'], cfg['lower_teeth_depth_offset']]
    shapes = cfg['tongue_strength'] * md['shapes_matrix_tongue'].astype(np.float64)  # 10 x nt x 3
    tongue = (md['shapes_mean_tongue'] + np.einsum('k,kvc->vc', rest[272:282], shapes) +
              [0, cfg['tongue_height_offset'], cfg['tongue_depth_offset']]) @ AX.T
    front = tongue[tongue[:, 1] > tongue[:, 1].max() - 0.5]  # the tip: the front end, at the middle
    tip = [tongue[:, 0].mean(), tongue[:, 1].max(), front[:, 2].mean()]
    return jaw @ AX.T, cfg['lower_teeth_strength'], tongue, np.array(tip), (shapes @ AX.T).transpose(1, 0, 2)


def mark(rest, raw_files):
    """Mark's neutral skin, jawOpen z, landmarks and the speech PCA basis (Skyrim axes)."""
    md = np.load(MARK + '/model_data.npz'); bs = np.load(MARK + '/bs_skin.npz')
    cfg = json.load(open(MARK + '/model_config.json'))['config']
    # the SDK skin animator's strengths, folded in: skin strength x the upper / lower face blend of the face mask
    y = md['shapes_mean_skin'][:, 1]; lo, hi = y.min(), y.max()
    fm = 1 / (1 + np.exp(-(cfg['face_mask_level'] - (y - lo) / (hi - lo)) / cfg['face_mask_softness']))
    w = cfg['skin_strength'] * (cfg['upper_face_strength'] * (1 - fm) + cfg['lower_face_strength'] * fm)
    B = ((md['shapes_matrix_skin'].astype(np.float64) * w[None, :, None]) @ AX.T).reshape(272, -1)
    # PCA of B^T (raw - rest) over the frames, through 272 x 272 matrices: with D^T D = Q L Q^T and H = L^1/2 Q^T,
    # the deltas' covariance is (H B)^T (H B), whose top eigenvectors are U = e^-1/2 u^T H B
    D = np.concatenate([np.load(f)[:, :272].astype(np.float64) - rest[:272] for f in raw_files])
    lam, Q = np.linalg.eigh(D.T @ D)
    H = np.sqrt(np.clip(lam, 0, None))[:, None] * Q.T
    e, u = np.linalg.eigh(H @ (B @ B.T) @ H.T)
    o = np.argsort(e)[::-1][:K]
    U = (u[:, o] / np.sqrt(e[o])).T @ H @ B  # K x nv*3, orthonormal rows
    proj = (B @ U.T).T  # K x 272
    basis = U.reshape(K, -1, 3).transpose(1, 0, 2)  # nv x K x 3
    std = np.linalg.norm(basis * np.sqrt(e[o] / len(D))[None, :, None], axis=2).max(1)
    midx = np.where(std > MOVING * std.max())[0]
    verts = bs['neutral'] @ AX.T
    jaw = (bs['jawOpen'] @ AX.T)[:, 2]
    blinks = [centroid(verts, bs[n] @ AX.T) for n in ('eyeBlinkLeft', 'eyeBlinkRight')]
    mid = 0.5 * (blinks[0][0] + blinks[1][0])
    landmarks = np.array([centroid(verts, sum(bs[n] @ AX.T for n in names) * (((verts[:, 0] - mid) * side > 0) if side
                                   else np.ones(len(verts)))[:, None]) for names, side in zip(LANDMARKS, SIDE)])
    info = dict(frames=len(D), k=K, explained=round(float(e[o].sum() / e.sum()), 5), nv=len(verts), moving=len(midx))
    return verts, jaw, landmarks, midx, basis[midx], proj, info, md, cfg


def export(out, raw_glob):
    rest = read_rest()
    raw_files = sorted(glob.glob(raw_glob, recursive=True))
    assert raw_files, 'no raw network output files'
    verts, jaw, landmarks, midx, basis, proj, info, md, cfg = mark(rest, raw_files)
    jaw_points, jaw_strength, tongue, tip, tongue_basis = mouth(rest, md, cfg)
    data = b''.join([b'A2FD', struct.pack('<II', 8, len(verts)), verts.astype('<f4').tobytes(), jaw.astype('<f2').tobytes(),
                     landmarks.astype('<f4').tobytes(), struct.pack('<II', K, len(midx)), midx.astype('<u4').tobytes(),
                     basis.astype('<f2').tobytes(), proj.astype('<f4').tobytes(), rest.astype('<f4').tobytes(),
                     jaw_points.astype('<f4').tobytes(), struct.pack('<fI', jaw_strength, len(tongue)),
                     tongue.astype('<f4').tobytes(), tip.astype('<f4').tobytes(), tongue_basis.astype('<f2').tobytes()])
    open(out, 'wb').write(data)
    info['bytes'] = len(data)
    return info


if __name__ == '__main__':
    print(export(sys.argv[1], sys.argv[2]))
