"""Fold Audio2Face-3D v3.0 into a small network for the plugin.

The diffusion network round-trips through full geometry twice: step 1 embeds
the noise x_T, step 2 embeds x_t1 = a*pad(step1) + b*x_T + c*n1, and the
prediction is pad(step2) de-normalised (skin 20x, tongue 3x, jaw/eyes 10x, plus
a per-identity offset). Everything between the 512-d hidden state in front of
each output layer and the next embedding is affine, so with the noise fixed:

  - step 1's embeddings of x_T become constants,
  - step 2's skin/tongue embeddings become h1 @ (W_out W_emb) + constants,
  - the outputs become the step-2 hidden states (skin_latent, tongue_latent,
    1x30x512 each) and jaw_eyes (1x30x19, de-normalised), with the output
    layers kept aside in heads.npz for the offline projection.

That drops the 72006/16806-wide layers (about 350 MB of weights) and every
geometry-sized activation. Batch is fixed at 1.

  venv/bin/python tools/a2f3_fold.py <out dir> [--seed N] [--check AUDIO]

Writes <out dir>/network_folded.onnx (fp32), network_fp16.onnx and heads.npz:
  skin_w (512, 72006) and tongue_w (512, 16806) with skin/tongue geometry
  delta = latent @ w + const[identity]; skin_const (3, 72006), tongue_const
  (3, 16806); noise_seed.
"""
import argparse
import os
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import a2f3_ref as R  # noqa: E402

SRC = os.path.join(R.MODEL_DIR, "network.onnx")
SKIN, TONGUE = R.SKIN, R.TONGUE
GE = R.SKIN + R.TONGUE  # gum (jaw, 15) + eyes (4) start
STEP1, STEP2 = "/model/geometry_embedding/{0}_embedding/{0}_embedding.0/MatMul", \
    "/model/geometry_embedding/{0}_embedding/{0}_embedding.0_1/MatMul"
FINAL1, FINAL2 = "/model/final_layer{0}/final_layer{0}.2/{1}", "/model/final_layer{0}/final_layer{0}.2_1/{1}"
HEADS = {"face": "", "tongue": "_tongue", "gum": "_gum", "eye": "_eye"}
SCALE = {"skin": 20.0, "tongue": 3.0, "ge": 10.0}


def node(g, name):
    return next(n for n in g.node if n.name == name)


def init_arr(g, name):
    return numpy_helper.to_array(next(t for t in g.initializer if t.name == name))


def weight(g, mm):
    """The initializer input of a MatMul node."""
    names = {t.name for t in g.initializer}
    return init_arr(g, next(i for i in mm.input if i in names))


def bias(g, add):
    names = {t.name for t in g.initializer}
    return init_arr(g, next(i for i in add.input if i in names))


def capture(model_path, tensors, feeds, int64=()):
    """Run the model with extra outputs (float unless named in int64); returns {tensor: value}."""
    import onnxruntime as ort
    m = onnx.load(model_path)
    have = {o.name for o in m.graph.output}
    for t in tensors:
        if t not in have:
            ty = TensorProto.INT64 if t in int64 else TensorProto.FLOAT
            m.graph.output.append(helper.make_tensor_value_info(t, ty, None))
    s = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])
    outs = [o.name for o in s.get_outputs()]
    return dict(zip(outs, s.run(outs, feeds)))


def prune(g):
    """Topologically sort (the new nodes were appended after their consumers), then drop nodes,
    initializers and inputs nothing reachable from the outputs uses."""
    have = {i.name for i in g.input} | {t.name for t in g.initializer} | {""}
    order, pending = [], list(g.node)
    while pending:
        rest = []
        for n in pending:
            if all(i in have for i in n.input):
                order.append(n)
                have.update(n.output)
            else:
                rest.append(n)
        assert len(rest) < len(pending), [n.name for n in rest[:5]]
        pending = rest
    need = {o.name for o in g.output}
    keep = []
    for n in reversed(order):
        if any(o in need for o in n.output):
            keep.append(n)
            need.update(i for i in n.input if i)
    del g.node[:]
    g.node.extend(reversed(keep))
    inits = [t for t in g.initializer if t.name in need]
    del g.initializer[:]
    g.initializer.extend(inits)
    ins = [i for i in g.input if i.name in need]
    del g.input[:]
    g.input.extend(ins)
    del g.value_info[:]


def fold(seed):
    m = onnx.load(SRC)
    g = m.graph
    noise = R.make_noise(np.random.default_rng(seed))[0]  # (3, 60, DIM)
    xT, n1 = noise[2], noise[1]

    # posterior coefficients x_t1 = a*pad(step1) + b*xT + c*n1 (scalar constants of the schedule)
    feeds = {"window": np.zeros((1, R.WIN), np.float32), "identity": np.eye(3, dtype=np.float32)[2:],
             "emotion": np.zeros((1, 30, 10), np.float32), "input_latents": np.zeros((2, 2, 1, 256), np.float32),
             "noise": noise[None]}
    coef = {k: node(g, k).input[0] for k in ("/Mul_2", "/Mul_4", "/Mul_7", "/Mul_10", "/Mul_12")}
    shapes = [n.output[0] for n in g.node if n.op_type == "Shape"]
    cap = capture(SRC, list(coef.values()) + shapes, feeds, set(shapes))
    a, b, c, d, e = (float(np.unique(cap[coef[k]])[0]) for k in ("/Mul_2", "/Mul_4", "/Mul_7", "/Mul_10", "/Mul_12"))
    for k in coef:
        assert np.unique(cap[coef[k]]).size == 1, k
    assert d == 1.0 and e == 0.0, (d, e)  # prediction = pad(step2) only
    xt1_const = b * xT + c * n1  # (60, DIM)

    ident_m = weight(g, node(g, "/MatMul"))  # (3, DIM) per-identity offset added to each step's output
    w_emb = {h: weight(g, node(g, STEP1.format(h))) for h in HEADS}
    w_out = {h: weight(g, node(g, FINAL1.format(HEADS[h], "MatMul"))) for h in HEADS}
    b_out = {h: bias(g, node(g, FINAL1.format(HEADS[h], "Add"))) for h in HEADS}
    cols = {"face": slice(0, SKIN), "tongue": slice(SKIN, GE), "gum": slice(GE, GE + 15), "eye": slice(GE + 15, R.DIM)}
    for h in HEADS:
        assert w_emb[h].shape[0] == cols[h].stop - cols[h].start and w_out[h].shape[1] == w_emb[h].shape[0], h

    new_nodes, new_inits = [], []

    def const(name, arr):
        new_inits.append(numpy_helper.from_array(np.ascontiguousarray(arr, np.float32), name))
        return name

    def replace_uses(old, new):
        for n in g.node:
            for k, i in enumerate(n.input):
                if i == old:
                    n.input[k] = new

    zeros15 = {}

    def pad_t(x, width, tag):
        """[1,30,w] -> zero-pad frames to [1,60,w] -> [60,1,w] (the embeddings' layout)."""
        z = zeros15.setdefault(width, const(f"fold/zeros15_{width}", np.zeros((1, 15, width))))
        new_nodes.append(helper.make_node("Concat", [z, x, z], [f"fold/{tag}_pad"], axis=1, name=f"fold/{tag}_pad"))
        new_nodes.append(helper.make_node("Transpose", [f"fold/{tag}_pad"], [f"fold/{tag}_t"], perm=[1, 0, 2],
                                          name=f"fold/{tag}_t"))
        return f"fold/{tag}_t"

    # step 1: every embedding of x_T is a constant
    for h in HEADS:
        mm = node(g, STEP1.format(h))
        replace_uses(mm.output[0], const(f"fold/{h}_emb1", (xT[:, cols[h]] @ w_emb[h])[:, None, :]))

    # step 2, skin and tongue: emb(x_t1) = a*pad(h1 @ W_out W_emb + (b_out + M[id]) W_emb) + emb(xt1_const)
    for h in ("face", "tongue"):
        we = w_emb[h]
        h1 = node(g, FINAL1.format(HEADS[h], "MatMul")).input[0]
        gmat = const(f"fold/{h}_G", a * (w_out[h] @ we))
        cid = const(f"fold/{h}_C", a * ((ident_m[:, cols[h]] + b_out[h]) @ we))  # (3, k): identity one-hot @ C
        new_nodes += [helper.make_node("MatMul", [h1, gmat], [f"fold/{h}_hg"], name=f"fold/{h}_hg"),
                      helper.make_node("MatMul", ["identity", cid], [f"fold/{h}_ic"], name=f"fold/{h}_ic"),
                      helper.make_node("Add", [f"fold/{h}_hg", f"fold/{h}_ic"], [f"fold/{h}_s"], name=f"fold/{h}_s")]
        t = pad_t(f"fold/{h}_s", we.shape[1], h)
        k2 = const(f"fold/{h}_K2", (xt1_const[:, cols[h]] @ we)[:, None, :])
        new_nodes.append(helper.make_node("Add", [t, k2], [f"fold/{h}_emb2"], name=f"fold/{h}_emb2"))
        replace_uses(node(g, STEP2.format(h)).output[0], f"fold/{h}_emb2")

    # step 2, gum and eyes (19 values): rebuild x_t1 for those columns and keep their small embeddings
    def ge_out(step):
        fl = FINAL1 if step == 1 else FINAL2
        outs = [node(g, fl.format(HEADS[h], "Add")).output[0] for h in ("gum", "eye")]
        name = f"fold/ge{step}"
        new_nodes.append(helper.make_node("Concat", outs, [name], axis=-1, name=name))
        return name

    ge_id = const("fold/ge_M", ident_m[:, GE:])
    new_nodes += [helper.make_node("MatMul", ["identity", ge_id], ["fold/ge_id"], name="fold/ge_id"),
                  helper.make_node("Add", [ge_out(1), "fold/ge_id"], ["fold/ge1_s"], name="fold/ge1_s"),
                  helper.make_node("Mul", ["fold/ge1_s", const("fold/a", np.array(a))], ["fold/ge1_a"], name="fold/ge1_a")]
    t = pad_t("fold/ge1_a", 19, "ge")
    new_nodes.append(helper.make_node("Add", [t, const("fold/ge_K2", xt1_const[:, None, GE:])], ["fold/ge_xt1"],
                                      name="fold/ge_xt1"))
    for h, (s0, s1) in (("gum", (0, 15)), ("eye", (15, 19))):
        mm = node(g, STEP2.format(h))
        out = f"fold/{h}_x"
        new_nodes.append(helper.make_node("Slice", ["fold/ge_xt1", const(f"fold/{h}_s0", np.array([s0], np.int64)),
                                                    const(f"fold/{h}_s1", np.array([s1], np.int64)),
                                                    const(f"fold/{h}_ax", np.array([2], np.int64))], [out], name=out))
        mm.input[0] = out
    # int64 constants must stay int64
    for t in new_inits:
        if t.name.endswith(("_s0", "_s1", "_ax")):
            t.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(t).astype(np.int64), t.name))

    # batch 1 makes every shape static: freeze them, or Shape nodes keep the geometry paths alive
    for t in shapes:
        name = "fold/shape" + t.replace("/", "_")
        new_inits.append(numpy_helper.from_array(cap[t].astype(np.int64), name))
        replace_uses(t, name)

    # outputs: step-2 hidden states and de-normalised jaw/eyes
    h2s = node(g, FINAL2.format("", "MatMul")).input[0]
    h2t = node(g, FINAL2.format("_tongue", "MatMul")).input[0]
    new_nodes += [helper.make_node("Identity", [h2s], ["skin_latent"], name="fold/skin_latent"),
                  helper.make_node("Identity", [h2t], ["tongue_latent"], name="fold/tongue_latent"),
                  helper.make_node("Add", [ge_out(2), "fold/ge_id"], ["fold/ge2_s"], name="fold/ge2_s"),
                  helper.make_node("Mul", ["fold/ge2_s", const("fold/ge_scale", np.array(SCALE["ge"]))], ["jaw_eyes"],
                                   name="fold/jaw_eyes")]
    g.node.extend(new_nodes)
    g.initializer.extend(new_inits)
    lat = next(o for o in g.output if o.name == "output_latents")
    del g.output[:]
    g.output.extend([helper.make_tensor_value_info("skin_latent", TensorProto.FLOAT, [1, 30, 512]),
                     helper.make_tensor_value_info("tongue_latent", TensorProto.FLOAT, [1, 30, 512]),
                     helper.make_tensor_value_info("jaw_eyes", TensorProto.FLOAT, [1, 30, 19]), lat])
    prune(g)
    onnx.checker.check_model(m)

    # heads: geometry delta = latent @ (scale * W_out) + const[identity]; the constant comes from the tail
    # (prediction = scale * (h W_out + b_out + M[id]) + tail offset[id]), measured below
    heads = {"skin_w": SCALE["skin"] * w_out["face"], "tongue_w": SCALE["tongue"] * w_out["tongue"],
             "noise_seed": np.array(seed), "coef": np.array([a, b, c])}
    return m, heads, noise


def tail_consts(heads, noise):
    """const[id] = prediction - latent @ w for each identity (constant over frames)."""
    sk, tg = [], []
    h2s, h2t = "/model/final_layer/final_layer.2_1/MatMul", "/model/final_layer_tongue/final_layer_tongue.2_1/MatMul"
    g = onnx.load(SRC, load_external_data=False).graph
    ts = [node(g, h2s).input[0], node(g, h2t).input[0]]
    rng = np.random.default_rng(1)
    for i in range(3):
        feeds = {"window": rng.standard_normal((1, R.WIN)).astype(np.float32) * 0.1,
                 "identity": np.eye(3, dtype=np.float32)[i:i + 1], "emotion": np.zeros((1, 30, 10), np.float32),
                 "input_latents": np.zeros((2, 2, 1, 256), np.float32), "noise": noise[None]}
        c = capture(SRC, ts, feeds)
        p = c["prediction"][0, 15:45]
        ks = p[:, :SKIN] - c[ts[0]][0] @ heads["skin_w"]
        kt = p[:, SKIN:GE] - c[ts[1]][0] @ heads["tongue_w"]
        assert np.abs(ks - ks[0]).max() < 1e-3 and np.abs(kt - kt[0]).max() < 1e-3
        sk.append(ks.mean(0))
        tg.append(kt.mean(0))
    heads["skin_const"] = np.stack(sk).astype(np.float32)
    heads["tongue_const"] = np.stack(tg).astype(np.float32)


def check(folded, heads, noise, audio):
    """Whole clip, chained latents: folded network + heads vs the original prediction."""
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 8
    orig = ort.InferenceSession(SRC, so, providers=["CPUExecutionProvider"])
    fold_s = ort.InferenceSession(folded, so, providers=["CPUExecutionProvider"])
    pcm = R.load_pcm(audio)
    for identity in (0, 2):
        ref, _ = R.run(orig, pcm, identity, noise=noise[None])
        got = run_folded(fold_s, pcm, identity, heads)
        err = np.abs(got - ref)
        print(f"identity {R.IDENTITIES[identity]}: skin max err {err[:, :SKIN].max():.3g}, tongue "
              f"{err[:, SKIN:GE].max():.3g}, jaw/eyes {err[:, GE:].max():.3g} (|skin| max {np.abs(ref[:, :SKIN]).max():.3f})")


def run_folded(sess, pcm, identity, heads, emotion=None):
    """Full prediction rebuilt from the folded outputs (a2f3_ref.run_latents + heads)."""
    y = R.run_latents(sess, pcm, identity, emotion)
    out = np.zeros((len(y), R.DIM), np.float32)
    out[:, :SKIN] = y[:, :512] @ heads["skin_w"] + heads["skin_const"][identity]
    out[:, SKIN:GE] = y[:, 512:1024] @ heads["tongue_w"] + heads["tongue_const"][identity]
    out[:, GE:] = y[:, 1024:]
    return out


def to_fp16(src, dst):
    from onnxconverter_common import float16
    m = float16.convert_float_to_float16(onnx.load(src), keep_io_types=True)
    outs = {o.name for o in m.graph.output}
    for n in m.graph.node:  # the converter leaves internal Cast(to=float) behind; the graph is fp16 inside
        if n.op_type == "Cast" and n.output[0] not in outs:
            for at in n.attribute:
                if at.name == "to" and at.i == TensorProto.FLOAT:
                    at.i = TensorProto.FLOAT16
    del m.graph.value_info[:]
    onnx.save(m, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    m, heads, noise = fold(a.seed)
    folded = os.path.join(a.out, "network_folded.onnx")
    onnx.save(m, folded)
    tail_consts(heads, noise)
    np.savez(os.path.join(a.out, "heads.npz"), **heads)
    print(f"folded: {os.path.getsize(folded) / 2**20:.0f} MB fp32, coefficients {heads['coef']}")
    if a.check:
        check(folded, heads, noise, a.check)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        to_fp16(folded, os.path.join(a.out, "network_fp16.onnx"))
    print(f"fp16: {os.path.getsize(os.path.join(a.out, 'network_fp16.onnx')) / 2**20:.0f} MB")


if __name__ == "__main__":
    sys.exit(main())
