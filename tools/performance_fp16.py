"""Store the Performance model's large weights as float16; the graph still computes in float32.

Run: <python with onnx> tools/performance_fp16.py <performance.onnx> <out.onnx>

Each float32 initializer with at least MIN_ELEMENTS values is stored as float16 and
read through Cast(to=float), which onnxruntime folds back into a float32 constant
when the session is created. Inputs, outputs, state, small tensors and Constant
nodes keep float32, so only the stored weights are rounded.
"""
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

MIN_ELEMENTS = 1024

m = onnx.load(sys.argv[1])
g = m.graph
if {i.name for i in g.input} & {t.name for t in g.initializer}:
    raise ValueError('Initializers listed as graph inputs are not supported')
casts, converted = [], 0
for t in g.initializer:
    a = numpy_helper.to_array(t)
    if t.data_type != TensorProto.FLOAT or a.size < MIN_ELEMENTS:
        continue
    half = a.astype(np.float16)
    if not np.isfinite(half).all():
        raise ValueError(t.name + ': outside the float16 range')
    name = t.name
    t.CopyFrom(numpy_helper.from_array(half, name + '_fp16'))
    casts.append(helper.make_node('Cast', [name + '_fp16'], [name], name=name + '_cast',
                                  to=TensorProto.FLOAT))
    converted += a.size
if not casts:
    raise ValueError('No float32 weights to convert')
nodes = casts + list(g.node)
del g.node[:]
g.node.extend(nodes)
onnx.checker.check_model(m, full_check=True)
onnx.save(m, sys.argv[2])
print(f'{len(casts)} tensors, {converted} values stored as float16')
