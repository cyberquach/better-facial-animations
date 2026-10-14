"""Audio2Emotion-v2.2 -> FP16 weights with FP32 normalization and I/O.

Run: venv/bin/python tools/a2e_fp16.py <network.onnx> <out.onnx>   (needs onnx, onnxconverter-common)
"""
import sys

import onnx
from onnx import TensorProto
from onnxconverter_common import float16

m = onnx.load(sys.argv[1])
# Half-precision normalization produces nonfinite logits on DirectML, even
# for the silence warmup; retain the original preprocessing arithmetic.
normalization = [n.name for n in m.graph.node
                 if not n.name.startswith('/model/') or 'layer_norm' in n.name]
if not normalization or '/Div_1' not in normalization:
    raise ValueError('Expected Audio2Emotion-v2.2 input normalization graph')
casts = [n.name for n in m.graph.node if n.op_type == 'Cast'
         and n.attribute[0].i == TensorProto.FLOAT and n.name not in normalization]
m = float16.convert_float_to_float16(m, keep_io_types=True, node_block_list=normalization)
for n in m.graph.node:  # the graph's own casts to float: the converter retypes their outputs but not the casts
    if n.name in casts:
        n.attribute[0].i = TensorProto.FLOAT16
onnx.checker.check_model(m)
onnx.save(m, sys.argv[2])
