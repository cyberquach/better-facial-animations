"""Export a trusted student checkpoint for BFA's [1,48,188] emotion frontend.

CPU only. Requires torch, numpy, onnx and onnxruntime. Run with --checkpoint,
--sha256 and --output; keep the resulting JSON alongside the ONNX model.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import onnx
import onnxruntime as ort
import torch
from emotion_student import Frontend, make_student


def export(checkpoint, digest, output):
    payload = checkpoint.read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if actual != digest:
        raise ValueError('Checkpoint SHA256 mismatch')
    torch.set_num_threads(2)
    # Only a locally produced, hash-verified checkpoint is accepted.
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model = make_student(state['width'], state['architecture']).eval()
    model.load_state_dict(state['model'], strict=True)
    if not all(torch.isfinite(t).all() for t in model.state_dict().values()):
        raise ValueError('Nonfinite checkpoint')
    rng = np.random.default_rng(20261102)
    t = np.arange(60000, dtype=np.float32)
    pcm = np.stack([np.zeros(60000, np.float32), np.full(60000,.125,np.float32),
                    np.sin(t*.0864)*.00001, np.sin(t*.0864)*.2,
                    rng.normal(0,.1,60000).astype(np.float32),
                    (.1*np.sin(t*.043)+.04*np.sin(t*.091))*(.5+.5*np.sin(t*.0009))])
    with torch.no_grad():
        mel = Frontend()(torch.from_numpy(pcm)).numpy()
        expected = model(torch.from_numpy(mel)).numpy()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, torch.from_numpy(mel[:1]), str(output),
                      input_names=['mel'], output_names=['logits'],
                      opset_version=17, dynamo=False)
    onnx.checker.check_model(onnx.load(output))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(output), sess_options=options,
                                   providers=['CPUExecutionProvider'])
    predicted = np.concatenate([session.run(None, {'mel':x[None]})[0] for x in mel])
    error = float(np.max(np.abs(predicted-expected)))
    if not np.isfinite(predicted).all() or error > 1e-3:
        raise ValueError(f'Export error: {error}')
    timings=[]
    for i in range(35):
        began=time.perf_counter()
        with torch.no_grad(): x=Frontend()(torch.from_numpy(pcm[i%len(pcm)][None])).numpy()
        session.run(None,{'mel':x})
        if i>=5:timings.append((time.perf_counter()-began)*1000)
    result={'checkpoint_sha256':actual,'optimizer_steps':state['steps'],
            'width':state['width'],'architecture':state['architecture'],
            'parameters':sum(p.numel() for p in model.parameters()),
            'onnx_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
            'onnx_bytes':output.stat().st_size,'max_export_logit_error':error,
            'cpu_threads':2,'cpu_frontend_model_median_ms':float(np.median(timings)),
            'input':'float32 [1,48,188], emotion_student.Frontend',
            'output':'float32 [1,6] logits: anger/disgust/fear/joy/neutral/sadness',
            'status':'experimental checkpoint; closed-corpus memorization, not held-out quality',
            'limitations':'Trained to memorize Full model answers on the training dialogue. Unfamiliar lines, especially obscure quest mods, may produce hit-or-miss results. Does not contain a lookup table or guarantee perfect reproduction.',
            'license':'NVIDIA Audio2Emotion agreement; Audio2Face project use only',
            'in_game_tested':False}
    output.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
    # Reference fixtures for the native C++ frontend and complete ONNX path.
    with output.with_suffix('.fixtures').open('wb') as f:
        np.array([len(pcm)],dtype='<u4').tofile(f)
        for audio,features,logits in zip(pcm,mel,expected):
            for a in (audio,features,logits):a.astype('<f4').tofile(f)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    export(args.checkpoint,args.sha256,args.output)
