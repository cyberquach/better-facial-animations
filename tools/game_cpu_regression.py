#!/usr/bin/env python3
"""Compare repeated game CPU captures. Sampling is not presented-frame timing.

Each capture contains cpu.jsonl (/proc ticks), frames.jsonl (engine counter),
stages.jsonl (named -start/-end windows), and deployment.json (artifact hashes).
The deployment files map must include every installed Installer/*.ini fragment
as well as BetterFacialAnimations.ini; omit fragments only for manual installs.
Use --baseline DIR ... --candidate DIR ...; fewer than three runs is blocked.
"""
import argparse
import json
import random
import statistics
from pathlib import Path


def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def capture(path):
    if (path / 'stall-probe.json').exists():
        raise ValueError('debugger-interrupted capture cannot establish CPU parity')
    cpu = rows(path / 'cpu.jsonl')
    frames = [r for r in rows(path / 'frames.jsonl')
              if 'response_ns' in r and r['response_ns'] - r['request_ns'] <= 8e6]
    stages = rows(path / 'stages.jsonl')
    result = {}
    ticks = lambda r: r['user_ticks'] + r['system_ticks']
    for start in stages:
        if not start['label'].endswith('-start'):
            continue
        name = start['label'][:-6]
        ends = [r for r in stages if r['label'] == name + '-end']
        if len(ends) != 1 or name in result:
            raise ValueError('incomplete or duplicate window: ' + name)
        samples = [r for r in cpu if start['time_ns'] <= r['time_ns'] <= ends[0]['time_ns']]
        if len(samples) < 3:
            raise ValueError('missing CPU samples: ' + name)
        a, b = samples[0], samples[-1]
        if b['time_ns'] - a['time_ns'] < 10e9 or a['pid'] != b['pid']:
            raise ValueError('short window or process changed: ' + name)
        fs = [r for r in frames if a['time_ns'] <= r['response_ns'] <= b['time_ns']]
        if len(fs) < 2 or fs[-1]['frame'] <= fs[0]['frame']:
            raise ValueError('no engine-frame progress: ' + name)
        count = fs[-1]['frame'] - fs[0]['frame']
        scale = 1000 / a['ticks_per_second'] / count
        main = lambda r: next(t for t in r['threads'] if t['tid'] == r['pid'])
        result[name] = {'process_cpu_ms_per_frame': (ticks(b) - ticks(a)) * scale,
                        'main_cpu_ms_per_frame': (ticks(main(b)) - ticks(main(a))) * scale}
    if not result:
        raise ValueError('no measured windows')
    deploy = json.loads((path / 'deployment.json').read_text())
    files = deploy['files']
    installer = tuple(sorted(
        (name.replace('\\', '/').lower(), digest) for name, digest in files.items()
        if '/installer/' in '/' + name.replace('\\', '/').lower()))
    return result, (deploy['modlist_sha256'], files['BetterFacialAnimations.ini'], installer)


def compare(baseline, candidate):
    if min(len(baseline), len(candidate)) < 3:
        raise ValueError('at least three complete launches per build are required')
    if len({p.resolve() for p in baseline + candidate}) != len(baseline + candidate):
        raise ValueError('duplicate capture directories are not independent launches')
    runs = [capture(p) for p in baseline + candidate]
    if len({r[1] for r in runs}) != 1:
        raise ValueError('profile modlist, initial INI or installer settings differ')
    windows = set(runs[0][0])
    if any(set(r[0]) != windows for r in runs):
        raise ValueError('measured windows differ')
    rng = random.Random(422)
    results = []
    for window in sorted(windows):
        for metric in ('process_cpu_ms_per_frame', 'main_cpu_ms_per_frame'):
            a = [r[0][window][metric] for r in runs[:len(baseline)]]
            b = [r[0][window][metric] for r in runs[len(baseline):]]
            # Resample whole launches, not frames: adjacent frames are dependent.
            deltas = sorted(statistics.mean(rng.choices(b, k=len(b))) -
                            statistics.mean(rng.choices(a, k=len(a))) for _ in range(5000))
            upper = deltas[4749]
            results.append(dict(window=window, metric=metric,
                                baseline=a, candidate=b,
                                delta_mean=statistics.mean(b)-statistics.mean(a),
                                delta_upper_one_sided_95=upper, passed=upper <= 0))
    return {'status': 'pass' if all(r['passed'] for r in results) else 'not_demonstrated',
            'scope': 'CPU per observed engine frame only; inspect frame stalls and visual/API correctness separately. '
                     'Three launches are a minimum, not a guarantee of adequate statistical power. '
                     'Bootstrap bounds do not eliminate systematic scene or system-load differences.',
            'metrics': results}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', nargs='+', type=Path, required=True)
    p.add_argument('--candidate', nargs='+', type=Path, required=True)
    args = p.parse_args()
    try:
        result = compare(args.baseline, args.candidate)
    except (OSError, ValueError, KeyError, StopIteration) as e:
        result = {'status': 'blocked', 'reason': str(e)}
    print(json.dumps(result, indent=2))
    return {'pass': 0, 'not_demonstrated': 1, 'blocked': 2}[result['status']]


if __name__ == '__main__':
    raise SystemExit(main())
