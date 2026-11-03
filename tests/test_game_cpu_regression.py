import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location('gate', Path(__file__).resolve().parents[1] / 'tools/game_cpu_regression.py')
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


class CpuGateTest(unittest.TestCase):
    def test_controls(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            def make(name, cost):
                p = root / name
                p.mkdir()
                def write(file, values):
                    (p / file).write_text(''.join(json.dumps(v) + '\n' for v in values))
                write('stages.jsonl', [{'label': 'idle-start', 'time_ns': 0}, {'label': 'idle-end', 'time_ns': 12e9}])
                write('cpu.jsonl', [{'time_ns': i * 1e9, 'pid': 10, 'ticks_per_second': 100,
                                    'user_ticks': cost * i, 'system_ticks': 0,
                                    'threads': [{'tid': 10, 'user_ticks': cost * i, 'system_ticks': 0}]}
                                   for i in range(13)])
                write('frames.jsonl', [{'request_ns': i * 1e9, 'response_ns': i * 1e9 + 1e6,
                                       'frame': 60 * i} for i in range(13)])
                (p / 'deployment.json').write_text(json.dumps({'modlist_sha256': 'profile', 'files': {'BetterFacialAnimations.ini': 'ini'}}))
                return p
            baseline = [make('old' + str(i), 100 + i) for i in range(3)]
            faster = [make('fast' + str(i), 80 + i) for i in range(3)]
            slower = [make('slow' + str(i), 120 + i) for i in range(3)]
            self.assertEqual(gate.compare(baseline, faster)['status'], 'pass')
            self.assertEqual(gate.compare(baseline, slower)['status'], 'not_demonstrated')
            with self.assertRaises(ValueError): gate.compare(baseline[:1], faster)
            with self.assertRaises(ValueError): gate.compare([baseline[0]] * 3, faster)
            # A common main INI does not imply common FOMOD selections.
            deployment = faster[0] / 'deployment.json'
            original = deployment.read_text()
            data = json.loads(original)
            data['files']['Installer/EmotionModel.ini'] = 'different-choice'
            deployment.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'installer settings differ'):
                gate.compare(baseline, faster)
            deployment.write_text(original)
            (faster[0] / 'stall-probe.json').write_text('{}')
            with self.assertRaises(ValueError): gate.compare(baseline, faster)
            (faster[0] / 'stall-probe.json').unlink()
            (faster[0] / 'stages.jsonl').write_text(json.dumps({'label': 'idle-start', 'time_ns': 0}))
            with self.assertRaises(ValueError): gate.compare(baseline, faster)


if __name__ == '__main__':
    unittest.main()
