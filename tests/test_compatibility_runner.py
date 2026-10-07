"""Reader/runner controls: malformed or absent fixtures must never pass."""
import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('compat', ROOT/'tools/compatibility.py')
compat = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compat)


def sub(tag, body):
    return tag + struct.pack('<H', len(body)) + body


def record(kind, body, fid=0, flags=0):
    return struct.pack('<4sIIIIHH', kind, len(body), flags, fid, 0, 44, 0) + body


class ReaderTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)/'SexLab.esm'
        self.rule = {'file': 'SexLab.esm', 'version': 'synthetic', 'records': [
            {'type': 'FACT', 'id': 'E50F', 'editor_id': 'SexLabAnimatingFaction'}]}
        self.body = sub(b'EDID', b'SexLabAnimatingFaction\0')
        self.header = record(b'TES4', sub(b'MAST', b'Skyrim.esm\0'))

    def check(self, data):
        self.path.write_bytes(data)
        return compat.check_asset(self.rule, self.path)

    def test_compressed_nested_record(self):
        body = struct.pack('<I', len(self.body)) + zlib.compress(self.body)
        faction = record(b'FACT', body, 0x0100E50F, 0x40000)
        group = b'GRUP' + struct.pack('<I', 24 + len(faction)) + bytes(16) + faction
        self.assertEqual(self.check(self.header + group)['records'][0]['form_id'], '0100E50F')

    def test_identity_controls(self):
        for kind, fid, flags, body in [
            (b'KYWD', 0x0100E50F, 0, self.body),
            (b'FACT', 0x0000E50F, 0, self.body),  # override, not owned
            (b'FACT', 0x0100E50F, 0x20, self.body),  # deleted
            (b'FACT', 0x0100E50F, 0, sub(b'EDID', b'Other\0')),
        ]:
            with self.subTest(kind=kind, fid=fid, flags=flags):
                with self.assertRaises(ValueError):
                    self.check(self.header + record(kind, body, fid, flags))

    def test_bounds_and_duplicates(self):
        good = record(b'FACT', self.body, 0x0100E50F)
        for data in [self.header + good[:-1], self.header + good + good,
                     self.header + record(b'FACT', b'bad', 0x0100E50F),
                     self.header + record(b'FACT', struct.pack('<I', 100) + zlib.compress(self.body), 0x0100E50F, 0x40000)]:
            with self.assertRaises(ValueError):
                self.check(data)

    def test_extended_subrecord(self):
        body = sub(b'XXXX', struct.pack('<I', len(b'SexLabAnimatingFaction\0')))
        body += b'EDID\0\0SexLabAnimatingFaction\0'
        self.check(self.header + record(b'FACT', body, 0x0100E50F))

    def test_hash_pin(self):
        receipt = self.check(self.header + record(b'FACT', self.body, 0x0100E50F))
        self.assertTrue(compat.check_asset(self.rule, self.path, receipt['sha256'])['pinned'])
        with self.assertRaises(ValueError):
            compat.check_asset(self.rule, self.path, '0' * 64)

    def test_missing_fixture_is_blocked(self):
        out = Path(self.temp.name)/'evidence'
        p = subprocess.run([sys.executable, str(ROOT/'tools/compatibility.py'),
                            '--tier', 'assets', '--case', 'sexlab-ae-166d',
                            '--output', str(out)], capture_output=True, text=True)
        self.assertEqual(p.returncode, 2, p.stderr)
        result = json.loads((out/'result.json').read_text())
        self.assertEqual(result['status'], 'blocked')
        self.assertEqual(result['results'][0]['status'], 'blocked')

    def test_follower_fixture_controls(self):
        # The asset gate must not turn missing or changed geometry into a pass.
        manifest = json.loads((ROOT/'tests/follower_teeth.json').read_text())
        fixture_dir = Path(self.temp.name)/'captures'
        fixture_dir.mkdir()
        fixture_map = Path(self.temp.name)/'fixtures.json'
        fixture_map.write_text(json.dumps({'follower-teeth': {'path': str(fixture_dir)}}))
        for label, expected in [('missing', 2), ('changed', 1)]:
            if label == 'changed':
                (fixture_dir/manifest['fixtures'][0]['file']).write_bytes(b'wrong geometry')
            out = Path(self.temp.name)/label
            process = subprocess.run(
                [sys.executable, str(ROOT/'tools/compatibility.py'),
                 '--tier', 'assets', '--case', 'follower-teeth',
                 '--fixtures', str(fixture_map), '--output', str(out)],
                capture_output=True, text=True)
            self.assertEqual(process.returncode, expected, process.stderr)
            report = json.loads((out/'result.json').read_text())
            self.assertEqual(report['status'], 'blocked' if expected == 2 else 'fail')

    def test_command_timeout(self):
        log = Path(self.temp.name)/'timeout.log'
        self.assertEqual(compat.run_command(
            [sys.executable, '-c', 'import time; time.sleep(30)'], log, 0.05), 124)
        self.assertIn('Timed out', log.read_text())

if __name__ == '__main__':
    unittest.main()
