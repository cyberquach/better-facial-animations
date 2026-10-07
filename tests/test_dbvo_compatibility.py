"""Offline controls for the pinned DBVO 2 fixture identity checker."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('dbvo', Path(__file__).resolve().parents[1] / 'tools/dbvo_compatibility.py')
dbvo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dbvo)


def fixture():
    data = bytearray(256)
    data[:2] = b'MZ'
    struct.pack_into('<I', data, 0x3c, 64)
    data[64:68] = b'PE\0\0'
    struct.pack_into('<H', data, 68, 0x8664)
    struct.pack_into('<H', data, 86, 0x2000)
    struct.pack_into('<H', data, 88, 0x20b)
    return bytes(data)


class CompatibilityTests(unittest.TestCase):
    def test_identity_and_silent_repack(self):
        data = fixture()
        baseline = {'sha256': hashlib.sha256(data).hexdigest()}
        self.assertEqual(dbvo.inspect(data, baseline)['status'], 'reviewed_binary')
        self.assertEqual(dbvo.inspect(data + b'changed', baseline)['status'], 'needs_review')

    def test_malformed_or_non_x64_dll_even_if_hash_matches(self):
        original = fixture()
        for data in [b'', original[:70], original.replace(b'MZ', b'XX'),
                     original[:68] + b'\x4c\x01' + original[70:],
                     original[:86] + b'\0\0' + original[88:]]:
            with self.subTest(size=len(data)):
                self.assertEqual(dbvo.inspect(data, {'sha256': hashlib.sha256(data).hexdigest()})['status'], 'needs_review')

    def run_local(self, data, baseline_data=None):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'tests').mkdir()
            (root / 'tests/dbvo2_baseline.json').write_text(json.dumps({
                'sha256': hashlib.sha256(baseline_data or fixture()).hexdigest(),
                'version': 'fixture-test'}))
            dll = root / 'DBVO.dll'
            if data is not None:
                dll.write_bytes(data)
            report = root / 'report.json'
            with patch.object(dbvo, 'ROOT', root), patch('sys.stdout', new=io.StringIO()):
                code = dbvo.main(['--dll', str(dll), '--report', str(report)])
            return code, json.loads(report.read_text())

    def test_local_reviewed_binary_passes(self):
        code, report = self.run_local(fixture())
        self.assertEqual(code, 0)
        self.assertEqual(report['status'], 'reviewed_binary')
        self.assertEqual(report['reviewed_version'], 'fixture-test')

    def test_local_changed_binary_fails(self):
        code, report = self.run_local(fixture() + b'changed')
        self.assertEqual(code, 1)
        self.assertEqual(report['status'], 'needs_review')

    def test_missing_fixture_blocks(self):
        code, report = self.run_local(None)
        self.assertEqual(code, 2)
        self.assertEqual(report['status'], 'blocked')

    def test_oversized_fixture_blocks(self):
        with patch.object(dbvo, 'DLL_LIMIT', 128):
            code, report = self.run_local(fixture())
        self.assertEqual(code, 2)
        self.assertEqual(report['status'], 'blocked')


if __name__ == '__main__':
    unittest.main()
