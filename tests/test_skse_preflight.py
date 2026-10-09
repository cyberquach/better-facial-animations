"""Real compiled PE exports exercise positive, negative and unknown controls."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class PreflightTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.directory = Path(cls.temp.name)
        source = cls.directory / 'fixture.cc'
        source.write_text('''#include "a2f/runtime/skse.h"
#ifndef FIXTURE_FLAGS
#define FIXTURE_FLAGS 4
#endif
extern "C" __declspec(dllexport) skse::SKSEPluginVersionData SKSEPlugin_Version = {
  1, 1, "fixture", "test", "", 0, FIXTURE_FLAGS, {skse::Version(1,6,1170),0}, 0};
''')
        cls.dll = cls.directory / 'fixture.dll'
        subprocess.run(shlex.split(os.environ.get('CXX', 'x86_64-w64-mingw32-g++')) +
                       ['-shared', '-I', str(ROOT / 'include'), str(source), '-o', str(cls.dll)],
                       check=True, capture_output=True, timeout=60)

    def check(self, path, runtime='1.6.1170'):
        result = subprocess.run([sys.executable, str(ROOT/'tools/skse_preflight.py'),
                                 str(path), '--runtime', runtime],
                                capture_output=True, text=True, timeout=10)
        return result.returncode, json.loads(result.stdout)

    def test_declared_match_and_mismatch(self):
        code, result = self.check(self.dll)
        self.assertEqual(code, 0)
        self.assertEqual(result['declaration']['explicit_runtimes'], ['1.6.1170.0'])
        self.assertEqual(self.check(self.dll, '1.6.640')[0], 1)

    def test_runtime_independence_is_not_a_false_mismatch(self):
        path = self.directory/'independent.dll'
        subprocess.run(shlex.split(os.environ.get('CXX', 'x86_64-w64-mingw32-g++')) +
                       ['-shared', '-DFIXTURE_FLAGS=5', '-I', str(ROOT / 'include'),
                        str(self.directory/'fixture.cc'), '-o', str(path)],
                       check=True, capture_output=True, timeout=60)
        self.assertEqual(self.check(path, '1.6.640')[0], 2)

    def test_no_export_is_unknown(self):
        # Equal-length export-name change retains an otherwise valid PE.
        path = self.directory/'legacy.dll'
        path.write_bytes(self.dll.read_bytes().replace(b'SKSEPlugin_Version\0', b'AnotherExportNameX\0'))
        self.assertEqual(self.check(path)[0], 2)

    def test_missing_and_truncated_cannot_pass(self):
        self.assertEqual(self.check(self.directory/'missing.dll')[0], 2)
        data = self.dll.read_bytes()
        path = self.directory/'truncated.dll'
        for size in (0, 63, 128, len(data)//4):
            path.write_bytes(data[:size])
            with self.subTest(size=size):
                self.assertEqual(self.check(path)[0], 2)


if __name__ == '__main__':
    unittest.main()
