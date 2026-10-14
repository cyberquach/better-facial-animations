"""Exercise the real release version selection without compiling or packaging."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ReleaseVersionTests(unittest.TestCase):
    def run_release(self, previous, model=True):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copyfile(ROOT / 'build.sh', root / 'build.sh')
            (root / 'VERSION').write_text('3.2.0\n')
            (root / 'release').mkdir()
            (root / 'release' / (
                f'Better Facial Animations-3.2.0.{previous}-fix-test.7z'
            )).touch()
            for variant in ('with-sentiment', 'without-sentiment'):
                (root / 'release' / f'Better Facial Animations-3.2.0.{previous}-feature-test-{variant}.7z').touch()
            if model:
                (root / 'emotion.onnx').touch()
            (root / 'bin').mkdir()
            make = root / 'bin/make'
            make.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > make-args\nexit 1\n')
            make.chmod(0o755)
            result = subprocess.run(
                ['sh', str(root / 'build.sh'), 'release'],
                env={**os.environ, 'PATH': str(root / 'bin') + ':' + os.environ['PATH'],
                     'JOBS': '1', 'EMOTION_MODEL': str(root / 'emotion.onnx')}, capture_output=True, text=True, timeout=10)
            args = root / 'make-args'
            return (result, args.read_text() if args.exists() else '',
                    (root / 'build').exists(), (root / 'VERSION').read_text())

    def test_last_representable_revision_reaches_compiler(self):
        result, args, built, version = self.run_release(14)
        self.assertIn('BFA_VERSION=3.2.0.15', args)
        self.assertTrue(built)
        self.assertNotIn('SKSE revision limit', result.stderr)
        self.assertEqual(version, '3.2.0\n')

    def test_missing_model_rejects_incomplete_release(self):
        result, args, built, _ = self.run_release(1, model=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Both release archives require EMOTION_MODEL', result.stderr)
        self.assertEqual(args, '')
        self.assertFalse(built)

    def test_overflow_is_rejected_before_build(self):
        for previous in (15, 16, 999):
            with self.subTest(previous=previous):
                result, args, built, version = self.run_release(previous)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('SKSE revision limit', result.stderr)
                self.assertEqual(args, '')
                self.assertFalse(built)
                self.assertEqual(version, '3.2.0\n')


if __name__ == '__main__':
    unittest.main()
