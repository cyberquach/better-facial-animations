"""Every src/**/*.cc has a unique basename: the compiler output paths use
these names, so duplicates would overwrite an object."""
from collections import Counter
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class SourceBasenameTests(unittest.TestCase):
    def test_unique(self):
        names = Counter(p.name for p in (ROOT / 'src').rglob('*.cc'))
        self.assertGreater(len(names), 10)
        self.assertEqual([n for n, c in names.items() if c > 1], [])



if __name__ == '__main__':
    unittest.main()
