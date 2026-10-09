"""Every src/**/*.cc has a unique basename: tools/build.mk finds sources by
vpath, so two files of one name would build the first found twice."""
from collections import Counter
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class SourceBasenameTests(unittest.TestCase):
    def test_unique(self):
        names = Counter(p.name for p in (ROOT / 'src').rglob('*.cc'))
        self.assertGreater(len(names), 10)
        self.assertEqual([n for n, c in names.items() if c > 1], [])

    def test_vpath_covers_every_directory(self):
        make = (ROOT / 'tools/build.mk').read_text()
        for directory in {p.parent.relative_to(ROOT) for p in (ROOT / 'src').rglob('*.cc')}:
            self.assertIn(str(directory), make)


if __name__ == '__main__':
    unittest.main()
