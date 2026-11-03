"""Standard-library guard against private host details in public training tools."""
import ipaddress
from pathlib import Path
import re
import unittest


class TrainingPortability(unittest.TestCase):
    def test_no_host_configuration(self):
        root = Path(__file__).resolve().parents[1]
        files = list((root/'tools/training').glob('*.py'))
        files += [root/'tools/training/README.md', root/'tools/emotion_student.py',
                  root/'tools/export_emotion.py', root/'tools/export_performance_stats.py']
        for path in files:
            text = path.read_text()
            self.assertIsNone(re.search(r'/(?:home|Users)/[^/\s]+/|[A-Za-z]:[\\/]+Users[\\/]+', text), path)
            for candidate in re.findall(r'(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])', text):
                address = ipaddress.ip_address(candidate)
                a, b, _, _ = address.packed
                self.assertFalse(a == 10 or (a == 172 and 16 <= b < 32) or (a, b) == (192, 168), path)
            self.assertIsNone(re.search(r'(?i)ssh://|\bssh\s+\w+@|IdentityFile|HostName\s+', text), path)


if __name__ == '__main__':
    unittest.main()
