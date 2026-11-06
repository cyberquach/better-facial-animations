"""Negative controls for the reusable audit's skip and surface gates."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools/face_audit'))
from checks import can_skip_speech, surface_regressions


class AuditGates(unittest.TestCase):
    def test_contact_only_changes_cannot_skip(self):
        plain = dict(fingerprint='same', has_repair_state=False)
        contact = dict(fingerprint='same', has_repair_state=True)
        self.assertTrue(can_skip_speech(plain, plain))
        self.assertFalse(can_skip_speech(plain, contact))
        self.assertFalse(can_skip_speech(contact, plain))
        self.assertFalse(can_skip_speech(contact, contact))
        self.assertFalse(can_skip_speech(plain, dict(fingerprint='same')))
        self.assertFalse(can_skip_speech(plain, dict(plain, fingerprint='other')))

    def test_each_surface_failure_is_independent(self):
        base = dict(edges=100, p95_strain=0.5, over_double_length=4)
        self.assertEqual(surface_regressions(base, base), [])
        self.assertEqual(surface_regressions(base, dict(base, over_double_length=5)),
                         ['upper_surface_severe_count_worse'])
        self.assertEqual(surface_regressions(base, dict(base, p95_strain=0.502)),
                         ['upper_surface_p95_worse'])
        self.assertEqual(surface_regressions(base, {}), ['upper_surface_unmeasured'])


if __name__ == '__main__':
    unittest.main()
