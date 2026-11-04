"""Run with: python -m unittest discover -s ai/tests -v"""
import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from occupancy_fusion import FusionEngine, Observation  # noqa: E402


NOW = datetime(2026, 11, 4, tzinfo=timezone.utc)


def observation(camera, probability, *, visibility=1, quality=1, when=NOW):
    return Observation(camera, "A1", probability, visibility, quality, when)


class FusionTests(unittest.TestCase):
    def test_requires_multiple_stable_observations(self):
        engine = FusionEngine(stable_updates=2)
        first = engine.observe(observation("cam1", .98), now=NOW)
        second = engine.observe(observation("cam1", .98, when=NOW + timedelta(seconds=1)),
                                now=NOW + timedelta(seconds=1))
        self.assertEqual(first["status"], "unknown")
        self.assertEqual(second["status"], "occupied")

    def test_conflicting_cameras_return_unknown(self):
        engine = FusionEngine(stable_updates=1)
        engine.observe(observation("cam1", .99), now=NOW)
        result = engine.observe(observation("cam2", .01), now=NOW)
        self.assertEqual(result["status"], "unknown")
        self.assertEqual(result["candidate"], "unknown")

    def test_expired_observations_excluded(self):
        engine = FusionEngine(max_age_seconds=10, stable_updates=1)
        engine.observe(observation("cam1", .98), now=NOW)
        self.assertEqual(engine.fuse("A1", now=NOW + timedelta(seconds=12))["status"], "unknown")

    def test_visibility_quality_reduce_influence(self):
        engine = FusionEngine(stable_updates=1, min_confidence=.70)
        engine.observe(observation("cam1", .97), now=NOW)
        result = engine.observe(observation("cam2", .03, visibility=.16, quality=.16), now=NOW)
        self.assertEqual(result["status"], "occupied")
        self.assertGreater(result["occupied_probability"], .90)

    def test_out_of_order_observation_does_not_replace_latest(self):
        engine = FusionEngine(stable_updates=1)
        engine.observe(observation("cam1", .98), now=NOW)
        result = engine.observe(observation("cam1", .02, when=NOW - timedelta(seconds=1)), now=NOW)
        self.assertEqual(result["status"], "occupied")

    def test_validation_rejects_impossible_probability(self):
        with self.assertRaises(ValueError):
            observation("cam1", 1.5)

    def test_naive_timestamp_rejected(self):
        with self.assertRaises(ValueError):
            observation("cam1", .9, when=datetime(2026, 11, 4))


if __name__ == "__main__":
    unittest.main()
