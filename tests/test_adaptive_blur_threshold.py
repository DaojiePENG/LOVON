import unittest

from deploy.adaptive_blur_threshold import AdaptiveBlurThreshold


class AdaptiveBlurThresholdTest(unittest.TestCase):
    def test_updates_only_after_two_second_window(self):
        gate = AdaptiveBlurThreshold()

        self.assertIsNone(gate.observe(100.0, timestamp=0.0))
        self.assertIsNone(gate.observe(100.0, timestamp=1.99))
        update = gate.observe(100.0, timestamp=2.0)

        self.assertIsNotNone(update)
        self.assertEqual(update.sample_count, 3)
        self.assertAlmostEqual(update.lower_quantile, 100.0)
        self.assertAlmostEqual(gate.threshold, 140.0)

    def test_each_window_uses_only_its_own_raw_variances(self):
        gate = AdaptiveBlurThreshold()
        gate.observe(100.0, timestamp=0.0)
        gate.observe(100.0, timestamp=2.0)

        gate.observe(200.0, timestamp=2.1)
        update = gate.observe(200.0, timestamp=4.0)

        self.assertAlmostEqual(update.lower_quantile, 200.0)
        self.assertAlmostEqual(gate.threshold, 152.0)

    def test_clips_threshold_to_half_and_one_and_a_half_initial_value(self):
        low_gate = AdaptiveBlurThreshold(alpha=1.0)
        low_gate.observe(0.0, timestamp=0.0)
        low_gate.observe(0.0, timestamp=2.0)
        self.assertAlmostEqual(low_gate.threshold, 75.0)

        high_gate = AdaptiveBlurThreshold(alpha=1.0)
        high_gate.observe(1000.0, timestamp=0.0)
        high_gate.observe(1000.0, timestamp=2.0)
        self.assertAlmostEqual(high_gate.threshold, 225.0)


if __name__ == "__main__":
    unittest.main()
