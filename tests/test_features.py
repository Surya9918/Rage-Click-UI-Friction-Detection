"""Unit tests for feature engineering calculations."""

import unittest
from src.feature_engineering import FeatureExtractor


class TestFeatureEngineering(unittest.TestCase):

    def setUp(self):
        self.config = {
            "window_seconds": 5.0,
            "rapid_click_interval_ms": 300.0,
            "direction_change_angle": 90.0,
            "movement_threshold_px": 5.0,
            "scroll_window_seconds": 2.0,
            "velocity_spike_clamp_px_s": 10000.0,
        }
        self.extractor = FeatureExtractor(self.config)

    def test_empty_events(self):
        feats = self.extractor.extract_features([])
        self.assertEqual(feats["click_frequency"], 0.0)
        self.assertEqual(feats["rapid_fire_clicks"], 0)
        self.assertEqual(feats["maximum_cursor_velocity"], 0.0)
        self.assertEqual(feats["erratic_direction_changes"], 0)
        self.assertEqual(feats["scroll_thrashing"], 0.0)

    def test_click_frequency_and_normal_clicks(self):
        events = [
            {"timestamp": 1000, "event_type": "mousedown", "target_element": "btn1"},
            {"timestamp": 2500, "event_type": "mousedown", "target_element": "btn2"},
            {"timestamp": 4000, "event_type": "mousedown", "target_element": "btn3"},
        ]
        feats = self.extractor.extract_features(events, window_duration=5.0)
        self.assertAlmostEqual(feats["click_frequency"], 3 / 5.0, places=3)
        self.assertEqual(feats["rapid_fire_clicks"], 0)

    def test_rapid_fire_clicks(self):
        events = [
            {"timestamp": 1000, "event_type": "mousedown", "target_element": "buttonA"},
            {"timestamp": 1200, "event_type": "mousedown", "target_element": "buttonA"},
            {"timestamp": 1400, "event_type": "mousedown", "target_element": "buttonA"},
            {"timestamp": 1600, "event_type": "mousedown", "target_element": "buttonA"},
        ]
        feats = self.extractor.extract_features(events, window_duration=5.0)
        self.assertEqual(feats["rapid_fire_clicks"], 3)
        self.assertAlmostEqual(feats["click_frequency"], 4 / 5.0, places=3)

    def test_rapid_fire_different_targets_not_counted(self):
        events = [
            {"timestamp": 1000, "event_type": "mousedown", "target_element": "buttonA"},
            {"timestamp": 1150, "event_type": "mousedown", "target_element": "buttonB"},
            {"timestamp": 1300, "event_type": "mousedown", "target_element": "buttonC"},
        ]
        feats = self.extractor.extract_features(events, window_duration=5.0)
        self.assertEqual(feats["rapid_fire_clicks"], 0)

    def test_cursor_velocity_calculation_and_zero_dt(self):
        events = [
            {"timestamp": 1000, "event_type": "mousemove", "x_coordinate": 0, "y_coordinate": 0},
            {"timestamp": 1000, "event_type": "mousemove", "x_coordinate": 0, "y_coordinate": 0},
            {"timestamp": 1100, "event_type": "mousemove", "x_coordinate": 100, "y_coordinate": 0},
        ]
        feats = self.extractor.extract_features(events, window_duration=5.0)
        self.assertAlmostEqual(feats["maximum_cursor_velocity"], 1000.0, delta=10.0)

    def test_erratic_direction_changes(self):
        events = [
            {"timestamp": 1000, "event_type": "mousemove", "x_coordinate": 0, "y_coordinate": 0},
            {"timestamp": 1100, "event_type": "mousemove", "x_coordinate": 100, "y_coordinate": 0},
            {"timestamp": 1200, "event_type": "mousemove", "x_coordinate": 10, "y_coordinate": 0},
        ]
        feats = self.extractor.extract_features(events, window_duration=5.0)
        self.assertEqual(feats["erratic_direction_changes"], 1)

    def test_jitter_ignored_for_direction_changes(self):
        events = [
            {"timestamp": 1000, "event_type": "mousemove", "x_coordinate": 100, "y_coordinate": 100},
            {"timestamp": 1100, "event_type": "mousemove", "x_coordinate": 101, "y_coordinate": 100},
            {"timestamp": 1200, "event_type": "mousemove", "x_coordinate": 100, "y_coordinate": 100},
        ]
        feats = self.extractor.extract_features(events, window_duration=5.0)
        self.assertEqual(feats["erratic_direction_changes"], 0)

    def test_scroll_thrashing_reversals(self):
        events = [
            {"timestamp": 1000, "event_type": "scroll", "scroll_y": 0},
            {"timestamp": 1200, "event_type": "scroll", "scroll_y": 500},
            {"timestamp": 1400, "event_type": "scroll", "scroll_y": 800},
            {"timestamp": 1800, "event_type": "scroll", "scroll_y": 400},
            {"timestamp": 2100, "event_type": "scroll", "scroll_y": 100},
            {"timestamp": 2500, "event_type": "scroll", "scroll_y": 550},
        ]
        feats = self.extractor.extract_features(events, window_duration=5.0)
        self.assertEqual(feats["scroll_thrashing"], 850.0)


if __name__ == "__main__":
    unittest.main()
