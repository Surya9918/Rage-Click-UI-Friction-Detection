"""Feature engineering module for calculating the 5 friction features from raw telemetry windows."""

import math
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd

from src.utils import load_config, get_logger

logger = get_logger(__name__)


class FeatureExtractor:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            config = load_config()
        self.config = config

        self.window_seconds: float = float(config.get("window_seconds", 5.0))
        self.rapid_click_interval_ms: float = float(config.get("rapid_click_interval_ms", 300.0))
        self.direction_change_angle: float = float(config.get("direction_change_angle", 90.0))
        self.movement_threshold_px: float = float(config.get("movement_threshold_px", 5.0))
        self.scroll_window_seconds: float = float(config.get("scroll_window_seconds", 2.0))
        self.velocity_spike_clamp: float = float(config.get("velocity_spike_clamp_px_s", 10000.0))

        self.feature_names = [
            "click_frequency",
            "rapid_fire_clicks",
            "maximum_cursor_velocity",
            "erratic_direction_changes",
            "scroll_thrashing",
        ]

    def extract_features(self, events: List[Dict[str, Any]], window_duration: Optional[float] = None) -> Dict[str, float]:
        duration = window_duration if window_duration is not None else self.window_seconds
        if duration <= 0:
            duration = 5.0

        if not events:
            return {
                "click_frequency": 0.0,
                "rapid_fire_clicks": 0.0,
                "maximum_cursor_velocity": 0.0,
                "erratic_direction_changes": 0.0,
                "scroll_thrashing": 0.0,
            }

        sorted_events = sorted(events, key=lambda e: float(e.get("timestamp", 0)))

        click_freq = self._calculate_click_frequency(sorted_events, duration)
        rapid_clicks = self._calculate_rapid_fire_clicks(sorted_events)
        max_velocity = self._calculate_maximum_cursor_velocity(sorted_events)
        erratic_changes = self._calculate_erratic_direction_changes(sorted_events)
        scroll_thrashing = self._calculate_scroll_thrashing(sorted_events)

        return {
            "click_frequency": float(round(click_freq, 4)),
            "rapid_fire_clicks": float(rapid_clicks),
            "maximum_cursor_velocity": float(round(max_velocity, 2)),
            "erratic_direction_changes": float(erratic_changes),
            "scroll_thrashing": float(round(scroll_thrashing, 2)),
        }

    def _calculate_click_frequency(self, events: List[Dict[str, Any]], duration: float) -> float:
        clicks = [e for e in events if e.get("event_type") in ("mousedown", "click")]
        return len(clicks) / duration

    def _calculate_rapid_fire_clicks(self, events: List[Dict[str, Any]]) -> int:
        clicks = [e for e in events if e.get("event_type") in ("mousedown", "click")]
        if len(clicks) < 2:
            return 0

        rapid_count = 0
        for i in range(1, len(clicks)):
            prev_click = clicks[i - 1]
            curr_click = clicks[i]

            prev_target = str(prev_click.get("target_element", "")).strip()
            curr_target = str(curr_click.get("target_element", "")).strip()

            prev_time = float(prev_click.get("timestamp", 0))
            curr_time = float(curr_click.get("timestamp", 0))
            delta_ms = curr_time - prev_time

            if delta_ms < 0:
                continue
            if delta_ms < 1.0 and (curr_time < 1e6 and prev_time < 1e6):
                delta_ms *= 1000.0

            if prev_target and curr_target and prev_target == curr_target:
                if 0 <= delta_ms <= self.rapid_click_interval_ms:
                    rapid_count += 1

        return rapid_count

    def _calculate_maximum_cursor_velocity(self, events: List[Dict[str, Any]]) -> float:
        moves = [
            e for e in events
            if e.get("event_type") in ("mousemove", "mousedown", "mouseup")
            and "x_coordinate" in e and "y_coordinate" in e
        ]
        if len(moves) < 2:
            return 0.0

        max_vel = 0.0
        for i in range(1, len(moves)):
            p1 = moves[i - 1]
            p2 = moves[i]

            t1 = float(p1.get("timestamp", 0))
            t2 = float(p2.get("timestamp", 0))
            dt = t2 - t1

            if dt <= 0:
                continue

            dt_seconds = dt / 1000.0 if dt > 1.0 else dt
            if dt_seconds <= 0.0001:
                continue

            x1, y1 = float(p1["x_coordinate"]), float(p1["y_coordinate"])
            x2, y2 = float(p2["x_coordinate"]), float(p2["y_coordinate"])

            dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            velocity = dist / dt_seconds

            if velocity > self.velocity_spike_clamp:
                velocity = self.velocity_spike_clamp

            if velocity > max_vel:
                max_vel = velocity

        return max_vel

    def _calculate_erratic_direction_changes(self, events: List[Dict[str, Any]]) -> int:
        moves = [
            e for e in events
            if e.get("event_type") in ("mousemove", "mousedown", "mouseup")
            and "x_coordinate" in e and "y_coordinate" in e
        ]
        if len(moves) < 3:
            return 0

        valid_vectors = []
        last_x = float(moves[0]["x_coordinate"])
        last_y = float(moves[0]["y_coordinate"])

        for i in range(1, len(moves)):
            curr_x = float(moves[i]["x_coordinate"])
            curr_y = float(moves[i]["y_coordinate"])
            dx = curr_x - last_x
            dy = curr_y - last_y
            dist = math.hypot(dx, dy)

            if dist >= self.movement_threshold_px:
                valid_vectors.append((dx, dy, dist))
                last_x = curr_x
                last_y = curr_y

        if len(valid_vectors) < 2:
            return 0

        erratic_count = 0
        for i in range(1, len(valid_vectors)):
            v1_x, v1_y, d1 = valid_vectors[i - 1]
            v2_x, v2_y, d2 = valid_vectors[i]

            dot = (v1_x * v2_x) + (v1_y * v2_y)
            cos_theta = dot / (d1 * d2)
            cos_theta = max(-1.0, min(1.0, cos_theta))

            angle_deg = math.degrees(math.acos(cos_theta))
            if angle_deg > self.direction_change_angle:
                erratic_count += 1

        return erratic_count

    def _calculate_scroll_thrashing(self, events: List[Dict[str, Any]]) -> float:
        scrolls = [e for e in events if e.get("event_type") == "scroll" and "scroll_y" in e]
        if len(scrolls) < 2:
            return 0.0

        scroll_deltas = []
        for i in range(1, len(scrolls)):
            prev_sy = float(scrolls[i - 1]["scroll_y"])
            curr_sy = float(scrolls[i]["scroll_y"])
            t = float(scrolls[i].get("timestamp", 0))

            dy = curr_sy - prev_sy
            if abs(dy) > 0.01:
                scroll_deltas.append((t, dy))

        if len(scroll_deltas) < 2:
            return 0.0

        total_reversed_distance = 0.0
        for i in range(1, len(scroll_deltas)):
            t_curr, dy_curr = scroll_deltas[i]
            t_prev, dy_prev = scroll_deltas[i - 1]

            if (dy_curr * dy_prev) < 0:
                dt = abs(t_curr - t_prev)
                dt_s = dt / 1000.0 if dt > 10.0 else dt
                if dt_s <= self.scroll_window_seconds:
                    total_reversed_distance += abs(dy_curr)

        return total_reversed_distance

    def extract_from_dataframe(self, df: pd.DataFrame, window_col: str = "window_id") -> pd.DataFrame:
        records = []
        for window_id, group in df.groupby(window_col):
            events = group.to_dict(orient="records")
            feats = self.extract_features(events)
            feats[window_col] = window_id
            records.append(feats)
        return pd.DataFrame(records)
