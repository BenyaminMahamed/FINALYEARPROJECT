#!/usr/bin/env python3
"""
System Configuration for Autonomous Vehicle
Student: Benyamin Mahamed (W1966430)
Project: Autonomous Self-Driving Car for Assisted Mobility

Centralizes all system parameters aligned with:
    - NFR-P1: Latency < 200ms
    - NFR-P2: Frame rate >= 8 FPS
    - NFR-S1: Obstacle emergency stop 100% reliable
    - NFR-S2: Manual emergency stop 100% reliable
    - NFR-U1: Override response < 50ms

Supports both hardcoded defaults and optional JSON configuration file
for easy parameter tuning without code modification.
"""

import json
import os
from typing import Dict, Any, Optional

# ============================================================================
# CONFIGURATION CLASS - Supports JSON file loading
# ============================================================================

class SystemConfig:
    """
    System-wide configuration with JSON file support.
    Loads parameters from settings.json if available, otherwise uses defaults.
    """

    def __init__(self, config_file: str = "settings.json"):
        if os.path.exists(config_file):
            self._load_from_file(config_file)
        else:
            self._load_defaults()

    def _load_from_file(self, config_file: str):
        print(f"[CONFIG] Loading from {config_file}")
        try:
            with open(config_file, 'r') as f:
                settings = json.load(f)
            self._apply_settings(settings)
            print(f"[CONFIG] Configuration loaded from file")
        except Exception as e:
            print(f"[CONFIG] Failed to load {config_file}: {e}")
            self._load_defaults()

    def _load_defaults(self):
        print(f"[CONFIG] Using default configuration")
        self._apply_settings(self._get_default_settings())

    def _apply_settings(self, settings: Dict[str, Any]):
        """Apply settings dictionary to configuration properties"""

        # === NFR PERFORMANCE TARGETS ===
        perf = settings.get('performance', {})
        self.LATENCY_TARGET_MS = perf.get('latency_target_ms', 200)
        self.MIN_FPS = perf.get('min_fps', 8)

        # === CAMERA CONFIGURATION ===
        camera = settings.get('camera', {})
        self.CAMERA_WIDTH  = camera.get('width', 640)
        self.CAMERA_HEIGHT = camera.get('height', 480)
        self.CAMERA_FPS    = camera.get('fps', 30)

        # === LANE DETECTION ===
        lane = settings.get('lane_detection', {})
        self.ROI_TOP_RATIO    = lane.get('roi_top_ratio', 0.35)        # FIX: was 0.60
        self.ROI_BOTTOM_RATIO = lane.get('roi_bottom_ratio', 1.0)
        self.BLUR_KERNEL_SIZE = lane.get('blur_kernel_size', 5)
        self.CANNY_LOW_THRESHOLD  = lane.get('canny_low', 50)
        self.CANNY_HIGH_THRESHOLD = lane.get('canny_high', 150)
        self.HOUGH_RHO             = lane.get('hough_rho', 2)
        self.HOUGH_THETA           = lane.get('hough_theta', 1)
        self.HOUGH_THRESHOLD       = lane.get('hough_threshold', 25)
        self.HOUGH_MIN_LINE_LENGTH = lane.get('hough_min_line_length', 20)
        self.HOUGH_MAX_LINE_GAP    = lane.get('hough_max_line_gap', 50)
        self.MIN_LANE_SLOPE        = lane.get('min_lane_slope', 0.5)   # FIX: was 0.3
        self.MAX_LANE_SLOPE        = lane.get('max_lane_slope', 2.0)   # FIX: was 3.0

        # === OBSTACLE DETECTION ===
        obstacle = settings.get('obstacle_detection', {})
        self.SAFETY_ZONE_WIDTH_RATIO  = obstacle.get('safety_zone_width_ratio', 0.4)
        self.SAFETY_ZONE_HEIGHT_RATIO = obstacle.get('safety_zone_height_ratio', 0.5)
        self.OBSTACLE_THRESHOLD_VALUE   = obstacle.get('threshold_value', 60)
        self.OBSTACLE_THRESHOLD_PERCENT = obstacle.get('min_blob_area_percent', 0.8)

        # === MOTOR CONTROL ===
        control = settings.get('control', {})
        self.BASE_SPEED = control.get('base_speed', 10)                # FIX: was 20
        self.MAX_SPEED  = control.get('max_speed', 100)
        self.MIN_SPEED  = control.get('min_speed', 10)
        self.MAX_STEER_ANGLE  = control.get('max_steer_angle', 25)
        self.STEER_SMOOTHING  = control.get('steer_smoothing', 0.75)   # FIX: was 0.5
        self.STEER_KP         = control.get('steer_kp', 0.85)          # FIX: was 0.30
        self.STEER_TRIM       = control.get('steer_trim', 0)

        # === LOGGING & DEBUG ===
        fusion = settings.get('data_fusion', {})
        self.OBSTACLE_PRIORITY = fusion.get('obstacle_priority', True)
        self.FUSION_MODE       = fusion.get('mode', 'priority')

        override = settings.get('remote_override', {})
        self.OVERRIDE_ENABLED    = override.get('enabled', True)
        self.OVERRIDE_TIMEOUT_MS = override.get('timeout_ms', 50)

        log_cfg = settings.get('logging', {})
        self.LOG_TELEMETRY      = log_cfg.get('enable_telemetry', True)
        self.LOG_DIRECTORY      = log_cfg.get('directory', 'test_logs')
        self.FRAME_LOG_INTERVAL = log_cfg.get('frame_interval', 30)

        debug = settings.get('debug', {})
        self.DEBUG_MODE        = debug.get('enable', True)
        self.SHOW_ROI          = debug.get('show_roi', True)
        self.SHOW_LANE_LINES   = debug.get('show_lane_lines', True)
        self.SHOW_SAFETY_ZONE  = debug.get('show_safety_zone', True)
        self.SAVE_DEBUG_FRAMES = debug.get('save_frames', False)

    @staticmethod
    def _get_default_settings() -> Dict[str, Any]:
        """Provides the baseline defaults if settings.json is missing."""
        return {
            "performance": {"latency_target_ms": 200, "min_fps": 8},
            "camera": {"width": 640, "height": 480, "fps": 30},
            "lane_detection": {
                "roi_top_ratio": 0.35,          # FIX: was 0.60
                "roi_bottom_ratio": 1.0,
                "blur_kernel_size": 5,
                "canny_low": 50, "canny_high": 150,
                "hough_rho": 2, "hough_theta": 1,
                "hough_threshold": 25,
                "hough_min_line_length": 20,
                "hough_max_line_gap": 50,
                "min_lane_slope": 0.5,          # FIX: was 0.3
                "max_lane_slope": 2.0           # FIX: was 3.0
            },
            "obstacle_detection": {
                "safety_zone_width_ratio": 0.4, "safety_zone_height_ratio": 0.5,
                "threshold_value": 60, "min_blob_area_percent": 0.8
            },
            "control": {
                "base_speed": 10,               # FIX: was 20
                "max_speed": 100, "min_speed": 10,
                "max_steer_angle": 25,
                "steer_smoothing": 0.75,        # FIX: was 0.5
                "steer_kp": 0.85,               # FIX: was 0.30
                "steer_trim": 0
            },
            "data_fusion": {"obstacle_priority": True, "mode": "priority"},
            "remote_override": {"enabled": True, "timeout_ms": 50},
            "logging": {"enable_telemetry": True, "directory": "test_logs", "frame_interval": 30},
            "debug": {"enable": True, "show_roi": True, "show_lane_lines": True, "show_safety_zone": True, "save_frames": False}
        }

    def export_to_json(self, filename: str = "settings_export.json"):
        settings = self._get_default_settings()
        with open(filename, 'w') as f:
            json.dump(settings, f, indent=2)
        print(f"[CONFIG] Configuration exported to {filename}")

    def print_summary(self):
        print("\n" + "=" * 60)
        print("SYSTEM CONFIGURATION SUMMARY")
        print("=" * 60)
        print(f"Latency Target: < {self.LATENCY_TARGET_MS}ms | FPS Min: {self.MIN_FPS}")
        print(f"Speed: {self.BASE_SPEED} | Steer Kp: {self.STEER_KP} | Trim: {self.STEER_TRIM:+d}")
        print("=" * 60 + "\n")


# ============================================================================
# GLOBAL INSTANCE
# ============================================================================
config = SystemConfig()


# ============================================================================
# BACKWARD COMPATIBILITY MAPPINGS (THE FIX IS HERE)
# ============================================================================
LATENCY_TARGET_MS = config.LATENCY_TARGET_MS
MIN_FPS           = config.MIN_FPS

CAMERA_WIDTH      = config.CAMERA_WIDTH
CAMERA_HEIGHT     = config.CAMERA_HEIGHT
CAMERA_FPS        = config.CAMERA_FPS

ROI_TOP_RATIO     = config.ROI_TOP_RATIO
ROI_BOTTOM_RATIO  = config.ROI_BOTTOM_RATIO
BLUR_KERNEL_SIZE  = config.BLUR_KERNEL_SIZE
CANNY_LOW_THRESHOLD   = config.CANNY_LOW_THRESHOLD
CANNY_HIGH_THRESHOLD  = config.CANNY_HIGH_THRESHOLD
HOUGH_RHO             = config.HOUGH_RHO
HOUGH_THETA           = config.HOUGH_THETA
HOUGH_THRESHOLD       = config.HOUGH_THRESHOLD
HOUGH_MIN_LINE_LENGTH = config.HOUGH_MIN_LINE_LENGTH
HOUGH_MAX_LINE_GAP    = config.HOUGH_MAX_LINE_GAP
MIN_LANE_SLOPE        = config.MIN_LANE_SLOPE
MAX_LANE_SLOPE        = config.MAX_LANE_SLOPE

SAFETY_ZONE_WIDTH_RATIO    = config.SAFETY_ZONE_WIDTH_RATIO
SAFETY_ZONE_HEIGHT_RATIO   = config.SAFETY_ZONE_HEIGHT_RATIO
OBSTACLE_THRESHOLD_VALUE   = config.OBSTACLE_THRESHOLD_VALUE
OBSTACLE_THRESHOLD_PERCENT = config.OBSTACLE_THRESHOLD_PERCENT

BASE_SPEED        = config.BASE_SPEED
MAX_SPEED         = config.MAX_SPEED
MIN_SPEED         = config.MIN_SPEED
MAX_STEER_ANGLE   = config.MAX_STEER_ANGLE
STEER_SMOOTHING   = config.STEER_SMOOTHING
STEER_KP          = config.STEER_KP
STEER_TRIM        = config.STEER_TRIM

OBSTACLE_PRIORITY = config.OBSTACLE_PRIORITY
FUSION_MODE       = config.FUSION_MODE
OVERRIDE_ENABLED  = config.OVERRIDE_ENABLED
OVERRIDE_TIMEOUT_MS = config.OVERRIDE_TIMEOUT_MS

LOG_TELEMETRY      = config.LOG_TELEMETRY
LOG_DIRECTORY      = config.LOG_DIRECTORY
FRAME_LOG_INTERVAL = config.FRAME_LOG_INTERVAL

DEBUG_MODE        = config.DEBUG_MODE
SHOW_ROI          = config.SHOW_ROI
SHOW_LANE_LINES   = config.SHOW_LANE_LINES
SHOW_SAFETY_ZONE  = config.SHOW_SAFETY_ZONE
SAVE_DEBUG_FRAMES = config.SAVE_DEBUG_FRAMES

if __name__ == "__main__":
    config.print_summary()
