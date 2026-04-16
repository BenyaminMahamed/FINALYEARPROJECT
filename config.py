#!/usr/bin/env python3
"""
System Configuration for Autonomous Vehicle
Student: Benyamin Mahamed (W1966430)
Project: Autonomous Self-Driving Car for Assisted Mobility

Centralizes all system parameters aligned with:
    - NFR-P1: Latency < 200ms
    - NFR-P2: Frame rate ≥ 8 FPS
    - NFR-S1: Obstacle emergency stop 100% reliable
    - NFR-S2: Manual emergency stop 100% reliable
    - NFR-U1: Override response < 50ms

Supports both hardcoded defaults and optional JSON configuration file
for easy parameter tuning without code modification.

Design Philosophy: Classical CV prioritized over Deep Learning for:
    - Deterministic behavior (safety-critical)
    - Real-time performance on embedded CPU
    - No training data required
    - Explainable results
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
    Enables parameter tuning without code modification for final testing.
    """
    
    def __init__(self, config_file: str = "settings.json"):
        """
        Initialize configuration from file or defaults.
        
        Args:
            config_file: Path to JSON configuration file (optional)
        """
        # Attempt to load from file
        if os.path.exists(config_file):
            self._load_from_file(config_file)
        else:
            self._load_defaults()
    
    def _load_from_file(self, config_file: str):
        """Load configuration from JSON file"""
        print(f"[CONFIG] Loading from {config_file}")
        try:
            with open(config_file, 'r') as f:
                settings = json.load(f)
            self._apply_settings(settings)
            print(f"[CONFIG] ✓ Configuration loaded from file")
        except Exception as e:
            print(f"[CONFIG] ✗ Failed to load {config_file}: {e}")
            print(f"[CONFIG] Using default settings")
            self._load_defaults()
    
    def _load_defaults(self):
        """Load default configuration values"""
        print(f"[CONFIG] Using default configuration")
        self._apply_settings(self._get_default_settings())
    
    def _apply_settings(self, settings: Dict[str, Any]):
        """Apply settings dictionary to configuration"""
        
        # === NFR PERFORMANCE TARGETS ===
        perf = settings.get('performance', {})
        self.LATENCY_TARGET_MS = perf.get('latency_target_ms', 200)  # NFR-P1
        self.MIN_FPS = perf.get('min_fps', 8)                        # NFR-P2
        
        # === CAMERA CONFIGURATION ===
        camera = settings.get('camera', {})
        self.CAMERA_WIDTH = camera.get('width', 640)
        self.CAMERA_HEIGHT = camera.get('height', 480)
        self.CAMERA_FPS = camera.get('fps', 30)
        
        # === LANE DETECTION (Classical CV - Canny + Hough) ===
        lane = settings.get('lane_detection', {})
        
        # Region of Interest
        self.ROI_TOP_RATIO = lane.get('roi_top_ratio', 0.55)
        self.ROI_BOTTOM_RATIO = lane.get('roi_bottom_ratio', 1.0)
        
        # Preprocessing
        self.BLUR_KERNEL_SIZE = lane.get('blur_kernel_size', 5)
        
        # Canny Edge Detection
        self.CANNY_LOW_THRESHOLD = lane.get('canny_low', 50)
        self.CANNY_HIGH_THRESHOLD = lane.get('canny_high', 150)
        
        # Hough Transform
        self.HOUGH_RHO = lane.get('hough_rho', 2)
        self.HOUGH_THETA = lane.get('hough_theta', 1)
        self.HOUGH_THRESHOLD = lane.get('hough_threshold', 50)
        self.HOUGH_MIN_LINE_LENGTH = lane.get('hough_min_line_length', 40)
        self.HOUGH_MAX_LINE_GAP = lane.get('hough_max_line_gap', 20)
        
        # Lane line filtering
        self.MIN_LANE_SLOPE = lane.get('min_lane_slope', 0.3)
        self.MAX_LANE_SLOPE = lane.get('max_lane_slope', 3.0)
        
        # === OBSTACLE DETECTION (Classical CV - Blob Detection) ===
        obstacle = settings.get('obstacle_detection', {})
        
        self.SAFETY_ZONE_WIDTH_RATIO = obstacle.get('safety_zone_width_ratio', 0.4)
        self.SAFETY_ZONE_HEIGHT_RATIO = obstacle.get('safety_zone_height_ratio', 0.5)
        self.OBSTACLE_THRESHOLD_VALUE = obstacle.get('threshold_value', 60)
        self.OBSTACLE_THRESHOLD_PERCENT = obstacle.get('min_blob_area_percent', 0.15)
        
        # === MOTOR CONTROL PARAMETERS ===
        control = settings.get('control', {})
        
        self.BASE_SPEED = control.get('base_speed', 30)
        self.MAX_SPEED = control.get('max_speed', 100)
        self.MIN_SPEED = control.get('min_speed', 10)
        
        # Steering
        self.MAX_STEER_ANGLE = control.get('max_steer_angle', 30)
        self.STEER_SMOOTHING = control.get('steer_smoothing', 0.3)
        self.STEER_KP = control.get('steer_kp', 0.5)
        
        # === DATA FUSION LOGIC (FR1.2) ===
        fusion = settings.get('data_fusion', {})
        
        self.OBSTACLE_PRIORITY = fusion.get('obstacle_priority', True)
        self.FUSION_MODE = fusion.get('mode', 'priority')
        
        # === REMOTE OVERRIDE (FR3.1, NFR-U1) ===
        override = settings.get('remote_override', {})
        
        self.OVERRIDE_ENABLED = override.get('enabled', True)
        self.OVERRIDE_TIMEOUT_MS = override.get('timeout_ms', 50)  # NFR-U1
        
        # === LOGGING & TELEMETRY (FR4.1) ===
        logging = settings.get('logging', {})
        
        self.LOG_TELEMETRY = logging.get('enable_telemetry', True)
        self.LOG_DIRECTORY = logging.get('directory', 'test_logs')
        self.FRAME_LOG_INTERVAL = logging.get('frame_interval', 30)
        
        # === DEBUG VISUALIZATION ===
        debug = settings.get('debug', {})
        
        self.DEBUG_MODE = debug.get('enable', True)
        self.SHOW_ROI = debug.get('show_roi', True)
        self.SHOW_LANE_LINES = debug.get('show_lane_lines', True)
        self.SHOW_SAFETY_ZONE = debug.get('show_safety_zone', True)
        self.SAVE_DEBUG_FRAMES = debug.get('save_frames', False)
    
@staticmethod
    def _get_default_settings() -> Dict[str, Any]:
        """
        Get default configuration settings.
        """
        return {
            "performance": {
                "latency_target_ms": 200,
                "min_fps": 8
            },
            "camera": {
                "width": 640,
                "height": 480,
                "fps": 30
            },
            "lane_detection": {
                "roi_top_ratio": 0.65,      # Adjusted to look closer at floor
                "roi_bottom_ratio": 1.0,
                "blur_kernel_size": 5,
                "canny_low": 50,
                "canny_high": 150,
                "hough_rho": 2,
                "hough_theta": 1,
                "hough_threshold": 20,      # Optimized sensitive Hough
                "hough_min_line_length": 15,
                "hough_max_line_gap": 50,
                "min_lane_slope": 0.3,
                "max_lane_slope": 3.0
            },
            "obstacle_detection": {
                "safety_zone_width_ratio": 0.4,
                "safety_zone_height_ratio": 0.5,
                "threshold_value": 60,
                "min_blob_area_percent": 0.8 # Ignores floor noise
            },
            "control": {
                "base_speed": 30,
                "max_speed": 100,
                "min_speed": 10,
                "max_steer_angle": 30,
                "steer_smoothing": 0.3,
                "steer_kp": 0.3              # Smoother assisted steering
            },
            "data_fusion": {
                "obstacle_priority": True,
                "mode": "priority"
            },
            "remote_override": {
                "enabled": True,
                "timeout_ms": 50
            },
            "logging": {
                "enable_telemetry": True,
                "directory": "test_logs",
                "frame_interval": 30
            },
            "debug": {
                "enable": True,
                "show_roi": True,
                "show_lane_lines": True,
                "show_safety_zone": True,
                "save_frames": False
            }
        }
    
    def export_to_json(self, filename: str = "settings_export.json"):
        """
        Export current configuration to JSON file.
        
        Useful for saving tuned parameters after testing.
        
        Args:
            filename: Output JSON filename
        """
        settings = {
            "performance": {
                "latency_target_ms": self.LATENCY_TARGET_MS,
                "min_fps": self.MIN_FPS
            },
            "camera": {
                "width": self.CAMERA_WIDTH,
                "height": self.CAMERA_HEIGHT,
                "fps": self.CAMERA_FPS
            },
            "lane_detection": {
                "roi_top_ratio": self.ROI_TOP_RATIO,
                "roi_bottom_ratio": self.ROI_BOTTOM_RATIO,
                "blur_kernel_size": self.BLUR_KERNEL_SIZE,
                "canny_low": self.CANNY_LOW_THRESHOLD,
                "canny_high": self.CANNY_HIGH_THRESHOLD,
                "hough_rho": self.HOUGH_RHO,
                "hough_theta": self.HOUGH_THETA,
                "hough_threshold": self.HOUGH_THRESHOLD,
                "hough_min_line_length": self.HOUGH_MIN_LINE_LENGTH,
                "hough_max_line_gap": self.HOUGH_MAX_LINE_GAP,
                "min_lane_slope": self.MIN_LANE_SLOPE,
                "max_lane_slope": self.MAX_LANE_SLOPE
            },
            "obstacle_detection": {
                "safety_zone_width_ratio": self.SAFETY_ZONE_WIDTH_RATIO,
                "safety_zone_height_ratio": self.SAFETY_ZONE_HEIGHT_RATIO,
                "threshold_value": self.OBSTACLE_THRESHOLD_VALUE,
                "min_blob_area_percent": self.OBSTACLE_THRESHOLD_PERCENT
            },
            "control": {
                "base_speed": self.BASE_SPEED,
                "max_speed": self.MAX_SPEED,
                "min_speed": self.MIN_SPEED,
                "max_steer_angle": self.MAX_STEER_ANGLE,
                "steer_smoothing": self.STEER_SMOOTHING,
                "steer_kp": self.STEER_KP
            },
            "data_fusion": {
                "obstacle_priority": self.OBSTACLE_PRIORITY,
                "mode": self.FUSION_MODE
            },
            "remote_override": {
                "enabled": self.OVERRIDE_ENABLED,
                "timeout_ms": self.OVERRIDE_TIMEOUT_MS
            },
            "logging": {
                "enable_telemetry": self.LOG_TELEMETRY,
                "directory": self.LOG_DIRECTORY,
                "frame_interval": self.FRAME_LOG_INTERVAL
            },
            "debug": {
                "enable": self.DEBUG_MODE,
                "show_roi": self.SHOW_ROI,
                "show_lane_lines": self.SHOW_LANE_LINES,
                "show_safety_zone": self.SHOW_SAFETY_ZONE,
                "save_frames": self.SAVE_DEBUG_FRAMES
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(settings, f, indent=2)
        
        print(f"[CONFIG] Configuration exported to {filename}")
    
    def print_summary(self):
        """Print configuration summary"""
        print("\n" + "="*60)
        print("SYSTEM CONFIGURATION SUMMARY")
        print("="*60)
        print(f"\n[PERFORMANCE TARGETS]")
        print(f"  NFR-P1 Latency:  < {self.LATENCY_TARGET_MS}ms")
        print(f"  NFR-P2 FPS:      ≥ {self.MIN_FPS}")
        
        print(f"\n[CAMERA]")
        print(f"  Resolution:      {self.CAMERA_WIDTH}×{self.CAMERA_HEIGHT}")
        print(f"  Target FPS:      {self.CAMERA_FPS}")
        
        print(f"\n[LANE DETECTION - Classical CV]")
        print(f"  Method:          Canny ({self.CANNY_LOW_THRESHOLD}-{self.CANNY_HIGH_THRESHOLD}) + Hough")
        print(f"  ROI:             Top {self.ROI_TOP_RATIO*100:.0f}% - Bottom {self.ROI_BOTTOM_RATIO*100:.0f}%")
        
        print(f"\n[OBSTACLE DETECTION - Blob Detection]")
        print(f"  Safety Zone:     {self.SAFETY_ZONE_WIDTH_RATIO*100:.0f}% width × {self.SAFETY_ZONE_HEIGHT_RATIO*100:.0f}% height")
        print(f"  Threshold:       {self.OBSTACLE_THRESHOLD_PERCENT*100:.0f}% of zone area")
        
        print(f"\n[MOTOR CONTROL]")
        print(f"  Speed Range:     {self.MIN_SPEED}-{self.MAX_SPEED} (Default: {self.BASE_SPEED})")
        print(f"  Steering Range:  ±{self.MAX_STEER_ANGLE}°")
        
        print(f"\n[SAFETY]")
        print(f"  Obstacle Priority: {self.OBSTACLE_PRIORITY}")
        print(f"  Fusion Mode:     {self.FUSION_MODE}")
        print(f"  Override Timeout: {self.OVERRIDE_TIMEOUT_MS}ms (NFR-U1: < 50ms)")
        
        print(f"\n[LOGGING]")
        print(f"  Telemetry:       {'Enabled' if self.LOG_TELEMETRY else 'Disabled'}")
        print(f"  Log Directory:   {self.LOG_DIRECTORY}/")
        print(f"  Debug Mode:      {'Enabled' if self.DEBUG_MODE else 'Disabled'}")
        
        print("="*60 + "\n")


# ============================================================================
# GLOBAL CONFIGURATION INSTANCE
# ============================================================================

# Create global configuration object
# Will automatically load from settings.json if it exists
config = SystemConfig()


# ============================================================================
# BACKWARD COMPATIBILITY - Direct variable access
# ============================================================================
# For existing code that uses "import config; config.CAMERA_WIDTH" style

# NFR Performance Targets
LATENCY_TARGET_MS = config.LATENCY_TARGET_MS
MIN_FPS = config.MIN_FPS

# Camera Configuration
CAMERA_WIDTH = config.CAMERA_WIDTH
CAMERA_HEIGHT = config.CAMERA_HEIGHT
CAMERA_FPS = config.CAMERA_FPS

# Lane Detection
ROI_TOP_RATIO = config.ROI_TOP_RATIO
ROI_BOTTOM_RATIO = config.ROI_BOTTOM_RATIO
BLUR_KERNEL_SIZE = config.BLUR_KERNEL_SIZE
CANNY_LOW_THRESHOLD = config.CANNY_LOW_THRESHOLD
CANNY_HIGH_THRESHOLD = config.CANNY_HIGH_THRESHOLD
HOUGH_RHO = config.HOUGH_RHO
HOUGH_THETA = config.HOUGH_THETA
HOUGH_THRESHOLD = config.HOUGH_THRESHOLD
HOUGH_MIN_LINE_LENGTH = config.HOUGH_MIN_LINE_LENGTH
HOUGH_MAX_LINE_GAP = config.HOUGH_MAX_LINE_GAP
MIN_LANE_SLOPE = config.MIN_LANE_SLOPE
MAX_LANE_SLOPE = config.MAX_LANE_SLOPE

# Obstacle Detection
SAFETY_ZONE_WIDTH_RATIO = config.SAFETY_ZONE_WIDTH_RATIO
OBSTACLE_THRESHOLD_VALUE = config.OBSTACLE_THRESHOLD_VALUE
OBSTACLE_THRESHOLD_PERCENT = config.OBSTACLE_THRESHOLD_PERCENT

# Motor Control
BASE_SPEED = config.BASE_SPEED
MAX_SPEED = config.MAX_SPEED
MIN_SPEED = config.MIN_SPEED
MAX_STEER_ANGLE = config.MAX_STEER_ANGLE
STEER_SMOOTHING = config.STEER_SMOOTHING
STEER_KP = config.STEER_KP

# Data Fusion
OBSTACLE_PRIORITY = config.OBSTACLE_PRIORITY
FUSION_MODE = config.FUSION_MODE

# Remote Override
OVERRIDE_ENABLED = config.OVERRIDE_ENABLED
OVERRIDE_TIMEOUT_MS = config.OVERRIDE_TIMEOUT_MS

# Logging
LOG_TELEMETRY = config.LOG_TELEMETRY
LOG_DIRECTORY = config.LOG_DIRECTORY
FRAME_LOG_INTERVAL = config.FRAME_LOG_INTERVAL

# Debug
DEBUG_MODE = config.DEBUG_MODE
SHOW_ROI = config.SHOW_ROI
SHOW_LANE_LINES = config.SHOW_LANE_LINES
SAVE_DEBUG_FRAMES = config.SAVE_DEBUG_FRAMES


# ============================================================================
# MODULE INITIALIZATION
# ============================================================================

if __name__ == "__main__":
    # If run directly, print configuration summary and export template
    print("\n" + "="*60)
    print("CONFIGURATION MODULE - Standalone Execution")
    print("="*60)
    
    config.print_summary()
    
    # Export template settings.json
    print("\n[ACTION] Exporting template configuration...")
    config.export_to_json("settings_template.json")
    print("[ACTION] ✓ Template saved to settings_template.json")
    print("\n[INFO] Rename to settings.json to use custom configuration\n")
