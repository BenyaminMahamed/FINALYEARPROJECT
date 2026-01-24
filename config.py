# config.py - System Configuration for Autonomous Vehicle
# Aligns with PPRS NFR requirements

# === NFR Performance Targets ===
LATENCY_TARGET_MS = 200  # NFR-P1: End-to-end latency requirement
MIN_FPS = 8              # NFR-P2: Minimum DL inference rate on RPi CPU

# === Camera Configuration ===
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30
PAN_CENTER = 0
TILT_CENTER = 0

# === Lane Detection (Classical CV - OpenCV) ===
# Region of Interest (ROI)
ROI_TOP_RATIO = 0.55     # Start ROI at 55% down from top
ROI_BOTTOM_RATIO = 1.0   # Full bottom

# Preprocessing
BLUR_KERNEL_SIZE = 5     # Gaussian blur to reduce noise

# Canny Edge Detection
CANNY_LOW_THRESHOLD = 50
CANNY_HIGH_THRESHOLD = 150

# Hough Transform Line Detection
HOUGH_RHO = 2            # Distance resolution in pixels
HOUGH_THETA = 1          # Angle resolution in degrees (converted to radians)
HOUGH_THRESHOLD = 50     # Minimum votes for line detection
HOUGH_MIN_LINE_LENGTH = 40
HOUGH_MAX_LINE_GAP = 20

# Lane line filtering
MIN_LANE_SLOPE = 0.3     # Filter out near-horizontal lines
MAX_LANE_SLOPE = 3.0     # Filter out near-vertical lines

# === Object Detection (Deep Learning - YOLO/MobileNet) ===
MODEL_PATH = "models/yolov5n.onnx"  # Nano model for speed
CONFIDENCE_THRESHOLD = 0.5           # Minimum confidence score
NMS_THRESHOLD = 0.4                  # Non-maximum suppression
INPUT_SIZE = (320, 320)              # Model input dimensions (smaller = faster)

# Safety zone for obstacle detection (FR2.2)
SAFETY_ZONE_DISTANCE_CM = 30  # Stop if obstacle within 30cm
SAFETY_ZONE_WIDTH_RATIO = 0.4  # Central 40% of frame width

# === Motor Control Parameters ===
BASE_SPEED = 35          # Default autonomous driving speed
MAX_SPEED = 60           # Safety speed limit
MIN_SPEED = 15           # Minimum speed to maintain motion

# Steering parameters
MAX_STEER_ANGLE = 35     # Maximum steering angle (degrees)
STEER_SMOOTHING = 0.3    # Smoothing factor (0-1, lower = smoother)

# PID-like steering correction (optional enhancement)
STEER_KP = 0.5           # Proportional gain for lane centering

# === Data Fusion Logic (FR1.2) ===
OBSTACLE_PRIORITY = True  # Obstacle detection overrides lane following
FUSION_MODE = "priority"  # Options: "priority", "weighted"

# === Ultrasonic Sensor (Secondary verification) ===
ULTRASONIC_ENABLED = True
ULTRASONIC_TRIGGER_DISTANCE = 25  # cm - matches DL safety zone

# === Remote Override (Flask/Bluetooth) ===
OVERRIDE_ENABLED = True
OVERRIDE_MODE = "flask"   # Options: "flask", "bluetooth"
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
OVERRIDE_TIMEOUT_MS = 50  # NFR-U1: Visual feedback within 50ms

# === Logging & Telemetry ===
LOG_TELEMETRY = True
LOG_DIRECTORY = "logs"
TELEMETRY_LOG_FILE = "telemetry.csv"
FRAME_LOG_INTERVAL = 30   # Log every N frames to reduce I/O

# Telemetry fields to log
TELEMETRY_FIELDS = [
    "timestamp",
    "mode",              # autonomous/manual
    "speed",
    "steering_angle",
    "lane_offset",       # pixels from center
    "obstacle_detected",
    "obstacle_distance",
    "fps",
    "latency_ms"
]

# === Debug Visualization ===
DEBUG_MODE = True         # Show processed frames with overlays
SHOW_ROI = True
SHOW_LANE_LINES = True
SHOW_BOUNDING_BOXES = True
SAVE_DEBUG_FRAMES = False  # Save annotated frames to disk