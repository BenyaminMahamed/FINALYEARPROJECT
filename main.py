#!/usr/bin/env python3
"""
Autonomous Vehicle Testing & Integration - v2.5
Student: Benyamin Mahamed (W1966430)
Project: Autonomous Self-Driving Car for Assisted Mobility

Main integration module implementing complete autonomous navigation system
with safety-first architecture for assisted mobility applications.

Target User: Jonathan (77) - Wheelchair user requiring autonomous navigation
assistance with mandatory manual override and emergency stop capability.

System Architecture (Integrated - Pi 5 Only):
    - Lane Detection (Classical CV: Canny + Hough)
    - Obstacle Detection (Blob Detection)
    - Motor Control (DC motors + steering servo)
    - Remote Override (Manual control takeover — thread-safe)
    - Data Fusion (Priority-based decision logic)

Safety Hierarchy (Highest to Lowest Priority):
    1. Manual Override  (FR3.1)    <- thread-safe via manual_override_flag
    2. Obstacle Detection (FR2.2, NFR-S1)
    3. Autonomous Navigation (FR1.1)
    4. Stopped State

Performance Targets:
    - NFR-P1: Latency < 200ms   (Achieved: ~7.7ms-15.6ms)
    - NFR-P2: FPS >= 8          (Achieved: ~25-40 FPS headless / ~8-12 FPS X11)
    - NFR-S1: Obstacle E-Stop   100% reliable
    - NFR-S2: Manual E-Stop     100% reliable

v2.5 Changes:
    - FIXED: cv2.waitKey(1) was called every frame even when headless,
      blocking the control loop and causing 4-5 FPS on browser SSH.
      Now skipped entirely when no X11 display is available via _poll_keys().
    - FIXED: DISPLAY_FPS lowered to 5, DISPLAY_SCALE to 0.5 — better defaults
      for browser-based SSH clients (e.g. Chromium terminal on Pi).
    - FIXED: VideoWriter tries MJPG before XVID for Pi 5 compatibility.
    - ADDED: Real-time control loop FPS printed every 60 frames in terminal
      so you can confirm the loop is running free of display overhead.
    - ADDED: Camera frame rate locked to 30 FPS via FrameDurationLimits to
      prevent Picamera2 over-capturing and filling the pipeline buffer.
    - ADDED: KeyboardInterrupt handling in run_integration_test finally block.
    - ADDED: Mode change events logged to JSONL for viva evidence.
    - All v2.4 features retained: dual-mode output, throttled X11, video
      recording, decoupled control loop, graceful X11 drop fallback.
    - Steering fix from v2.3 retained: raw_steering passed through directly,
      gain/direction owned by STEER_KP + STEER_SMOOTHING in config.py.
    - MANUAL mode label renders in MAGENTA (255, 0, 255) for Priority 1 evidence.
"""

from control_logic import RobotMuscle
from lane_detection import LaneDetector
from object_detection import ObstacleDetector
from remote_override import RemoteOverride, manual_override_flag
import cv2
import time
import config
import sys
import numpy as np
import os
from datetime import datetime
from typing import Tuple, Optional, Dict, Any
import csv
import json


# ============================================================================
# DISPLAY CONFIGURATION — tune these without touching any other code
# ============================================================================

# X11 refresh rate — kept low so rendering never stalls the control loop.
# The control loop and video recording ALWAYS run at full camera speed.
# At 5 FPS the window updates every 200ms which is fine for monitoring.
DISPLAY_FPS   = 5     # Hz — X11 window refresh rate

# Scale the X11 window. 0.5 = 320x240 over network instead of 640x480.
# Recommended for browser-based SSH — halves bandwidth, doubles perceived speed.
DISPLAY_SCALE = 0.5

# Always write annotated frames to test_logs/*.avi — works with no monitor.
RECORD_VIDEO  = True


# ============================================================================
# MODE COLOUR MAP  (single source of truth used by all display functions)
# ============================================================================

MODE_COLOURS = {
    "MANUAL":        (255,   0, 255),   # Magenta  — Priority 1 (FR3.1)
    "OBSTACLE STOP": (  0,   0, 255),   # Red      — Priority 2 (NFR-S1)
    "AUTONOMOUS":    (  0, 255,   0),   # Green    — Priority 3 (FR1.1)
    "STOPPED":       (128, 128, 128),   # Grey     — Default idle
}


# ============================================================================
# PERFORMANCE LOGGER
# ============================================================================

class PerformanceLogger:
    """
    Comprehensive data logging system for performance validation.

    Addresses IPD feedback: "some features incomplete or unclear"
    Provides evidence for all NFR requirements through detailed logging.

    Logs:
        - Frame-by-frame metrics (CSV)
        - Session summaries (JSON)
        - Event logs (JSONL)
    """

    def __init__(self, log_dir: str = "test_logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        self.session_id   = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.metrics_file = os.path.join(log_dir, f"metrics_{self.session_id}.csv")
        self._init_metrics_log()

        self.summary_file = os.path.join(log_dir, f"summary_{self.session_id}.json")
        self.session_data = {
            'session_id':           self.session_id,
            'start_time':           time.time(),
            'start_time_readable':  datetime.now().isoformat(),
            'frames_processed':     0,
            'emergency_stops':      0,
            'mode_changes':         0,
            'total_latency_ms':     0,
            'errors':               [],
            'test_mode':            None
        }

        print(f"[LOGGER] Session {self.session_id} — Logging to {log_dir}/")

    def _init_metrics_log(self):
        with open(self.metrics_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'frame_id', 'fps', 'latency_ms',
                'steering_angle', 'lane_offset', 'confidence',
                'obstacle_detected', 'distance_estimate', 'mode', 'speed'
            ])

    def log_frame(self, frame_id: int, metrics: Dict[str, Any]):
        with open(self.metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                time.time(),
                frame_id,
                metrics.get('fps', 0),
                metrics.get('latency_ms', 0),
                metrics.get('steering_angle', 0),
                metrics.get('lane_offset', 0),
                metrics.get('confidence', 0),
                metrics.get('obstacle_detected', False),
                metrics.get('distance_estimate', 0),
                metrics.get('mode', 'UNKNOWN'),
                metrics.get('speed', 0)
            ])
        self.session_data['frames_processed'] += 1
        self.session_data['total_latency_ms'] += metrics.get('latency_ms', 0)

    def log_event(self, event_type: str, details: str):
        event = {
            'timestamp':     time.time(),
            'time_readable': datetime.now().isoformat(),
            'type':          event_type,
            'details':       details
        }
        if event_type == 'emergency_stop':
            self.session_data['emergency_stops'] += 1
        elif event_type == 'mode_change':
            self.session_data['mode_changes'] += 1
        elif event_type == 'error':
            self.session_data['errors'].append(details)

        events_file = os.path.join(self.log_dir, f"events_{self.session_id}.jsonl")
        with open(events_file, 'a') as f:
            f.write(json.dumps(event) + '\n')

        print(f"[EVENT] {event_type.upper()}: {details}")

    def set_test_mode(self, mode: str):
        self.session_data['test_mode'] = mode

    def save_summary(self):
        self.session_data['end_time']          = time.time()
        self.session_data['end_time_readable'] = datetime.now().isoformat()
        self.session_data['duration_seconds']  = (
            self.session_data['end_time'] - self.session_data['start_time']
        )
        if self.session_data['frames_processed'] > 0:
            self.session_data['avg_latency_ms'] = (
                self.session_data['total_latency_ms'] /
                self.session_data['frames_processed']
            )
        else:
            self.session_data['avg_latency_ms'] = 0

        with open(self.summary_file, 'w') as f:
            json.dump(self.session_data, f, indent=2)

        print(f"\n[LOGGER] Session summary saved to {self.summary_file}")
        print(f"[LOGGER] Frames processed: {self.session_data['frames_processed']}")
        print(f"[LOGGER] Average latency:  {self.session_data['avg_latency_ms']:.2f}ms")


# ============================================================================
# VIDEO WRITER HELPER
# ============================================================================

def make_video_writer(log_dir: str, tag: str,
                      width: int, height: int) -> Optional[cv2.VideoWriter]:
    """
    Create a VideoWriter saving to test_logs/<tag>_<timestamp>.avi
    Tries MJPG first (better Pi 5 compatibility), falls back to XVID.
    Returns None if both fail — non-fatal, run continues without recording.
    """
    os.makedirs(log_dir, exist_ok=True)
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = os.path.join(log_dir, f"{tag}_{timestamp}.avi")

    for codec in ('MJPG', 'XVID'):
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(video_path, fourcc, 15, (width, height))
        if writer.isOpened():
            print(f"[RECORD] Recording ({codec}) -> {video_path}")
            return writer
        writer.release()

    print(f"[RECORD] WARNING — VideoWriter could not open {video_path}")
    return None


# ============================================================================
# CAMERA CLASS - Picamera2 Integration
# ============================================================================

class Camera:
    """
    Camera wrapper using Picamera2 for Raspberry Pi 5.
    Provides robust initialisation and error handling (FR4.2).

    v2.5: FrameDurationLimits locks capture to 30 FPS so Picamera2 does not
    over-capture and fill the pipeline buffer faster than it can drain.
    """

    def __init__(self, width: int = 640, height: int = 480):
        self.width  = width
        self.height = height
        self.camera = None

        try:
            from picamera2 import Picamera2
            print(f"[CAMERA] Initialising Picamera2 ({width}x{height})...")

            self.camera = Picamera2()
            camera_config = self.camera.create_preview_configuration(
                main={"size": (width, height), "format": "RGB888"}
            )
            self.camera.configure(camera_config)
            self.camera.start()

            # Lock to 30 FPS — 33333 microseconds per frame
            try:
                self.camera.set_controls(
                    {"FrameDurationLimits": (33333, 33333)}
                )
                print("[CAMERA] Frame rate locked to 30 FPS")
            except Exception:
                pass  # Non-fatal — older picamera2 may not support this

            time.sleep(2)
            print("[CAMERA] SUCCESS — Picamera2 ready")

        except Exception as e:
            print(f"[CAMERA] Failed to initialise: {e}")
            import traceback
            traceback.print_exc()
            self.camera = None

    def is_opened(self) -> bool:
        return self.camera is not None

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self.camera is None:
            return False, None
        try:
            frame = self.camera.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            return True, frame
        except Exception as e:
            print(f"[CAMERA] Read error: {e}")
            return False, None

    def release(self):
        if self.camera:
            try:
                self.camera.stop()
                self.camera.close()
                print("[CAMERA] Released")
            except Exception:
                pass


# ============================================================================
# AUTONOMOUS VEHICLE CLASS - Main Integration
# ============================================================================

class AutonomousVehicle:
    """
    Main system integrating vision, obstacle detection, and control.
    Shared hardware architecture to prevent GPIO pin conflicts.

    v2.5 key changes:
        - cv2.waitKey() SKIPPED entirely when no X11 — primary fix for
          4-5 FPS on browser SSH. Control loop now runs at full camera speed.
        - X11 throttled to DISPLAY_FPS=5 Hz, scaled to DISPLAY_SCALE=0.5.
        - Real-time loop FPS printed in terminal every 60 frames.
        - VideoWriter tries MJPG before XVID for Pi 5 compatibility.
        - Camera locked to 30 FPS via FrameDurationLimits.
        - Mode change events logged to JSONL for viva evidence.
    """

    def __init__(self, simulation_mode: bool = False, enable_logging: bool = True):
        self.simulation_mode = simulation_mode
        self.enable_logging  = enable_logging

        # ------------------------------------------------------------------
        # 1. SHARED HARDWARE
        # ------------------------------------------------------------------
        self.shared_px = None
        if not self.simulation_mode:
            try:
                from picarx import Picarx
                self.shared_px = Picarx()
                print("[INIT] ✓ Shared hardware interface (Picarx) ready")
            except Exception as e:
                print(f"[INIT] ✗ Hardware initialisation failed: {e}")
                self.shared_px = None

        # ------------------------------------------------------------------
        # 2. SUBSYSTEMS
        # ------------------------------------------------------------------
        print("[INIT] Initialising subsystems...")
        self.lane_detector     = LaneDetector()
        self.obstacle_detector = ObstacleDetector()
        self.remote_override   = RemoteOverride(picarx_instance=self.shared_px)

        # ------------------------------------------------------------------
        # 3. LOGGER
        # ------------------------------------------------------------------
        if self.enable_logging:
            self.logger = PerformanceLogger()
            self.lane_detector.set_event_callback(self.logger.log_event)
            self.obstacle_detector.set_event_callback(self.logger.log_event)
            self.remote_override.set_event_callback(self.logger.log_event)
        else:
            self.logger = None

        # ------------------------------------------------------------------
        # 4. MOTOR CONTROL
        # ------------------------------------------------------------------
        if not simulation_mode:
            self.motor_control = RobotMuscle(picarx_instance=self.shared_px)
            if self.logger:
                self.motor_control.set_event_callback(self.logger.log_event)
            print("[INIT] ✓ Full system initialised — MOTORS ACTIVE")
        else:
            self.motor_control = None
            print("[INIT] ✓ Simulation mode — motors disabled")

        # ------------------------------------------------------------------
        # 5. BACKGROUND OVERRIDE LISTENER
        #    Started explicitly after all input() prompts in each run_* method.
        # ------------------------------------------------------------------
        self._listener_started = False
        self.running           = False
        self.autonomous_active = False

    # -----------------------------------------------------------------------
    # Terminal helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _restore_terminal():
        """Restore cooked terminal mode — guards against raw-mode stdin."""
        try:
            import termios
            fd = sys.stdin.fileno()
            attrs = termios.tcgetattr(fd)
            attrs[3] |= termios.ECHO | termios.ICANON
            termios.tcsetattr(fd, termios.TCSADRAIN, attrs)
        except Exception:
            pass

    def _start_listener_once(self):
        """Start the background override listener exactly once per session."""
        if not self._listener_started:
            self.remote_override.start_listener()
            self._listener_started = True

    # -----------------------------------------------------------------------
    # Display / key helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _x11_available() -> bool:
        return bool(os.environ.get('DISPLAY'))

    @staticmethod
    def _scale_frame(frame: np.ndarray, scale: float) -> np.ndarray:
        """Downscale for X11 — reduces SSH bandwidth."""
        if scale == 1.0:
            return frame
        h, w = frame.shape[:2]
        return cv2.resize(frame, (int(w * scale), int(h * scale)),
                          interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def _poll_keys(has_display: bool) -> int:
        """
        Poll cv2 for keypresses ONLY when X11 is active.

        v2.5 FIX: cv2.waitKey(1) triggers X11 event processing which blocks
        on a network round-trip to the display server on every single call.
        Over browser SSH this adds ~50-200ms per frame, collapsing FPS to 4-5.
        Skipping it entirely when headless recovers the full control loop rate.

        Returns keycode on X11, or 0xFF (no-op) when headless.
        """
        if has_display:
            return cv2.waitKey(1) & 0xFF
        return 0xFF

    # -----------------------------------------------------------------------
    # Camera helpers
    # -----------------------------------------------------------------------

    def _open_camera(self, max_retries: int = 3) -> Tuple[Optional[Camera], bool]:
        print("\n[CAMERA] Opening camera...")
        for attempt in range(max_retries):
            try:
                cap = Camera(config.CAMERA_WIDTH, config.CAMERA_HEIGHT)
                if cap.is_opened():
                    ret, test_frame = cap.read()
                    if ret and test_frame is not None:
                        print(f"[CAMERA] ✓ Test frame OK: {test_frame.shape}")
                        return cap, True
                    else:
                        print(f"[CAMERA] ✗ Test frame failed "
                              f"(attempt {attempt + 1}/{max_retries})")
                        cap.release()
                if attempt < max_retries - 1:
                    print("[CAMERA] Retrying in 2 seconds...")
                    time.sleep(2)
            except Exception as e:
                print(f"[CAMERA] Error on attempt {attempt + 1}: {e}")
                if self.logger:
                    self.logger.log_event('error', f"Camera init failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)

        print("[CAMERA] ✗ Failed after all retries")
        return None, False

    # -----------------------------------------------------------------------
    # Test Modes
    # -----------------------------------------------------------------------

    def run_heartbeat_test(self):
        """Hardware validation — servos and motors (FR1.2)."""
        if self.simulation_mode:
            print("[SKIP] Heartbeat test (simulation mode)")
            return

        if self.logger:
            self.logger.set_test_mode("heartbeat")

        print("\n" + "="*60)
        print("HEARTBEAT TEST — Hardware Validation")
        print("="*60)

        try:
            print("\n[1/2] Testing camera servos...")
            self.motor_control.test_servos()
            if self.logger:
                self.logger.log_event('hardware_test', 'Servo test completed')

            self._restore_terminal()
            input("Press ENTER to test motors (ensure clear space)...")
            print("\n[2/2] Testing motors...")
            self.motor_control.test_motors()
            if self.logger:
                self.logger.log_event('hardware_test', 'Motor test completed')

            print("\n✓ Heartbeat test complete — hardware operational")

        except Exception as e:
            print(f"\n✗ Heartbeat test failed: {e}")
            if self.logger:
                self.logger.log_event('error', f"Heartbeat test failed: {e}")
            self.motor_control.emergency_stop()

        finally:
            try:
                self.motor_control.print_statistics()
                if self.logger:
                    self.logger.save_summary()
            except Exception as e:
                print(f"[SUMMARY ERROR] Could not render final stats: {e}")

    # -----------------------------------------------------------------------

    def run_vision_test(self):
        """
        Vision-only testing — Lane Detection Algorithm Validation.
        Validates FR1.1, NFR-P1 (< 200ms), NFR-P2 (>= 8 FPS).

        v2.5: waitKey skipped when headless — control loop runs free.
              VideoWriter records full-speed annotated frames to file.
              FPS reported every 60 frames in terminal.
        """
        if self.logger:
            self.logger.set_test_mode("vision_test")

        print("\n" + "="*60)
        print("VISION TEST — Lane Detection (Classical CV)")
        print("="*60)
        print("\nValidates: FR1.1 | NFR-P1 (latency) | NFR-P2 (FPS)")
        print("Controls:  'q' Quit | 's' Save frame | 'p' Pause/Resume")
        print("           (OpenCV window must have focus for key controls)")

        has_display = self._x11_available()
        if has_display:
            print(f"[DISPLAY] X11 active — throttled to {DISPLAY_FPS} FPS, "
                  f"scale={DISPLAY_SCALE}x")
        else:
            print("[DISPLAY] Headless — recording to file, Ctrl+C to stop")

        cap, success = self._open_camera()
        if not success:
            if self.logger:
                self.logger.save_summary()
            return

        # Camera warmup
        print("\n[CAMERA] Warming up...")
        warmup_success = False
        for i in range(10):
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"  Warmup frame {i+1}: OK — Shape: {frame.shape}")
                warmup_success = True
                if i >= 3:
                    break
            else:
                print(f"  Warmup frame {i+1}: Failed")
            time.sleep(0.1)

        if not warmup_success:
            print("[ERROR] Camera warmup failed")
            if self.logger:
                self.logger.log_event('error', 'Camera warmup failed')
                self.logger.save_summary()
            cap.release()
            return

        # Video recording
        video_out = None
        if RECORD_VIDEO:
            video_out = make_video_writer(
                "test_logs", "vision",
                config.CAMERA_WIDTH, config.CAMERA_HEIGHT
            )

        print("[CAMERA] Ready! Starting detection...\n")

        frame_count     = 0
        fps_start       = time.time()
        latencies       = []
        steering_angles = []
        confidences     = []
        paused          = False
        fps             = 0.0
        display_frame   = None

        display_interval = 1.0 / DISPLAY_FPS
        last_display_t   = 0.0

        try:
            while True:
                if not paused:
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        print("[WARN] Failed to grab frame, retrying...")
                        time.sleep(0.05)
                        continue

                    frame_count += 1
                    start_time   = time.time()
                    result       = self.lane_detector.process_frame(frame)
                    latency_ms   = (time.time() - start_time) * 1000

                    latencies.append(latency_ms)
                    steering_angles.append(result['steering_angle'])
                    confidences.append(result['confidence'])

                    elapsed = time.time() - fps_start
                    fps     = frame_count / elapsed if elapsed > 0 else 0.0

                    if self.logger:
                        self.logger.log_frame(frame_count, {
                            'fps':               fps,
                            'latency_ms':        latency_ms,
                            'steering_angle':    result['steering_angle'],
                            'lane_offset':       result['lane_offset'],
                            'confidence':        result['confidence'],
                            'obstacle_detected': False,
                            'distance_estimate': 0,
                            'mode':              'VISION_TEST',
                            'speed':             0
                        })

                    # Annotated frame
                    display_frame = (result['debug_frame']
                                     if result['debug_frame'] is not None
                                     else frame.copy())
                    self._draw_vision_metrics(display_frame, latency_ms, fps, result)

                    # Video — full speed, no throttle
                    if video_out is not None:
                        video_out.write(display_frame)

                    # X11 — throttled
                    now = time.time()
                    if has_display and (now - last_display_t) >= display_interval:
                        try:
                            scaled = self._scale_frame(display_frame, DISPLAY_SCALE)
                            cv2.imshow('Vision Test — Lane Detection', scaled)
                            last_display_t = now
                        except Exception:
                            has_display = False  # X11 dropped, continue headless

                    # Terminal progress every 60 frames
                    if frame_count % 60 == 0:
                        avg_lat = np.mean(latencies[-60:]) if latencies else 0.0
                        print(f"[{frame_count:5d}] FPS: {fps:5.1f} | "
                              f"Latency: {avg_lat:5.1f}ms | "
                              f"Steer: {result['steering_angle']:+4d}deg | "
                              f"Conf: {result['confidence']:.2f}")

                # KEY POLL — only when X11 active (v2.5 core fix)
                key = self._poll_keys(has_display)
                if key == ord('q'):
                    break
                elif key == ord('s') and display_frame is not None:
                    fname = f"lane_test_{frame_count}.jpg"
                    cv2.imwrite(fname, display_frame)
                    print(f"[SAVED] {fname}")
                elif key == ord('p'):
                    paused = not paused
                    print(f"[{'PAUSED' if paused else 'RESUMED'}]")

        except KeyboardInterrupt:
            print("\n[INTERRUPT] Vision test stopped by user")
        except Exception as e:
            print(f"\n[ERROR] Vision test exception: {e}")
            if self.logger:
                self.logger.log_event('error', f"Vision test exception: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if video_out is not None:
                video_out.release()
                print("[RECORD] Vision test video saved.")
            cap.release()
            cv2.destroyAllWindows()

            print("\n" + "="*60)
            print("POST-TEST REPORT: VISION TEST")
            print("="*60)
            try:
                if latencies:
                    self._print_vision_summary(
                        latencies, steering_angles, confidences, fps
                    )
                else:
                    print("\n[INFO] No frames processed — cannot generate summary")
                self.lane_detector.print_statistics()
                if self.logger:
                    self.logger.save_summary()
            except Exception as e:
                print(f"[SUMMARY ERROR] Could not render final stats: {e}")

    # -----------------------------------------------------------------------

    def run_integration_test(self):
        """
        Full Integration Test with ALL safety features.

        v2.5 performance behaviour:
            - waitKey() SKIPPED when headless — was collapsing FPS to 4-5.
              Control loop now runs at full camera capture rate.
            - VideoWriter records every control-loop frame to test_logs/.
              Works with no monitor — safe for viva environments.
            - X11 window throttled to DISPLAY_FPS=5 Hz, scale=0.5.
            - Real-time FPS + mode printed in terminal every 60 frames.

        Override behaviour (unchanged from v2.3):
            - 'o' in TERMINAL = PRIMARY override toggle (background thread).
            - 'o' in OpenCV window = SECONDARY fallback.
            - manual_override_flag = single source of truth (FR3.1).
            - MANUAL renders MAGENTA for Priority 1 validation evidence.
        """
        if self.logger:
            self.logger.set_test_mode(
                "integration_test" +
                ("_simulation" if self.simulation_mode else "_live")
            )

        print("\n" + "="*60)
        print(f"FULL INTEGRATION TEST — "
              f"{'SIMULATION' if self.simulation_mode else 'LIVE MOTORS'}")
        print("="*60)

        if not self.simulation_mode:
            print("\n WARNING: Motors will move!")
            print("  - Vehicle on track with clear lane markings")
            print("  - Clear path ahead")
            print("  - Emergency stop accessible (ESC / Ctrl+C)")
            self._restore_terminal()

        # Start background override listener — after all input() calls
        self._start_listener_once()

        has_display = self._x11_available()
        if has_display:
            print(f"\n[DISPLAY] X11 active — throttled to {DISPLAY_FPS} FPS, "
                  f"scale={DISPLAY_SCALE}x")
            print("[DISPLAY] OpenCV window must have focus for SPACE/WASD/ESC/q")
        else:
            print("\n[DISPLAY] Headless — recording to file only")
            print("[DISPLAY] Ctrl+C in terminal to stop")

        print("\nControls:")
        print("  SPACE  — Start / Stop autonomous mode")
        print("  'o'    — Toggle MANUAL override (terminal, no X11 focus needed)")
        print("  WASD   — Manual drive (override active, OpenCV window focus)")
        print("  ESC    — Emergency stop")
        print("  'q'    — Quit")

        cap, success = self._open_camera()
        if not success:
            if self.logger:
                self.logger.save_summary()
            return

        print("\n[CAMERA] Warming up...")
        for _ in range(5):
            cap.read()
        time.sleep(0.5)
        print("[CAMERA] Ready!\n")

        # Video recording
        video_out = None
        if RECORD_VIDEO:
            video_out = make_video_writer(
                "test_logs", "integration",
                config.CAMERA_WIDTH, config.CAMERA_HEIGHT
            )

        self.running           = True
        self.autonomous_active = False
        frame_count            = 0
        fps_start              = time.time()
        fps                    = 0.0
        last_mode              = None

        display_interval = 1.0 / DISPLAY_FPS
        last_display_t   = 0.0

        try:
            while self.running:
                ret, frame = cap.read()
                if not ret or frame is None:
                    print("[WARN] Frame grab failed, retrying...")
                    time.sleep(0.05)
                    continue

                frame_count += 1
                start_time   = time.time()

                # ── Vision pipeline ──────────────────────────────────────────
                lane_result     = self.lane_detector.process_frame(frame)
                obstacle_result = self.obstacle_detector.detect_obstacle(frame)
                latency_ms      = (time.time() - start_time) * 1000

                # ── DECISION LOGIC — Priority-based data fusion ──────────────

                if manual_override_flag.is_set():
                    # PRIORITY 1: MANUAL OVERRIDE (FR3.1)
                    mode  = "MANUAL"
                    speed = 0

                elif obstacle_result['obstacle_detected']:
                    # PRIORITY 2: OBSTACLE STOP (NFR-S1)
                    mode  = "OBSTACLE STOP"
                    speed = 0
                    if not self.simulation_mode:
                        self.motor_control.emergency_stop()
                    if frame_count % 10 == 0:
                        print(f"[SAFETY] STOP — Obstacle at "
                              f"{obstacle_result['distance_estimate']}cm")

                elif self.autonomous_active:
                    # PRIORITY 3: AUTONOMOUS LANE FOLLOWING (FR1.1)
                    mode = "AUTONOMOUS"

                    # Pass steering angle directly — direction and gain
                    # controlled by STEER_KP + STEER_SMOOTHING in config.py
                    raw_steering   = lane_result.get('steering_angle', 0)
                    steering_angle = raw_steering

                    if lane_result['confidence'] > 0.5:
                        speed = config.BASE_SPEED
                    elif lane_result['confidence'] > 0.2:
                        speed = config.MIN_SPEED
                    else:
                        speed = 0   # Lost lane — safety stop

                    if not self.simulation_mode:
                        self.motor_control.px.backward(speed)
                        self.motor_control.px.set_dir_servo_angle(steering_angle)
                        if frame_count % 5 == 0:
                            print(f"[AUTO] Steer: {steering_angle:.1f}° | "
                                  f"Speed: {speed} | "
                                  f"Conf: {lane_result['confidence']:.2f}")
                else:
                    # DEFAULT: STOPPED
                    mode  = "STOPPED"
                    speed = 0
                    if not self.simulation_mode:
                        self.motor_control.emergency_stop()

                # Log mode changes for viva evidence
                if mode != last_mode:
                    if self.logger:
                        self.logger.log_event(
                            'mode_change', f"{last_mode} -> {mode}"
                        )
                    last_mode = mode

                # ── Performance logging ──────────────────────────────────────
                elapsed = time.time() - fps_start
                fps     = frame_count / elapsed if elapsed > 0 else 0.0

                if self.logger:
                    self.logger.log_frame(frame_count, {
                        'fps':               fps,
                        'latency_ms':        latency_ms,
                        'steering_angle':    lane_result['steering_angle'],
                        'lane_offset':       lane_result['lane_offset'],
                        'confidence':        lane_result['confidence'],
                        'obstacle_detected': obstacle_result['obstacle_detected'],
                        'distance_estimate': obstacle_result['distance_estimate'],
                        'mode':              mode,
                        'speed':             speed
                    })

                # Terminal FPS report every 60 frames
                if frame_count % 60 == 0:
                    print(f"[{frame_count:5d}] FPS: {fps:5.1f} | "
                          f"Latency: {latency_ms:5.1f}ms | Mode: {mode}")

                # ── Annotated display frame ──────────────────────────────────
                display_frame = self._create_full_display(
                    frame, lane_result, obstacle_result, mode, fps, latency_ms
                )

                # Video — full control-loop speed, always
                if video_out is not None:
                    video_out.write(display_frame)

                # X11 — throttled, non-blocking
                now = time.time()
                if has_display and (now - last_display_t) >= display_interval:
                    try:
                        scaled = self._scale_frame(display_frame, DISPLAY_SCALE)
                        cv2.imshow('Integration Test', scaled)
                        last_display_t = now
                    except Exception:
                        has_display = False  # X11 dropped — continue headless

                # ── KEY POLL — only when X11 active (v2.5 core fix) ─────────
                key = self._poll_keys(has_display)

                if key == ord('q'):
                    print("\n[QUIT] Stopping integration test...")
                    break

                elif key == ord(' '):
                    if not manual_override_flag.is_set():
                        self.autonomous_active = not self.autonomous_active
                        status = "STARTED" if self.autonomous_active else "STOPPED"
                        print(f"\n[AUTO] Autonomous mode {status}")
                    else:
                        print("[WARN] Deactivate manual override first ('o')")

                elif key == ord('o'):
                    if manual_override_flag.is_set():
                        self.remote_override.deactivate_override()
                        print("[OVERRIDE] Deactivated via OpenCV key")
                    else:
                        self.remote_override.activate_override()
                        self.autonomous_active = False
                        print("[OVERRIDE] Activated via OpenCV key")

                elif key == 27:   # ESC — emergency stop
                    self.autonomous_active = False
                    self.remote_override.emergency_stop()
                    if not self.simulation_mode:
                        self.motor_control.emergency_stop()
                    print("[SAFETY] Emergency stop — all systems halted")

                elif manual_override_flag.is_set():
                    if key == ord('w'):
                        self.remote_override.process_manual_command('forward')
                    elif key == ord('s'):
                        self.remote_override.process_manual_command('backward')
                    elif key == ord('a'):
                        self.remote_override.process_manual_command('left')
                    elif key == ord('d'):
                        self.remote_override.process_manual_command('right')

        except KeyboardInterrupt:
            print("\n[INTERRUPT] Integration test stopped by user")
        except Exception as e:
            print(f"\n[ERROR] Integration test exception: {e}")
            import traceback
            traceback.print_exc()

        finally:
            self.running = False
            if not self.simulation_mode and self.motor_control:
                self.motor_control.emergency_stop()
            self.remote_override.emergency_stop()
            if video_out is not None:
                video_out.release()
                print("[RECORD] Integration test video saved.")
            cap.release()
            cv2.destroyAllWindows()

            print("\n" + "="*60)
            print("POST-TEST REPORT: INTEGRATION SESSION")
            print("="*60)
            try:
                self.lane_detector.print_statistics()
                self.obstacle_detector.print_statistics()
                self.remote_override.print_statistics()
                if self.logger:
                    self.logger.save_summary()
            except Exception as e:
                print(f"[SUMMARY ERROR] Could not render final stats: {e}")

            print("\n[INFO] Returning to Main Menu...")

    # -----------------------------------------------------------------------
    # Display helpers
    # -----------------------------------------------------------------------

    def _create_full_display(self, frame: np.ndarray,
                             lane_result: Dict, obstacle_result: Dict,
                             mode: str, fps: float,
                             latency_ms: float) -> np.ndarray:
        """
        Build annotated display frame written to video and shown on X11.
        Mode colour from MODE_COLOURS — MANUAL always MAGENTA (FR3.1 evidence).
        """
        if lane_result['debug_frame'] is not None:
            display = lane_result['debug_frame'].copy()
        else:
            display = frame.copy()

        if obstacle_result['debug_frame'] is not None:
            mask = np.all(obstacle_result['debug_frame'] == frame, axis=2)
            display[~mask] = obstacle_result['debug_frame'][~mask]

        height, width = display.shape[:2]

        # Semi-transparent info panel
        overlay = display.copy()
        cv2.rectangle(overlay, (10, 10), (450, 140), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)

        y = 35
        for text in [
            f"Mode: {mode}",
            f"FPS: {fps:.1f}  |  Latency: {latency_ms:.1f}ms",
            f"Lane Confidence: {lane_result['confidence']:.2f}",
            f"Obstacle: {'YES' if obstacle_result['obstacle_detected'] else 'NO'}"
        ]:
            cv2.putText(display, text, (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y += 25

        mode_colour = MODE_COLOURS.get(mode, (255, 255, 255))
        cv2.putText(display, mode, (10, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, mode_colour, 3)

        return display

    def _draw_vision_metrics(self, frame: np.ndarray, latency_ms: float,
                             fps: float, result: Dict):
        """Draw metrics overlay on vision test frames."""
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 160), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        y = 35
        latency_colour = (
            (0, 255, 0) if latency_ms < config.LATENCY_TARGET_MS
            else (0, 0, 255)
        )
        cv2.putText(frame, f"Latency: {latency_ms:.1f}ms", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, latency_colour, 2)
        y += 25

        for text in [
            f"FPS: {fps:.1f}",
            f"Steering: {result['steering_angle']:+4d} deg",
            f"Offset:   {result['lane_offset']:+4d} px",
            f"Confidence: {result['confidence']:.2f}"
        ]:
            cv2.putText(frame, text, (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y += 25

    def _print_vision_summary(self, latencies: list, steering_angles: list,
                              confidences: list, fps: float):
        """Print comprehensive vision test summary against NFR targets."""
        print("\n" + "="*60)
        print("LANE DETECTION TEST SUMMARY")
        print("="*60)

        if not latencies:
            print("No data collected")
            return

        avg_latency  = np.mean(latencies)
        max_latency  = np.max(latencies)
        min_latency  = np.min(latencies)
        latency_pass = avg_latency < config.LATENCY_TARGET_MS

        print(f"\n[1] LATENCY (NFR-P1: < {config.LATENCY_TARGET_MS}ms)")
        print(f"    Average: {avg_latency:6.2f}ms  "
              f"{'PASS' if latency_pass else 'FAIL'}")
        print(f"    Min:     {min_latency:6.2f}ms")
        print(f"    Max:     {max_latency:6.2f}ms")
        print(f"    Std Dev: {np.std(latencies):6.2f}ms")
        if latency_pass:
            print(f"    Performance: "
                  f"{config.LATENCY_TARGET_MS / avg_latency:.1f}x better than target")

        fps_pass = fps >= config.MIN_FPS
        print(f"\n[2] FRAME RATE (NFR-P2: >= {config.MIN_FPS} FPS)")
        print(f"    FPS: {fps:.1f}  {'PASS' if fps_pass else 'FAIL'}")
        if fps_pass:
            print(f"    Performance: {fps / config.MIN_FPS:.1f}x better than target")

        avg_steering        = np.mean(np.abs(steering_angles))
        steering_smoothness = np.sum(np.abs(np.diff(steering_angles)))
        print(f"\n[3] STEERING BEHAVIOUR")
        print(f"    Avg magnitude:  {avg_steering:.1f} deg")
        print(f"    Total changes:  {steering_smoothness:.1f} deg (lower = smoother)")
        print(f"    Max:            {np.max(np.abs(steering_angles)):.1f} deg")

        avg_confidence  = np.mean(confidences)
        low_conf_frames = int(np.sum(np.array(confidences) < 0.5))
        low_conf_pct    = (low_conf_frames / len(confidences)) * 100
        print(f"\n[4] DETECTION RELIABILITY")
        print(f"    Avg confidence:        {avg_confidence:.2f}")
        print(f"    Low confidence (<0.5): "
              f"{low_conf_frames} frames ({low_conf_pct:.1f}%)")

        passed = sum([latency_pass, fps_pass, avg_confidence > 0.5])
        print(f"\n[5] OVERALL: {passed}/3 tests passed")
        if passed == 3:
            print("    Status: READY FOR INTEGRATION")
        elif passed >= 2:
            print("    Status: NEEDS TUNING")
        else:
            print("    Status: REQUIRES OPTIMISATION")

        print("="*60 + "\n")


# ============================================================================
# MAIN MENU & ENTRY POINT
# ============================================================================

def print_menu():
    print("\n" + "="*60)
    print("AUTONOMOUS VEHICLE — TESTING SUITE v2.5")
    print("Student: Benyamin Mahamed (W1966430)")
    print("Target:  Jonathan (77) — Assisted Mobility Platform")
    print("="*60)
    print("\n1. Heartbeat Test        (Hardware validation)")
    print("2. Vision Test           (Lane detection only)")
    print("3. Integration Test      — SIMULATION (all systems)")
    print("4. Integration Test      — LIVE MOTORS (full autonomous)")
    print("5. Exit")
    print()


def main():
    sys.stdout.flush()

    print("\n" + "="*60)
    print("AUTONOMOUS VEHICLE TESTING SYSTEM v2.5")
    print("="*60)
    print("\nStudent: Benyamin Mahamed (W1966430)")
    print("Project: Autonomous Self-Driving Car for Assisted Mobility")
    print("Target:  Jonathan (77) — Wheelchair User")
    print("="*60)

    if os.environ.get('DISPLAY'):
        print(f"\n[DISPLAY] X11 detected — display at {DISPLAY_FPS} FPS, "
              f"scale={DISPLAY_SCALE}x — control loop runs free")
        print("[DISPLAY] OpenCV window must have focus for key controls")
    else:
        print("\n[DISPLAY] Headless — all output recorded to test_logs/")
        print("[DISPLAY] Ctrl+C to stop any test")
    print()

    vehicle_sim  = None
    vehicle_live = None

    while True:
        AutonomousVehicle._restore_terminal()

        if sys.stdin.isatty():
            try:
                import termios
                termios.tcflush(sys.stdin, termios.TCIFLUSH)
            except ImportError:
                pass

        print_menu()
        choice = input("Select test (1-5): ").strip()

        if not choice:
            continue

        if choice == '1':
            print("\n[INFO] Initialising hardware interface...")
            if vehicle_sim is not None:
                vehicle_sim = None
                time.sleep(0.2)
            if vehicle_live is None:
                vehicle_live = AutonomousVehicle(
                    simulation_mode=False, enable_logging=True
                )
            vehicle_live.run_heartbeat_test()

        elif choice in ('2', '3'):
            if vehicle_sim is None:
                print("\n[INFO] Initialising simulation subsystems...")
                vehicle_sim = AutonomousVehicle(
                    simulation_mode=True, enable_logging=True
                )
            if choice == '2':
                vehicle_sim.run_vision_test()
            else:
                vehicle_sim.run_integration_test()

        elif choice == '4':
            print("\n" + "="*60)
            print("WARNING: LIVE MOTOR MODE")
            print("="*60)
            print("\nRequirements:")
            print("  - Vehicle on track with clear lane markings")
            print("  - Clear path ahead")
            print("  - Emergency stop accessible (ESC / Ctrl+C)")
            print("  - Adequate lighting")
            print("\nSafety controls:")
            print("  ESC    — immediate emergency stop (OpenCV window focus)")
            print("  Ctrl+C — emergency stop (terminal, always works)")
            print("  'o'    — toggle manual override (terminal, no X11 needed)")
            print("  'q'    — quit safely")

            AutonomousVehicle._restore_terminal()

            if vehicle_sim is not None:
                print("\n[INFO] Releasing simulation hardware resources...")
                vehicle_sim = None
                time.sleep(0.5)

            print("[INFO] Initialising hardware interface...")
            if vehicle_live is None:
                vehicle_live = AutonomousVehicle(
                    simulation_mode=False, enable_logging=True
                )
            vehicle_live.run_integration_test()

        elif choice == '5':
            print("\n" + "="*60)
            print("SHUTTING DOWN")
            print("="*60)
            if vehicle_live and vehicle_live.motor_control:
                vehicle_live.motor_control.emergency_stop()
            break

        else:
            print(f"\n[ERROR] Invalid choice '{choice}'. Please select 1-5.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n" + "="*60)
        print("EMERGENCY SHUTDOWN")
        print("="*60)
        print("\n[INTERRUPT] Keyboard interrupt detected")
        print("[SAFETY] Terminating all processes...\n")
        sys.exit(0)
    except Exception as e:
        print("\n\n" + "="*60)
        print("FATAL ERROR")
        print("="*60)
        print(f"\n[ERROR] {e}\n")
        import traceback
        traceback.print_exc()
        print("\n[CRITICAL] System terminated due to unhandled exception\n")
        sys.exit(1)
