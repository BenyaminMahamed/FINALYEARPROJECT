#!/usr/bin/env python3
"""
Remote Override System - Manual Safety Control
Student: Benyamin Mahamed (W1966430)
Project: Autonomous Self-Driving Car for Assisted Mobility

Architecture Note:
    Override detection uses a background daemon thread reading raw terminal
    stdin (termios), completely decoupled from cv2.waitKey() / X11 focus.
    A threading.Event flag (manual_override_flag) is checked each frame by
    main.py to determine the active priority mode without blocking the vision
    pipeline.

    Priority Hierarchy (highest → lowest):
        1. MANUAL   — this module (FR3.1)
        2. OBSTACLE — obstacle_detector.py (NFR-S2)
        3. AUTO     — lane_follower.py
"""

import sys
import tty
import termios
import threading
import time
from picarx import Picarx
from typing import Optional

# ---------------------------------------------------------------------------
# Module-level shared flag
# ---------------------------------------------------------------------------
# Checked every frame by main.py.  threading.Event is atomically thread-safe;
# no additional locks are required for simple is_set() / set() / clear() calls.
manual_override_flag = threading.Event()


class RemoteOverride:
    """
    Manual override system with priority-based control (FR3.1, FR3.3, NFR-S2).

    Usage (from main.py):
        from remote_override import RemoteOverride, manual_override_flag

        override = RemoteOverride(picarx_instance=px)
        override.start_listener()          # launches background thread

        # Inside main vision loop:
        if manual_override_flag.is_set():
            <skip autonomous logic, draw MANUAL label>
    """

    # ------------------------------------------------------------------
    # Constants
    # ------------------------------------------------------------------
    MIN_SPEED          = 10
    MAX_SPEED          = 100
    DEFAULT_SPEED      = 30
    MAX_STEERING_ANGLE = 30

    def __init__(self, picarx_instance: Optional[Picarx] = None):
        """
        Initialise remote override system.

        Args:
            picarx_instance: Shared Picarx object to avoid GPIO conflicts.
                             If None a new instance is created (standalone use).
        """
        print("[OVERRIDE] Initialising remote override system...")
        try:
            if picarx_instance:
                self.px = picarx_instance
                print("[OVERRIDE] ✓ Using shared hardware interface")
            else:
                self.px = Picarx()
                print("[OVERRIDE] ✓ New hardware interface created")
        except Exception as e:
            print(f"[OVERRIDE] ✗ Failed to initialise hardware: {e}")
            raise

        # Internal state
        self.override_active  = False
        self.speed            = self.DEFAULT_SPEED
        self.steering         = 0
        self._listener_thread: Optional[threading.Thread] = None

        # Session statistics (for Section 6.2 / validation logging)
        self.activation_count    = 0
        self.deactivation_count  = 0
        self.emergency_stop_count = 0
        self.command_count       = 0

        # Optional external event callback (e.g. CSV logger in main.py)
        self.event_callback = None

        print("[OVERRIDE] Remote override system ready")

    # ------------------------------------------------------------------
    # Event callback
    # ------------------------------------------------------------------
    def set_event_callback(self, callback):
        """Register an external callback for event logging."""
        self.event_callback = callback

    def _log_event(self, event_type: str, details: str):
        """Forward event to registered callback if present."""
        if self.event_callback:
            self.event_callback(event_type, details)

    # ------------------------------------------------------------------
    # Background keyboard listener
    # ------------------------------------------------------------------
    def start_listener(self):
        """
        Launch the background daemon thread that listens for keypresses.

        Runs independently of cv2.waitKey() so X11 window focus is NOT
        required.  The thread is a daemon — it exits automatically when
        main.py terminates.

        Key bindings:
            o / O  — toggle MANUAL override on/off
            ESC    — immediate emergency stop
        """
        if self._listener_thread and self._listener_thread.is_alive():
            print("[OVERRIDE] Listener already running — skipping")
            return

        self._listener_thread = threading.Thread(
            target=self._keyboard_listener,
            name="OverrideListener",
            daemon=True        # dies automatically with the main process
        )
        self._listener_thread.start()
        print("[OVERRIDE] ✓ Background keyboard listener started")
        print("[OVERRIDE]   Press 'o' to toggle MANUAL mode | ESC = emergency stop")

    def _get_key(self) -> str:
        """Read a single raw keypress from stdin without requiring Enter."""
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            key = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return key

    def _keyboard_listener(self):
        """
        Internal thread target.  Blocks on raw stdin; updates shared flag and
        override state when 'o' or ESC is pressed.
        """
        print("[OVERRIDE LISTENER] Listening on stdin (thread: OverrideListener)")
        while True:
            try:
                key = self._get_key().lower()

                if key == 'o':
                    if self.override_active:
                        self.deactivate_override()
                    else:
                        self.activate_override()

                elif key == '\x1b':   # ESC
                    self.emergency_stop()

            except Exception as e:
                # Prevent a transient stdin error from killing the thread
                print(f"[OVERRIDE LISTENER] Warning — key read error: {e}")
                time.sleep(0.05)

    # ------------------------------------------------------------------
    # Override state management
    # ------------------------------------------------------------------
    def activate_override(self):
        """
        Activate manual override (FR3.1).
        Stops motors, centres steering, sets shared flag and updates state.
        """
        start_time = time.time()
        self.px.stop()
        self.px.set_dir_servo_angle(0)
        self.override_active = True
        self.activation_count += 1
        manual_override_flag.set()                          # ← notify main loop
        elapsed_ms = (time.time() - start_time) * 1000
        print(f"[OVERRIDE] ⚠ MANUAL override ACTIVATED  "
              f"(response: {elapsed_ms:.2f}ms | count: {self.activation_count})")
        self._log_event('mode_change',
                        f'Manual activated — count: {self.activation_count}')

    def deactivate_override(self):
        """
        Deactivate manual override (FR3.1).
        Stops motors, resets steering and speed, clears shared flag.
        """
        self.px.stop()
        self.px.set_dir_servo_angle(0)
        self.override_active  = False
        self.speed            = self.DEFAULT_SPEED
        self.steering         = 0
        self.deactivation_count += 1
        manual_override_flag.clear()                        # ← notify main loop
        print(f"[OVERRIDE] Manual override DEACTIVATED  "
              f"(count: {self.deactivation_count})")
        self._log_event('mode_change',
                        f'Manual deactivated — count: {self.deactivation_count}')

    def is_active(self) -> bool:
        """Return True if manual override is currently active."""
        return self.override_active

    # ------------------------------------------------------------------
    # Manual command processing
    # ------------------------------------------------------------------
    def process_manual_command(self, command: str):
        """
        Execute a manual drive command (FR3.3).
        No-op if override is not active (safety guard).

        Args:
            command: One of 'forward', 'backward', 'left', 'right', 'stop'
        """
        if not self.override_active:
            return

        self.command_count += 1

        if command == 'forward':
            self.px.forward(self.speed)

        elif command == 'backward':
            self.px.backward(self.speed)

        elif command == 'left':
            # Positive angle = physical left after hardware calibration
            self.steering = self.MAX_STEERING_ANGLE
            self.px.set_dir_servo_angle(self.steering)

        elif command == 'right':
            # Negative angle = physical right after hardware calibration
            self.steering = -self.MAX_STEERING_ANGLE
            self.px.set_dir_servo_angle(self.steering)

        elif command == 'stop':
            self.px.stop()
            self.px.set_dir_servo_angle(0)

        else:
            print(f"[OVERRIDE] Unknown command ignored: '{command}'")

    # ------------------------------------------------------------------
    # Emergency stop
    # ------------------------------------------------------------------
    def emergency_stop(self):
        """
        Immediate full stop — highest priority action (NFR-S2).
        Clears override state AND the shared flag so main loop also reacts.
        """
        self.px.stop()
        self.px.set_dir_servo_angle(0)
        self.override_active = False
        self.emergency_stop_count += 1
        manual_override_flag.clear()                        # ← notify main loop
        print(f"[OVERRIDE] ⚠ EMERGENCY STOP executed  "
              f"(count: {self.emergency_stop_count})")
        self._log_event('emergency_stop',
                        f'E-stop triggered — count: {self.emergency_stop_count}')

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------
    def print_statistics(self):
        """Print session statistics for NFR validation / report evidence."""
        print("\n" + "=" * 60)
        print("  REMOTE OVERRIDE — SESSION STATISTICS")
        print("=" * 60)
        print(f"  Override activations :  {self.activation_count}")
        print(f"  Override deactivations: {self.deactivation_count}")
        print(f"  Emergency stops      :  {self.emergency_stop_count}")
        print(f"  Commands processed   :  {self.command_count}")
        print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Standalone test mode  (python3 remote_override.py)
# ---------------------------------------------------------------------------
class ManualControl:
    """
    Standalone WASD manual control interface for isolated hardware testing.
    Not used when remote_override is imported by main.py.
    """

    def __init__(self):
        print("\n[MANUAL CONTROL] Initialising standalone mode...")
        self.override = RemoteOverride()
        self.override.activate_override()
        self.running = True

    def _get_key(self) -> str:
        """Single raw keypress."""
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            key = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return key

    def _display_instructions(self):
        print("\n" + "=" * 60)
        print("  MANUAL OVERRIDE CONTROL — Standalone Test Mode")
        print("  Student: Benyamin Mahamed (W1966430)")
        print("=" * 60)
        print("  W — Forward      S — Backward")
        print("  A — Left         D — Right")
        print("  SPACE — Stop")
        print("  ESC — Emergency stop (NFR-S2)")
        print("  Q — Quit")
        print("=" * 60 + "\n")

    def run(self):
        """Blocking WASD control loop (standalone only)."""
        self._display_instructions()
        try:
            while self.running:
                key = self._get_key().lower()
                if   key == 'w':      self.override.process_manual_command('forward')
                elif key == 's':      self.override.process_manual_command('backward')
                elif key == 'a':      self.override.process_manual_command('left')
                elif key == 'd':      self.override.process_manual_command('right')
                elif key == ' ':      self.override.process_manual_command('stop')
                elif key == '\x1b':   self.override.emergency_stop()
                elif key == 'q':
                    print("\n[QUIT] Shutting down manual control...")
                    self.running = False
        except KeyboardInterrupt:
            print("\n[INTERRUPT] Keyboard interrupt received")
        except Exception as e:
            print(f"\n[ERROR] {e}")
        finally:
            self._cleanup()

    def _cleanup(self):
        print("\n[CLEANUP] Executing emergency stop...")
        self.override.emergency_stop()
        self.override.print_statistics()
        print("[CLEANUP] ✓ Complete")


def main():
    """Entry point for standalone manual control testing."""
    try:
        ManualControl().run()
    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
