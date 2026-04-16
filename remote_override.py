#!/usr/bin/env python3
"""
Remote Override System - Manual Safety Control
Student: Benyamin Mahamed (W1966430)
Project: Autonomous Self-Driving Car for Assisted Mobility
"""

import sys
import tty
import termios
from picarx import Picarx
from typing import Dict, Optional
import time

class RemoteOverride:
    """
    Manual override system with priority-based control.
    """
    
    # Speed and steering limits
    MIN_SPEED = 10
    MAX_SPEED = 100
    DEFAULT_SPEED = 30
    MAX_STEERING_ANGLE = 30
    
    def __init__(self, picarx_instance: Optional[Picarx] = None):
        """
        Initialize remote override system.
        """
        try:
            print("[OVERRIDE] Initializing remote override system...")
            # Use shared hardware instance if provided to prevent GPIO conflicts
            if picarx_instance:
                self.px = picarx_instance
                print("[OVERRIDE] ✓ Using shared hardware interface")
            else:
                self.px = Picarx()
                print("[OVERRIDE] ✓ New hardware interface created")
        except Exception as e:
            print(f"[OVERRIDE] ✗ Failed to initialize: {e}")
            raise
        
        # State variables
        self.override_active = False
        self.speed = self.DEFAULT_SPEED
        self.steering = 0
        
        # Statistics for validation
        self.activation_count = 0
        self.deactivation_count = 0
        self.emergency_stop_count = 0
        self.command_count = 0
        
        # Event callback (optional - for logging integration)
        self.event_callback = None
        
        print("[OVERRIDE] Remote override system initialized")

    def set_event_callback(self, callback):
        """
        Set callback function for event logging.
        """
        self.event_callback = callback

    def _log_event(self, event_type: str, details: str):
        """Internal event logging"""
        if self.event_callback:
            self.event_callback(event_type, details)

    def activate_override(self):
        """Activate manual override mode (FR3.1)."""
        start_time = time.time()
        self.px.stop()
        self.px.set_dir_servo_angle(0)
        self.override_active = True
        self.activation_count += 1
        response_time_ms = (time.time() - start_time) * 1000
        print(f"[OVERRIDE] ⚠ Manual override ACTIVATED (Response: {response_time_ms:.2f}ms)")
        self._log_event('mode_change', f'Override activated (count: {self.activation_count})')

    def deactivate_override(self):
        """Deactivate manual override mode (FR3.1)."""
        self.px.stop()
        self.px.set_dir_servo_angle(0)
        self.override_active = False
        self.speed = self.DEFAULT_SPEED
        self.steering = 0
        self.deactivation_count += 1
        print(f"[OVERRIDE] Manual override DEACTIVATED (count: {self.deactivation_count})")
        self._log_event('mode_change', f'Override deactivated (count: {self.deactivation_count})')

    def is_active(self) -> bool:
        """Check if override is currently active."""
        return self.override_active

def process_manual_command(self, command: str):
        """Process manual control command (FR3.3)."""
        if not self.override_active:
            return
        
        self.command_count += 1
        
        if command == 'forward':
            # If backward() was making it go back, use forward()
            self.px.forward(self.speed) 
            
        elif command == 'backward':
            # If forward() was making it go back, use backward()
            self.px.backward(self.speed)
            
        elif command == 'left':
            # Flip the sign: if -30 went right, use +30
            self.steering = self.MAX_STEERING_ANGLE 
            self.px.set_dir_servo_angle(self.steering)
            
        elif command == 'right':
            # Flip the sign: if +30 went right, use -30
            self.steering = -self.MAX_STEERING_ANGLE
            self.px.set_dir_servo_angle(self.steering)
            
        elif command == 'stop':
            self.px.stop()
            self.px.set_dir_servo_angle(0)

    def emergency_stop(self):
        """Execute emergency stop (NFR-S2)."""
        self.px.stop()
        self.px.set_dir_servo_angle(0)
        self.override_active = False
        self.emergency_stop_count += 1
        print(f"[OVERRIDE] ⚠ EMERGENCY STOP (Count: {self.emergency_stop_count})")
        self._log_event('emergency_stop', f'Emergency stop triggered')

    def print_statistics(self):
        """Print session statistics for validation"""
        print("\n" + "="*60)
        print("REMOTE OVERRIDE - SESSION STATISTICS")
        print("="*60)
        print(f"  Override activations:    {self.activation_count}")
        print(f"  Emergency stops:         {self.emergency_stop_count}")
        print(f"  Commands processed:      {self.command_count}")
        print("="*60 + "\n")

class ManualControl:
    """
    Standalone manual control interface for testing.
    
    Provides interactive keyboard control for validating:
        - FR3.1: Manual override activation
        - FR3.3: WASD manual control
        - NFR-S2: Emergency stop reliability
        - NFR-U1: Override response time
    """
    
    def __init__(self):
        """Initialize standalone manual control"""
        print("\n[MANUAL CONTROL] Initializing standalone mode...")
        self.override = RemoteOverride()
        self.override.activate_override()
        self.running = True
    
    def get_key(self) -> str:
        """
        Get single keypress without requiring Enter.
        
        Returns:
            Single character key press
        """
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            key = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return key
    
    def display_instructions(self):
        """Display control instructions"""
        print("\n" + "="*60)
        print("MANUAL OVERRIDE CONTROL - Standalone Test Mode")
        print("Student: Benyamin Mahamed (W1966430)")
        print("="*60)
        print("\nControls:")
        print("  W - Forward")
        print("  S - Backward")
        print("  A - Left")
        print("  D - Right")
        print("  SPACE - Stop")
        print("  + / - - Speed adjustment")
        print("  ESC - Emergency stop (NFR-S2)")
        print("  Q - Quit")
        print("="*60 + "\n")
        print("[READY] Manual override active - awaiting commands...\n")
    
    def run(self):
        """
        Run manual control loop for testing.
        
        Validates manual control subsystem functionality.
        """
        self.display_instructions()
        
        try:
            while self.running:
                key = self.get_key().lower()
                
                # Motion controls
                if key == 'w':
                    self.override.process_manual_command('forward')
                elif key == 's':
                    self.override.process_manual_command('backward')
                elif key == 'a':
                    self.override.process_manual_command('left')
                elif key == 'd':
                    self.override.process_manual_command('right')
                    
                # Stop
                elif key == ' ':
                    self.override.process_manual_command('stop')
                    
                # Speed controls
                elif key == '+' or key == '=':
                    self.override.process_manual_command('speed_up')
                elif key == '-' or key == '_':
                    self.override.process_manual_command('speed_down')
                    
                # Emergency stop
                elif key == '\x1b':  # ESC key
                    self.override.emergency_stop()
                    
                # Quit
                elif key == 'q':
                    print("\n[QUIT] Shutting down manual control...")
                    self.running = False
                    
        except KeyboardInterrupt:
            print("\n\n[INTERRUPT] Keyboard interrupt detected")
            
        except Exception as e:
            print(f"\n[ERROR] Manual control exception: {e}")
            import traceback
            traceback.print_exc()
            
        finally:
            self._cleanup()
    
    def _cleanup(self):
        """Safe shutdown with statistics"""
        print("\n[CLEANUP] Executing emergency stop...")
        self.override.emergency_stop()
        
        # Print statistics
        self.override.print_statistics()
        
        print("[CLEANUP] ✓ Complete - Vehicle in safe state")
        print("Goodbye!\n")


def main():
    """
    Entry point for standalone manual control testing.
    
    Used for validating FR3.1, FR3.3, NFR-S2, and NFR-U1 requirements.
    """
    print("\n" + "="*60)
    print("REMOTE OVERRIDE SYSTEM - Standalone Test")
    print("Autonomous Self-Driving Car for Assisted Mobility")
    print("="*60)
    
    try:
        controller = ManualControl()
        controller.run()
    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
