# remote_override.py - Remote Override System
# Allows manual control override of autonomous system

import sys
import tty
import termios
from picarx import Picarx


class RemoteOverride:
    """
    Remote override system for manual control
    Provides keyboard-based manual override of autonomous driving
    """
    
    def __init__(self):
        self.px = Picarx()
        self.override_active = False
        self.speed = 30
        self.steering = 0
        print("[OVERRIDE] Remote override system initialized")
    
    def activate_override(self):
        """Activate manual override mode"""
        self.override_active = True
        self.px.stop()
        self.px.set_dir_servo_angle(0)
        print("[OVERRIDE] ⚠ Manual override ACTIVATED")
    
    def deactivate_override(self):
        """Deactivate manual override, return to autonomous"""
        self.override_active = False
        self.px.stop()
        self.px.set_dir_servo_angle(0)
        self.speed = 30
        self.steering = 0
        print("[OVERRIDE] Manual override DEACTIVATED - returning to autonomous")
    
    def is_active(self):
        """Check if override is currently active"""
        return self.override_active
    
    def process_manual_command(self, command):
        """
        Process manual control command
        
        Args:
            command: string command ('forward', 'backward', 'left', 'right', 'stop')
        """
        if not self.override_active:
            return
        
        if command == 'forward':
            self.px.backward(self.speed)  # Swapped for your hardware
        elif command == 'backward':
            self.px.forward(self.speed)   # Swapped for your hardware
        elif command == 'left':
            self.steering = -30
            self.px.set_dir_servo_angle(self.steering)
        elif command == 'right':
            self.steering = 30
            self.px.set_dir_servo_angle(self.steering)
        elif command == 'stop':
            self.px.stop()
            self.steering = 0
            self.px.set_dir_servo_angle(0)
        elif command == 'speed_up':
            self.speed = min(100, self.speed + 10)
            print(f"[OVERRIDE] Speed: {self.speed}")
        elif command == 'speed_down':
            self.speed = max(10, self.speed - 10)
            print(f"[OVERRIDE] Speed: {self.speed}")
    
    def emergency_stop(self):
        """Emergency stop - works in any mode"""
        self.px.stop()
        self.px.set_dir_servo_angle(0)
        self.override_active = False
        print("[OVERRIDE] ⚠ EMERGENCY STOP")
    
    def get_status(self):
        """Get current override status"""
        return {
            'active': self.override_active,
            'speed': self.speed,
            'steering': self.steering
        }


# Standalone manual control mode for testing
class ManualControl:
    """Standalone manual control interface"""
    
    def __init__(self):
        self.override = RemoteOverride()
        self.override.activate_override()
        self.running = True
    
    def get_key(self):
        """Get single keypress"""
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            key = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return key
    
    def run(self):
        """Run manual control loop"""
        print("\n" + "="*50)
        print("MANUAL OVERRIDE CONTROL")
        print("="*50)
        print("\nControls:")
        print("  W - Forward")
        print("  S - Backward")
        print("  A - Left")
        print("  D - Right")
        print("  SPACE - Stop")
        print("  +/- - Speed")
        print("  Q - Quit")
        print("="*50 + "\n")
        
        try:
            while self.running:
                key = self.get_key().lower()
                
                if key == 'w':
                    self.override.process_manual_command('forward')
                elif key == 's':
                    self.override.process_manual_command('backward')
                elif key == 'a':
                    self.override.process_manual_command('left')
                elif key == 'd':
                    self.override.process_manual_command('right')
                elif key == ' ':
                    self.override.process_manual_command('stop')
                elif key == '+' or key == '=':
                    self.override.process_manual_command('speed_up')
                elif key == '-' or key == '_':
                    self.override.process_manual_command('speed_down')
                elif key == 'q':
                    print("\nQuitting...")
                    self.running = False
                    
        except KeyboardInterrupt:
            print("\n\nInterrupted")
        finally:
            self.override.emergency_stop()
            print("Goodbye!\n")


if __name__ == "__main__":
    # Test standalone manual control
    controller = ManualControl()
    controller.run()