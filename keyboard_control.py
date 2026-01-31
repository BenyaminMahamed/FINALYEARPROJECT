#!/usr/bin/env python3
"""
Keyboard Control for PiCar-X
Control the car manually using WASD keys
"""

from picarx import Picarx
import sys
import tty
import termios

class KeyboardController:
    def __init__(self):
        self.px = Picarx()
        self.speed = 30  # Default speed
        self.steering_angle = 0
        self.running = True
        
    def get_key(self):
        """Get single keypress from user"""
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            key = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return key
    
    def display_controls(self):
        """Display control instructions"""
        print("\n" + "="*50)
        print("PiCar-X Keyboard Control")
        print("="*50)
        print("\nControls:")
        print("  W - Forward")
        print("  S - Backward")
        print("  A - Turn Left")
        print("  D - Turn Right")
        print("  SPACE - Stop")
        print("  + - Increase Speed")
        print("  - - Decrease Speed")
        print("  Q - Quit")
        print("\nCurrent Speed:", self.speed)
        print("="*50 + "\n")
    
    def run(self):
        """Main control loop"""
        self.display_controls()
        
        try:
            while self.running:
                key = self.get_key().lower()
                
                if key == 'w':
                    print("Moving FORWARD...")
                    self.px.backward(self.speed)  # SWAPPED
                    
                elif key == 's':
                    print("Moving BACKWARD...")
                    self.px.forward(self.speed)  # SWAPPED
                    
                elif key == 'a':
                    print("Turning LEFT...")
                    self.steering_angle = -30
                    self.px.set_dir_servo_angle(self.steering_angle)
                    
                elif key == 'd':
                    print("Turning RIGHT...")
                    self.steering_angle = 30
                    self.px.set_dir_servo_angle(self.steering_angle)
                    
                elif key == ' ':
                    print("STOPPED")
                    self.px.stop()
                    self.steering_angle = 0
                    self.px.set_dir_servo_angle(self.steering_angle)
                    
                elif key == '+' or key == '=':
                    self.speed = min(100, self.speed + 10)
                    print(f"Speed increased to: {self.speed}")
                    
                elif key == '-' or key == '_':
                    self.speed = max(10, self.speed - 10)
                    print(f"Speed decreased to: {self.speed}")
                    
                elif key == 'q':
                    print("\nQuitting...")
                    self.running = False
                    
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
        
        finally:
            # Clean up - stop the car
            print("Stopping car...")
            self.px.stop()
            self.px.set_dir_servo_angle(0)
            print("Goodbye!\n")

if __name__ == "__main__":
    controller = KeyboardController()
    controller.run() 