#!/usr/bin/env python3
"""
Keyboard Control for PiCar-X
Control the car manually using WASD keys.
Used for Priority 1 Manual Override validation.
"""

from picarx import Picarx
import sys
import tty
import termios
import config # Import global settings

class KeyboardController:
    def __init__(self):
        # Initialize hardware using shared config parameters
        self.px = Picarx()
        self.speed = config.BASE_SPEED
        self.steering_angle = 0
        self.trim = config.STEER_TRIM
        self.running = True
        
    def get_key(self):
        """Get single keypress from terminal without needing Enter"""
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            key = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return key
    
    def display_controls(self):
        """Display control instructions for the User Manual/Appendix"""
        print("\n" + "="*50)
        print("PiCar-X Keyboard Control - MANUAL OVERRIDE")
        print("="*50)
        print("\nControls:")
        print("  W - Forward")
        print("  S - Backward")
        print("  A - Turn Left")
        print("  D - Turn Right")
        print("  SPACE - Stop/Center")
        print("  + - Increase Speed")
        print("  - - Decrease Speed")
        print("  Q - Quit")
        print(f"\nCurrent Speed: {self.speed} | Trim: {self.trim}")
        print("="*50 + "\n")
    
    def run(self):
        """Main control loop for Manual Mode"""
        self.display_controls()
        
        try:
            while self.running:
                key = self.get_key().lower()
                
                # FIXED: Motor Polarity
                if key == 'w':
                    print("Moving FORWARD...")
                    self.px.forward(self.speed) 
                    
                elif key == 's':
                    print("Moving BACKWARD...")
                    self.px.backward(self.speed)  
                    
                # FIXED: Steering Inversion
                # In standard PiCar-X, negative is Left, positive is Right
                elif key == 'a':
                    print("Turning LEFT...")
                    self.steering_angle = -30 + self.trim
                    self.px.set_dir_servo_angle(self.steering_angle)
                    
                elif key == 'd':
                    print("Turning RIGHT...")
                    self.steering_angle = 30 + self.trim
                    self.px.set_dir_servo_angle(self.steering_angle)
                    
                elif key == ' ':
                    print("STOPPED / CENTERED")
                    self.px.stop()
                    self.steering_angle = 0 + self.trim
                    self.px.set_dir_servo_angle(self.steering_angle)
                    
                elif key == '+' or key == '=':
                    self.speed = min(config.MAX_SPEED, self.speed + 10)
                    print(f"Speed increased to: {self.speed}")
                    
                elif key == '-' or key == '_':
                    self.speed = max(config.MIN_SPEED, self.speed - 10)
                    print(f"Speed decreased to: {self.speed}")
                    
                elif key == 'q':
                    print("\nQuitting Manual Mode...")
                    self.running = False
                    
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
        
        finally:
            # Emergency Stop on Exit
            print("Cleaning up hardware states...")
            self.px.stop()
            self.px.set_dir_servo_angle(0 + self.trim)
            print("Hardware neutralized. Goodbye!\n")

if __name__ == "__main__":
    controller = KeyboardController()
    controller.run()
