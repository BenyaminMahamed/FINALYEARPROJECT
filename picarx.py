# picarx.py - Mock version for development without hardware
# This simulates the PiCar-X API for testing on non-Raspberry Pi systems

class Picarx:
    """
    Mock PiCar-X class for development
    Simulates the real hardware interface without actual motors/servos
    """
    
    def __init__(self):
        print("[MOCK] PiCar-X initialized (simulation mode)")
        self.speed = 0
        self.steering_angle = 0
        self.camera_pan = 0
        self.camera_tilt = 0
    
    def forward(self, speed):
        """Simulate forward motion"""
        self.speed = speed
        if speed > 0:
            print(f"[MOCK] Moving forward at speed {speed}")
        else:
            print(f"[MOCK] Stopped")
    
    def backward(self, speed):
        """Simulate backward motion"""
        self.speed = -speed
        print(f"[MOCK] Moving backward at speed {speed}")
    
    def stop(self):
        """Simulate stop"""
        self.speed = 0
        print("[MOCK] Stopped")
    
    def set_dir_servo_angle(self, angle):
        """Simulate steering"""
        self.steering_angle = angle
        print(f"[MOCK] Steering angle set to {angle}°")
    
    def set_camera_servo1_angle(self, angle):
        """Simulate camera pan servo"""
        self.camera_pan = angle
        print(f"[MOCK] Camera pan set to {angle}°")
    
    def set_camera_servo2_angle(self, angle):
        """Simulate camera tilt servo"""
        self.camera_tilt = angle
        print(f"[MOCK] Camera tilt set to {angle}°")
    
    def get_distance(self):
        """Simulate ultrasonic distance sensor"""
        return 50.0