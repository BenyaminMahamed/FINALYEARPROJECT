# control_logic.py - Motor Control and Actuation Module
# Implements FR1.2, FR2.2, FR3.1, FR3.2

from picarx import Picarx
import time
import config


class RobotMuscle:
    """
    Integrated Control Module (Raspberry Pi 4B)
    Handles all motor actuation and servo control
    """
    
    def __init__(self):
        self.px = Picarx()
        self.is_autonomous = False
        self.current_speed = 0
        self.current_steering = 0
        
        # Steering smoothing
        self.target_steering = 0
        
        print("[MUSCLE] Control module initialized")
    
    def test_servos(self):
        """Heartbeat test for camera servo system"""
        print("[TEST] Camera servo range check...")
        angles = [0, 20, 0, -20, 0]
        
        for angle in angles:
            self.px.set_camera_servo1_angle(angle)
            print(f"  Pan: {angle}°")
            time.sleep(0.5)
        
        self.px.set_camera_servo1_angle(config.PAN_CENTER)
        print("[TEST] Servo test complete\n")
    
    def test_motors(self):
        """Basic motor functionality validation"""
        print("[TEST] Motor system check...")
        
        # Forward test
        print("  → Forward (low speed)")
        self.px.forward(25)
        time.sleep(1.5)
        
        # Steering test
        print("  ← Left turn")
        self.px.set_dir_servo_angle(-20)
        time.sleep(0.8)
        
        print("  → Right turn")
        self.px.set_dir_servo_angle(20)
        time.sleep(0.8)
        
        # Center steering
        self.px.set_dir_servo_angle(0)
        
        # Backward test
        print("  ← Backward (low speed)")
        self.px.backward(25)
        time.sleep(1.5)
        
        # Emergency stop
        self.emergency_stop()
        print("[TEST] Motor test complete\n")
    
    def execute_motion(self, speed, steering_angle):
        """
        Primary motion execution method (FR1.2)
        Called by data fusion/decision logic
        
        Args:
            speed: Target speed (-100 to 100, negative = backward)
            steering_angle: Steering angle in degrees (-40 to 40)
        """
        # Safety constraints
        speed = max(-config.MAX_SPEED, min(config.MAX_SPEED, speed))
        steering_angle = max(-config.MAX_STEER_ANGLE, 
                           min(config.MAX_STEER_ANGLE, steering_angle))
        
        # Apply steering smoothing
        self.target_steering = steering_angle
        smooth_steering = (self.current_steering * (1 - config.STEER_SMOOTHING) + 
                          self.target_steering * config.STEER_SMOOTHING)
        
        # Execute commands
        if speed > 0:
            self.px.forward(int(speed))
        elif speed < 0:
            self.px.backward(int(abs(speed)))
        else:
            self.px.forward(0)
        
        self.px.set_dir_servo_angle(int(smooth_steering))
        
        # Update state
        self.current_speed = speed
        self.current_steering = smooth_steering
    
    def emergency_stop(self):
        """
        Immediate halt - implements FR2.2 and FR3.1
        NFR-S1: Must achieve 100% reliability
        """
        self.px.forward(0)
        self.px.set_dir_servo_angle(0)
        self.is_autonomous = False
        self.current_speed = 0
        self.current_steering = 0
        print("[SAFETY] ⚠ Emergency stop executed")
    
    def set_autonomous_mode(self, enabled):
        """Toggle autonomous/manual control (FR3.2)"""
        self.is_autonomous = enabled
        mode = "AUTONOMOUS" if enabled else "MANUAL"
        print(f"[MODE] Switched to {mode}")
        
        if not enabled:
            # Safety: stop when switching to manual
            self.emergency_stop()
    
    def get_telemetry(self):
        """Return current state for logging"""
        return {
            'speed': self.current_speed,
            'steering_angle': self.current_steering,
            'is_autonomous': self.is_autonomous
        }