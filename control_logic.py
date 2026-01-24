from picarx import Picarx
import time
import config


class RobotMuscle:
    def __init__(self):
        # Integrated Control via RPi 4B [cite: 142]
        self.px = Picarx()

    def test_servos(self):
        print("Starting Servo Heartbeat Test...")
        # Testing panning for Jonathan's obstacle detection [cite: 77]
        angles = [0, 20, 0, -20, 0]
        for angle in angles:
            self.px.set_camera_servo1_angle(angle)
            print(f"Panning to: {angle} degrees")
            time.sleep(0.5)
        print("Servo Test Complete.")

    def emergency_stop(self):
        # Functional Safety requirement (FR2.2) [cite: 204]
        self.px.forward(0)
        self.px.set_dir_servo_angle(0)