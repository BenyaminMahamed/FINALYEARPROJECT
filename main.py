from control_logic import RobotMuscle
import time


def main():
    # Initialize the muscle module
    car = RobotMuscle()

    # Execute Heartbeat
    try:
        car.test_servos()
        # Optional: Add a tiny forward/backward nudge to test motors (FR1.2)
    except KeyboardInterrupt:
        car.emergency_stop()  # Mandatory safety override [cite: 50]


if __name__ == "__main__":
    main()