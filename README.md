# Autonomous Navigation System for Assisted Mobility
A classical computer vision system for real-time lane following and obstacle detection,
built on a Raspberry Pi 5 with a SunFounder PiCar-X robot car.

Developed as a Final Year Project at the University of Westminster, demonstrating
autonomous navigation using OpenCV without deep learning or neural networks.

---

## About the Project

This project implements an autonomous navigation pipeline for a physical robot car
using classical computer vision techniques. The system detects road lanes via colour
masking and contour analysis, controls steering through a PID controller, and halts
the vehicle when an obstacle is detected via ultrasonic sensor.

The system runs entirely on-device on a Raspberry Pi 5 and operates across four modes:
STOPPED, AUTONOMOUS, OBSTACLE STOP, and MANUAL override.

---

## Features

- Real-time lane detection using HSV colour masking and contour analysis
- PID-based steering control for smooth lane following
- Ultrasonic obstacle detection with automatic emergency stop
- Keyboard-based manual override (press `o` to activate)
- Modular 7-module architecture
- Runs on Raspberry Pi 5 with no internet connection required
- Headless operation supported for accurate FPS measurement

---

## Hardware Requirements

- Raspberry Pi 5 (4GB or 8GB)
- SunFounder PiCar-X robot car kit
- Raspberry Pi Camera Module (mounted front-facing)
- Ultrasonic distance sensor (included with PiCar-X)
- MicroSD card (32GB+ recommended)
- Power supply or battery pack compatible with PiCar-X

---

## Technologies Used

- Python 3
- OpenCV — lane detection and image processing
- NumPy — numerical operations
- RPi.GPIO / SunFounder PiCar-X SDK — motor and servo control
- pigpio — GPIO management on Raspberry Pi
- Keyboard input handling for manual override

---

## Installation

### On the Raspberry Pi

1. Clone the repositorygit clone https://github.com/BenyaminMahamed/FINALYEARPROJECT.git
cd FINALYEARPROJECT

2. Install system dependenciessudo apt update
sudo apt install python3-opencv python3-numpy pigpio
sudo systemctl enable pigpiod
sudo systemctl start pigpiod

3. Install Python dependenciespip install -r requirements.txt

4. Ensure the PiCar-X SDK is installed and calibrated per the
SunFounder documentation before running.

---

## Usage

### Run the main systempython main.py

### Run in headless mode (recommended for accurate FPS)python main.py --headless

### Run a specific test modepython main.py --mode 4

Mode reference:
- Mode 1 — Vision/lane detection test
- Mode 2 — Integration test (STOPPED/AUTONOMOUS)
- Mode 3 — Obstacle detection test
- Mode 4 — Live autonomous driving on track

---

## Project StructureFINALYEARPROJECT/
├── config.py              # Configuration and tuning parameters

├── control_logic.py       # PID steering and drive control

├── keyboard_control.py    # Keyboard manual override handler

├── lane_detection.py      # HSV masking, contour analysis, lane centroid

├── object_detection.py    # Ultrasonic obstacle detection logic

├── remote_override.py     # Manual/remote override logic

├── main.py                # Application entry point and mode selector

├── test_logs/             # CSV and JSON output from test runs

├── models/                # Placeholder for future model extensions

├── .gitignore

└── README.md

---

## Controls

| Key | Action |
|-----|--------|
| `o` | Toggle manual override |
| `w` | Forward |
| `a` | Left |
| `s` | Backward |
| `d` | Right |
| `+` | Increase Speed |
| `-` | Decrease Speed |
| `q` | Quit |

---

## System Modes

| Mode | Label | Description |
|------|-------|-------------|
| STOPPED | System idle, motors off |
| AUTONOMOUS | Lane following active |
| OBSTACLE STOP | Obstacle detected, halted |
| MANUAL | Keyboard override active |

---

## Demo

A full video demonstration of the system running autonomously on a physical track,
including obstacle detection and manual override, is available at:

**[Demo Video — link to be added post-testing]**

---

## Future Improvements

- Deep learning-based lane and object detection
- Improved robustness in variable lighting conditions
- Traffic sign recognition
- Higher FPS optimisation
- GUI dashboard for live telemetry

---

## License

MIT License. Free to use, modify, and distribute for academic and research purposes.

---

## Author

**Benyamin Mahamed**
BSc Computer Science, University of Westminster
Final Year Project — 2025/26
