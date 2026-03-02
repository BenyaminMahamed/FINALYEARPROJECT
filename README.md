# Lane Detection and Object Detection System

A Python-based computer vision system that performs real-time lane detection and object detection using video input.
This project was developed as a Final Year Project and demonstrates the use of OpenCV and machine learning techniques for road scene analysis and basic driving control logic.

---

## About the Project

This project focuses on detecting road lanes and objects such as vehicles or obstacles from video frames.
It combines classical computer vision techniques with object detection models to simulate decision-making logic for autonomous driving systems.

The system processes real-time video or recorded footage and provides control logic that can be overridden using keyboard inputs.

---

## Features

- Real-time lane detection using OpenCV
- Object detection for vehicles and obstacles
- Keyboard-based control override
- Modular and extensible code structure
- Suitable for simulation and research purposes

---

## Technologies Used

- Python
- OpenCV
- NumPy
- Machine Learning / Object Detection Models
- Keyboard input handling

---

## Installation

1. Clone the repository

git clone https://github.com/BenyaminMahamed/FINALYEARPROJECT.git  
cd FINALYEARPROJECT  

2. Create a virtual environment (recommended)

python -m venv venv  

3. Activate the virtual environment

Windows:  
venv\Scripts\activate  

macOS / Linux:  
source venv/bin/activate  

4. Install dependencies

pip install opencv-python numpy  

---

## Usage

Run the main application:

python main.py  

To use a video file instead of a camera feed:

python main.py --video sample_video.mp4  

The system will display detected lanes and objects in real time.

---

## Project Structure

FINALYEARPROJECT/  
├── config.py  
├── control_logic.py  
├── keyboard_control.py  
├── lane_detection.py  
├── object_detection.py  
├── remote_override.py  
├── main.py  
├── models/  
├── .gitignore  
└── README.md  

---

## Controls

- Keyboard input allows manual override of automated controls
- Used mainly for testing and simulation purposes

---

## Future Improvements

- Improve detection accuracy using deep learning models
- Add real-world vehicle integration
- Implement traffic sign recognition
- Improve performance and optimization
- Add a graphical user interface

---

## Author

Benyamin Mahamed  
Final Year Project