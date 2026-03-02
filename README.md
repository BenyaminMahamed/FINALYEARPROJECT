# Lane Detection and Object Detection System

A Python-based computer vision system that performs real-time lane detection and obstacle detection using video input.
This project was developed as a **Final Year Project** and demonstrates the use of OpenCV and classical computer vision techniques for road scene analysis and basic driving control logic.

---

## About the Project

This project focuses on detecting road lanes and obstacles from video frames using traditional computer vision methods.
It applies image processing techniques such as edge detection, masking, and blob detection to analyze road scenes and simulate decision-making logic for autonomous driving systems.

The system processes real-time camera input or recorded video footage and includes control logic that can be manually overridden using keyboard inputs for testing and simulation purposes.

---

## Features

- Real-time lane detection using OpenCV  
- Obstacle detection using classical computer vision techniques  
- Keyboard-based manual control override  
- Modular and well-structured codebase  
- Suitable for simulation, experimentation, and academic research  

---

## Technologies Used

- Python  
- OpenCV for computer vision and image processing  
- NumPy for numerical and array operations  
- Real-time video frame processing  
- Keyboard input handling for manual override  
- Configuration management using Python modules  

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

To use a video file instead of a live camera feed:

python main.py --video sample_video.mp4  

The system will display detected lane markings and obstacles in real time.

---

## Project Structure

FINALYEARPROJECT/  
├── config.py                # Configuration settings  
├── control_logic.py         # Steering and control logic  
├── keyboard_control.py      # Keyboard override controls  
├── lane_detection.py        # Lane detection algorithms  
├── object_detection.py      # Obstacle detection logic  
├── remote_override.py       # Manual / remote override logic  
├── main.py                  # Application entry point  
├── models/                  # Placeholder for future model extensions  
├── .gitignore  
└── README.md  

---

## Controls

- Keyboard input allows manual override of automated behavior  
- Designed primarily for testing and simulation environments  

---

## Future Improvements

- Integrate deep learning-based object detection models  
- Improve detection robustness in complex environments  
- Add traffic sign and signal recognition  
- Optimize performance for higher frame rates  
- Develop a graphical user interface (GUI)  

---

## License

This project is licensed under the **MIT License**.
You are free to use, modify, and distribute this project for academic and research purposes.

---

## Author

**Benyamin Mahamed**  
Final Year Project