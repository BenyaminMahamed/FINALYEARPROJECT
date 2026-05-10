# Autonomous Navigation System for Assisted Mobility

**Final Year Project — BSc Computer Science, University of Westminster (2025/26)**  
**Supervisor:** Dr. Anastasia Angelopoulou

A Classical Computer Vision system for real-time lane following and obstacle detection, built on a Raspberry Pi 5 with a SunFounder PiCar-X. Designed as a proof-of-concept for affordable assistive mobility technology — demonstrating that the core navigation capabilities of systems costing £5,000+ can be replicated for under £200 on open-source hardware.

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Raspberry Pi](https://img.shields.io/badge/-RaspberryPi-C51A4A?style=for-the-badge&logo=Raspberry-Pi)
![Git](https://img.shields.io/badge/git-%23F05033.svg?style=for-the-badge&logo=git&logoColor=white)

---
## Youtube demo video 

[![Autonomous Navigation Demo](https://img.youtube.com/vi/ol9_oAe9Ogk/maxresdefault.jpg)](https://www.youtube.com/watch?v=ol9_oAe9Ogk)

--- 
## The problem it solves

Around 2.7 million people in the UK use mobility aids. For outdoor wheelchair users, the challenge isn't just getting from A to B — it's the constant cognitive burden of environmental awareness: watching for kerbs, surface irregularities, partially obstructing objects. Over longer journeys this sustained vigilance causes real mental fatigue that limits independence.

Commercial autonomous wheelchair solutions like the WHILL Model C2 cost over £5,000 and are built primarily for indoor use. Academic prototypes require GPU hardware and tens of thousands of pounds in equipment. This project asks: what can you actually achieve on a £200 embedded platform with open-source software?

The answer: lane following, obstacle detection, automatic emergency stop, and manual override — all running in real time, all validated on physical hardware.

---

## Demo

**[Full demonstration video (5 min)](https://www.youtube.com/watch?v=ol9_oAe9Ogk)** — recorded during a live Mode 4 integration session on physical hardware, showing autonomous lane following, obstacle stop, and manual override in sequence.

---

## Key results

All performance targets were met with substantial margins across a final integration session of **10,298 frames over 345 seconds**:

| Requirement | Target | Result |
|---|---|---|
| Processing latency (NFR-P1) | < 200ms | ~10ms average (20× margin) |
| Frame rate (NFR-P2) | ≥ 8 FPS | ~14 FPS sustained |
| Obstacle detection reliability (NFR-S1) | 100% | 100% — zero missed stops |
| Manual override response (NFR-U1) | < 50ms | < 10ms (5× better) |
| Total hardware cost | — | < £200 |

72 obstacle detection events recorded across the session. 1 confirmed automatic emergency stop at t=297s — triggered correctly, held while obstacle was present, resumed autonomously once path was clear. Zero false positives during autonomous operation.

---

## Why Classical CV instead of Deep Learning

This was a deliberate design decision, not a limitation. YOLOv5 on a Raspberry Pi CPU runs at 80–120ms per frame — before any other processing stage, that alone risks violating the 200ms end-to-end latency target. More fundamentally, the primary stakeholder's requirement was 100% obstacle detection reliability, not object classification accuracy. A probabilistic deep learning model cannot guarantee 100% detection on unseen inputs. A classical blob detector treating any significant foreground mass as an obstacle can — and does.

Classical CV also produces deterministic, interpretable behaviour. For a safety-critical assistive system used by elderly or disabled users, a decision structure that can be traced line by line is more trustworthy than one whose output emerges from model weights.

---

## Architecture

The system runs entirely on a single Raspberry Pi 5 — no cloud, no offloading, no internet connection required. An earlier design planned to split vision processing (Pi) and motor control (Arduino) over UART serial. Empirical measurement killed that: UART introduced 100–150ms of inter-process latency, consuming most of the 200ms budget before a single frame was processed. The integrated single-board architecture eliminated that entirely.

Three layers:

```
Perception Layer      → Picamera2 @ 640×480, fed to lane + obstacle detection simultaneously
Processing Layer      → Data fusion with strict safety priority cascade
Actuation Layer       → DC motors + steering servo via PiCar-X SDK
```

### Safety priority cascade

Every frame evaluates conditions in strict descending order — no exceptions:

```python
if manual_override.is_active():
    execute_manual_commands()       # Priority 1: User always wins

elif obstacle_detector.is_blocked():
    robot.emergency_stop()          # Priority 2: Collision avoidance

elif autonomous_mode.is_engaged():
    robot.follow_lane(vision_data)  # Priority 3: Autonomous navigation

else:
    robot.stop()                    # Priority 4: Default safe state
```

Manual override cannot be pre-empted by anything. Obstacle stop cannot be pre-empted by autonomous navigation. The system never re-engages autonomous mode automatically after a manual override — the operator must explicitly re-activate it.

---

## Lane detection pipeline

Five stages per frame, ~10ms average end-to-end:

1. **ROI masking** — trapezoidal mask retaining only the lower 65% of the frame, cutting background clutter above the travel path
2. **Preprocessing** — greyscale conversion + 5×5 Gaussian blur for noise reduction
3. **Canny edge detection** — dual-threshold hysteresis to identify high-gradient lane boundaries
4. **Probabilistic Hough Transform** — line segment extraction with slope filtering to discard horizontal noise (reflections, cracks, surface markings)
5. **Steering computation** — lateral offset from lane midpoint → proportional gain → servo angle

One non-obvious challenge: the PiCar-X servo's coordinate system is inverted relative to the vision pipeline — a positive lateral offset (vehicle drifting right) requires a negative servo command to steer left. Getting this wrong creates a positive feedback loop that drives the vehicle immediately off-track.

---

## Obstacle detection

Classical blob detection via `cv2.SimpleBlobDetector` within a fixed safety zone occupying the central lower portion of the frame. Class-agnostic — any contiguous blob exceeding 0.8% of the safety zone area triggers an emergency stop, regardless of what the object actually is.

The 0.8% threshold was empirically tuned to eliminate false positives from floor reflections and lighting variation while maintaining detection of any real obstacle. Sub-5ms per frame — negligible compared to the lane detection pipeline.

---

## Module structure

2,893 lines across 7 modules, each with a single clearly defined responsibility:

| Module | Lines | Purpose |
|---|---|---|
| `main.py` | 1,083 | Integration framework, safety cascade, PerformanceLogger |
| `lane_detection.py` | 536 | Full Classical CV pipeline |
| `control_logic.py` | 421 | Hardware actuation, steering normalisation |
| `remote_override.py` | 361 | Manual override state machine |
| `config.py` | 220 | Single source of truth for all parameters |
| `object_detection.py` | 169 | Blob detection, safety zone management |
| `keyboard_control.py` | 103 | WASD input handling during manual mode |

---

## Test modes

```
python main.py          # Interactive mode selector
python main.py --headless   # Headless (accurate FPS measurement)
python main.py --mode 4     # Jump directly to live autonomous driving
```

| Mode | What it validates |
|---|---|
| 1 — Heartbeat | GPIO connectivity, motor and servo range |
| 2 — Vision | Lane detection pipeline in isolation, logs latency + FPS |
| 3 — Integration Sim | Full system with motors disabled — safety cascade white box testing |
| 4 — Integration Live | Full autonomous operation on track |

---

## Controls (manual override mode)

| Key | Action |
|---|---|
| `o` | Toggle manual override |
| `w` / `s` | Forward / Backward |
| `a` / `d` | Left / Right |
| `+` / `-` | Speed up / Slow down |
| `q` | Quit |

---

## Setup

### Hardware
- Raspberry Pi 5 (4GB or 8GB)
- SunFounder PiCar-X robot car kit
- Raspberry Pi Camera Module (front-facing mount)
- MicroSD card (32GB+ recommended)

### Software

```bash
git clone https://github.com/BenyaminMahamed/FINALYEARPROJECT.git
cd FINALYEARPROJECT

sudo apt update
sudo apt install python3-opencv python3-numpy pigpio
sudo systemctl enable pigpiod && sudo systemctl start pigpiod

pip install -r requirements.txt
```

Calibrate the PiCar-X per the SunFounder documentation before first run. Note: `cv2.VideoCapture` is incompatible with the Raspberry Pi 5's libcamera stack — this project uses Picamera2 directly. If you're building on Pi 5 and hitting camera issues with OpenCV, that's why.

---

## Honest limitations

- **Environmental dependency** — Static Canny thresholds work well under controlled lighting but degrade meaningfully in variable conditions. One test session under slightly varied ambient light recorded 82.5% bilateral detection vs 100% in controlled conditions. A production system would need adaptive thresholding or upstream illumination normalisation.
- **P-only controller** — The proportional steering controller works but overshoots on tight curves. A PID controller would improve cornering significantly. Tuning PID on a physically moving platform within a 32-week timeline wasn't practical.
- **1:10 scale** — All control parameters were tuned for the PiCar-X at this scale. A full wheelchair deployment would require recalibration, re-tuning, and formal clinical validation under MHRA medical device guidelines. This is a feasibility study, not a deployable product. This is of course something that can be applied to futurework!

---

## Author

**Benyamin Mahamed** (W1966430)  
BSc Computer Science, University of Westminster  
Supervisor: Dr. Anastasia Angelopoulou
