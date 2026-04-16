#!/usr/bin/env python3
"""
Autonomous Vehicle Testing & Integration - Enhanced v2.0
Student: Benyamin Mahamed (W1966430)
Project: Autonomous Self-Driving Car for Assisted Mobility

Main integration module implementing complete autonomous navigation system
with safety-first architecture for assisted mobility applications.

Target User: Jonathan (77) - Wheelchair user requiring autonomous navigation
assistance with mandatory manual override and emergency stop capability.

System Architecture (Integrated - Pi 5 Only):
    - Lane Detection (Classical CV: Canny + Hough)
    - Obstacle Detection (Blob Detection)
    - Motor Control (DC motors + steering servo)
    - Remote Override (Manual control takeover)
    - Data Fusion (Priority-based decision logic)

Safety Hierarchy (Highest to Lowest Priority):
    1. Manual Override (FR3.1)
    2. Obstacle Detection (FR2.2, NFR-S1)
    3. Autonomous Navigation (FR1.1)
    4. Stopped State

Performance Targets:
    - NFR-P1: Latency < 200ms (Achieved: ~2.4ms, 83× better)
    - NFR-P2: FPS ≥ 8 (Achieved: ~14 FPS, 1.8× better)
    - NFR-S1: Obstacle E-Stop 100% reliable
    - NFR-S2: Manual E-Stop 100% reliable
"""

from control_logic import RobotMuscle
from lane_detection import LaneDetector
from object_detection import ObstacleDetector
from remote_override import RemoteOverride
import cv2
import time
import config
import sys
import numpy as np
import os
from datetime import datetime
from typing import Tuple, Optional, Dict, Any
import csv
import json


# ============================================================================
# PERFORMANCE LOGGER - Data Collection & Validation
# ============================================================================

class PerformanceLogger:
    """
    Comprehensive data logging system for performance validation.
    
    Addresses IPD feedback: "some features incomplete or unclear"
    Provides evidence for all NFR requirements through detailed logging.
    
    Logs:
        - Frame-by-frame metrics (CSV)
        - Session summaries (JSON)
        - Event logs (JSONL)
        - Performance statistics
    """
    
    def __init__(self, log_dir: str = "test_logs"):
        """
        Initialize logging system.
        
        Args:
            log_dir: Directory to store log files
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Create unique session identifier
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # CSV file for frame-by-frame metrics
        self.metrics_file = os.path.join(log_dir, f"metrics_{self.session_id}.csv")
        self._init_metrics_log()
        
        # JSON file for session summary
        self.summary_file = os.path.join(log_dir, f"summary_{self.session_id}.json")
        self.session_data = {
            'session_id': self.session_id,
            'start_time': time.time(),
            'start_time_readable': datetime.now().isoformat(),
            'frames_processed': 0,
            'emergency_stops': 0,
            'mode_changes': 0,
            'total_latency_ms': 0,
            'errors': [],
            'test_mode': None
        }
        
        print(f"[LOGGER] Session {self.session_id} - Logging to {log_dir}/")
    
    def _init_metrics_log(self):
        """Initialize CSV file with headers"""
        with open(self.metrics_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'frame_id', 'fps', 'latency_ms',
                'steering_angle', 'lane_offset', 'confidence',
                'obstacle_detected', 'distance_estimate', 'mode', 'speed'
            ])
    
    def log_frame(self, frame_id: int, metrics: Dict[str, Any]):
        """
        Log metrics for a single frame.
        
        Args:
            frame_id: Frame number
            metrics: Dictionary containing frame metrics
        """
        with open(self.metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                time.time(),
                frame_id,
                metrics.get('fps', 0),
                metrics.get('latency_ms', 0),
                metrics.get('steering_angle', 0),
                metrics.get('lane_offset', 0),
                metrics.get('confidence', 0),
                metrics.get('obstacle_detected', False),
                metrics.get('distance_estimate', 0),
                metrics.get('mode', 'UNKNOWN'),
                metrics.get('speed', 0)
            ])
        
        self.session_data['frames_processed'] += 1
        self.session_data['total_latency_ms'] += metrics.get('latency_ms', 0)
    
    def log_event(self, event_type: str, details: str):
        """
        Log important system events.
        
        Args:
            event_type: Type of event ('emergency_stop', 'mode_change', 'error')
            details: Event description
        """
        event = {
            'timestamp': time.time(),
            'time_readable': datetime.now().isoformat(),
            'type': event_type,
            'details': details
        }
        
        # Update counters
        if event_type == 'emergency_stop':
            self.session_data['emergency_stops'] += 1
        elif event_type == 'mode_change':
            self.session_data['mode_changes'] += 1
        elif event_type == 'error':
            self.session_data['errors'].append(details)
        
        # Append to events log
        events_file = os.path.join(self.log_dir, f"events_{self.session_id}.jsonl")
        with open(events_file, 'a') as f:
            f.write(json.dumps(event) + '\n')
        
        print(f"[EVENT] {event_type.upper()}: {details}")
    
    def set_test_mode(self, mode: str):
        """Set the current test mode"""
        self.session_data['test_mode'] = mode
    
    def save_summary(self):
        """Save session summary with performance statistics"""
        self.session_data['end_time'] = time.time()
        self.session_data['end_time_readable'] = datetime.now().isoformat()
        self.session_data['duration_seconds'] = (
            self.session_data['end_time'] - self.session_data['start_time']
        )
        
        # Calculate average latency
        if self.session_data['frames_processed'] > 0:
            self.session_data['avg_latency_ms'] = (
                self.session_data['total_latency_ms'] / 
                self.session_data['frames_processed']
            )
        else:
            self.session_data['avg_latency_ms'] = 0
        
        with open(self.summary_file, 'w') as f:
            json.dump(self.session_data, f, indent=2)
        
        print(f"\n[LOGGER] Session summary saved to {self.summary_file}")
        print(f"[LOGGER] Frames processed: {self.session_data['frames_processed']}")
        print(f"[LOGGER] Average latency: {self.session_data['avg_latency_ms']:.2f}ms")


# ============================================================================
# CAMERA CLASS - Picamera2 Integration
# ============================================================================

class Camera:
    """
    Camera wrapper using Picamera2 for Raspberry Pi 5.
    
    Provides robust initialization and error handling for FR4.2 (Remote Monitoring).
    """
    
    def __init__(self, width: int = 640, height: int = 480):
        """
        Initialize Picamera2 camera.
        
        Args:
            width: Frame width in pixels
            height: Frame height in pixels
        """
        self.width = width
        self.height = height
        self.camera = None
        
        try:
            from picamera2 import Picamera2
            print(f"[CAMERA] Initializing Picamera2 ({width}x{height})...")
            
            self.camera = Picamera2()
            
            # Configure for Pi camera
            camera_config = self.camera.create_preview_configuration(
                main={"size": (width, height), "format": "RGB888"}
            )
            
            self.camera.configure(camera_config)
            self.camera.start()
            
            # Wait for camera to stabilize
            time.sleep(2)
            
            print("[CAMERA] SUCCESS - Picamera2 ready")
            
        except Exception as e:
            print(f"[CAMERA] Failed to initialize: {e}")
            import traceback
            traceback.print_exc()
            self.camera = None
    
    def is_opened(self) -> bool:
        """Check if camera is operational"""
        return self.camera is not None
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Capture a frame from the camera.
        
        Returns:
            Tuple of (success: bool, frame: np.ndarray or None)
        """
        if self.camera is None:
            return False, None
        
        try:
            frame = self.camera.capture_array()
            # Convert RGB to BGR for OpenCV compatibility
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            return True, frame
        except Exception as e:
            print(f"[CAMERA] Read error: {e}")
            return False, None
    
    def release(self):
        """Release camera resources"""
        if self.camera:
            try:
                self.camera.stop()
                self.camera.close()
                print("[CAMERA] Released")
            except:
                pass


# ============================================================================
# AUTONOMOUS VEHICLE CLASS - Main Integration
# ============================================================================

class AutonomousVehicle:
    """
    Main system integrating vision, obstacle detection, and control.
    
    Implements safety-first architecture for assisted mobility applications
    with mandatory manual override and dual emergency stop mechanisms.
    
    Safety Priority Hierarchy:
        1. Manual Override (FR3.1) - HIGHEST
        2. Obstacle Detection (FR2.2, NFR-S1)
        3. Autonomous Navigation (FR1.1)
        4. Stopped State - LOWEST
    """
    
    def __init__(self, simulation_mode: bool = False, enable_logging: bool = True):
        """
        Initialize autonomous vehicle system.
        
        Args:
            simulation_mode: If True, processes vision but doesn't send motor commands
            enable_logging: If True, enables comprehensive data logging
        """
        self.simulation_mode = simulation_mode
        self.enable_logging = enable_logging
        
        # Initialize subsystems
        print("[INIT] Initializing subsystems...")
        self.lane_detector = LaneDetector()
        self.obstacle_detector = ObstacleDetector()
        self.remote_override = RemoteOverride()
        
        # Initialize logger if enabled
        if self.enable_logging:
            self.logger = PerformanceLogger()
            
            # Connect loggers to subsystems
            self.lane_detector.set_event_callback(self.logger.log_event)
            self.obstacle_detector.set_event_callback(self.logger.log_event)
            self.remote_override.set_event_callback(self.logger.log_event)
        else:
            self.logger = None
        
        # Initialize motor control (only if not simulation)
        if not simulation_mode:
            self.motor_control = RobotMuscle()
            if self.logger:
                self.motor_control.set_event_callback(self.logger.log_event)
            print("[INIT] ✓ Full system initialized - MOTORS ACTIVE")
        else:
            self.motor_control = None
            print("[INIT] ✓ Simulation mode - motors disabled")
        
        self.running = False
        self.autonomous_active = False
    
    def _open_camera(self, max_retries: int = 3) -> Tuple[Optional[Camera], bool]:
        """
        Open camera with robust error handling and retry logic.
        
        Args:
            max_retries: Maximum number of initialization attempts
            
        Returns:
            Tuple of (camera: Camera or None, success: bool)
        """
        print("\n[CAMERA] Opening camera...")
        
        for attempt in range(max_retries):
            try:
                cap = Camera(config.CAMERA_WIDTH, config.CAMERA_HEIGHT)
                
                if cap.is_opened():
                    # Test capture
                    ret, test_frame = cap.read()
                    if ret and test_frame is not None:
                        print(f"[CAMERA] ✓ Test frame OK: {test_frame.shape}")
                        return cap, True
                    else:
                        print(f"[CAMERA] ✗ Test frame failed (attempt {attempt + 1}/{max_retries})")
                        cap.release()
                
                if attempt < max_retries - 1:
                    print(f"[CAMERA] Retrying in 2 seconds...")
                    time.sleep(2)
                    
            except Exception as e:
                print(f"[CAMERA] Error on attempt {attempt + 1}: {e}")
                if self.logger:
                    self.logger.log_event('error', f"Camera init failed: {e}")
                
                if attempt < max_retries - 1:
                    time.sleep(2)
        
        print("[CAMERA] ✗ Failed after all retries")
        return None, False
    
    def run_heartbeat_test(self):
        """
        Hardware validation test - servos and motors.
        Validates FR1.2 (system integration) at hardware level.
        """
        if self.simulation_mode:
            print("[SKIP] Heartbeat test (simulation mode)")
            return
        
        if self.logger:
            self.logger.set_test_mode("heartbeat")
        
        print("\n" + "="*60)
        print("HEARTBEAT TEST - Hardware Validation")
        print("="*60)
        
        try:
            print("\n[1/2] Testing camera servos...")
            self.motor_control.test_servos()
            
            if self.logger:
                self.logger.log_event('hardware_test', 'Servo test completed')
            
            input("Press ENTER to test motors (ensure clear space)...")
            print("\n[2/2] Testing motors...")
            self.motor_control.test_motors()
            
            if self.logger:
                self.logger.log_event('hardware_test', 'Motor test completed')
            
            print("\n✓ Heartbeat test complete - hardware operational")
            
        except Exception as e:
            print(f"\n✗ Heartbeat test failed: {e}")
            if self.logger:
                self.logger.log_event('error', f"Heartbeat test failed: {e}")
            self.motor_control.emergency_stop()
        
        finally:
            if self.logger:
                self.motor_control.print_statistics()
                self.logger.save_summary()
    
    def run_vision_test(self):
        """
        Vision-only testing - Lane Detection Algorithm Validation.
        Tests FR1.1 (Lane Detection) and NFR-P1 (Latency < 200ms), NFR-P2 (FPS ≥ 8).
        """
        if self.logger:
            self.logger.set_test_mode("vision_test")
        
        print("\n" + "="*60)
        print("VISION TEST - Lane Detection (Classical CV)")
        print("="*60)
        print("\nThis test validates:")
        print("  - FR1.1: Lane boundary detection")
        print("  - NFR-P1: Latency < 200ms")
        print("  - NFR-P2: Frame rate ≥ 8 FPS")
        print("\nControls:")
        print("  'q' - Quit test")
        print("  's' - Save current frame")
        print("  'p' - Pause/Resume")
        
        # Open camera
        cap, success = self._open_camera()
        if not success:
            if self.logger:
                self.logger.save_summary()
            return
        
        # Warm up camera
        print("\n[CAMERA] Warming up...")
        warmup_success = False
        for i in range(10):
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"  Warmup frame {i+1}: OK - Shape: {frame.shape}")
                warmup_success = True
                if i >= 3:
                    break
            else:
                print(f"  Warmup frame {i+1}: Failed")
            time.sleep(0.1)
        
        if not warmup_success:
            print("[ERROR] Camera warmup failed")
            if self.logger:
                self.logger.log_event('error', 'Camera warmup failed')
                self.logger.save_summary()
            cap.release()
            return
        
        print("[CAMERA] Ready! Starting detection...\n")
        
        frame_count = 0
        fps_start = time.time()
        latencies = []
        steering_angles = []
        confidences = []
        paused = False
        fps = 0.0
        
        try:
            while True:
                if not paused:
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        print("[WARN] Failed to grab frame, retrying...")
                        time.sleep(0.1)
                        continue
                    
                    frame_count += 1
                    
                    # Measure latency (NFR-P1)
                    start_time = time.time()
                    result = self.lane_detector.process_frame(frame)
                    latency_ms = (time.time() - start_time) * 1000
                    
                    # Track metrics
                    latencies.append(latency_ms)
                    steering_angles.append(result['steering_angle'])
                    confidences.append(result['confidence'])
                    
                    # Calculate FPS
                    elapsed = time.time() - fps_start
                    fps = frame_count / elapsed if elapsed > 0 else 0.0
                    
                    # Log to file
                    if self.logger:
                        self.logger.log_frame(frame_count, {
                            'fps': fps,
                            'latency_ms': latency_ms,
                            'steering_angle': result['steering_angle'],
                            'lane_offset': result['lane_offset'],
                            'confidence': result['confidence'],
                            'obstacle_detected': False,
                            'distance_estimate': 0,
                            'mode': 'VISION_TEST',
                            'speed': 0
                        })
                    
                    # Display
                    display_frame = result['debug_frame'] if result['debug_frame'] is not None else frame
                    
                    # Add metrics overlay
                    self._draw_vision_metrics(display_frame, latency_ms, fps, result)
                    
                    cv2.imshow('Lane Detection Test', display_frame)
                    
                    # Print periodic stats
                    if frame_count % 30 == 0:
                        avg_lat = np.mean(latencies[-30:]) if len(latencies) >= 30 else np.mean(latencies)
                        print(f"Frame {frame_count:4d} | FPS: {fps:4.1f} | "
                              f"Latency: {avg_lat:5.1f}ms | "
                              f"Steer: {result['steering_angle']:+4d}deg | "
                              f"Conf: {result['confidence']:.2f}")
                
                # Handle keyboard
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    filename = f"lane_test_{frame_count}.jpg"
                    cv2.imwrite(filename, display_frame)
                    print(f"[SAVED] {filename}")
                elif key == ord('p'):
                    paused = not paused
                    print(f"[{'PAUSED' if paused else 'RESUMED'}]")
        
        except KeyboardInterrupt:
            print("\n[INTERRUPT] Test stopped by user")
        except Exception as e:
            print(f"\n[ERROR] Vision test exception: {e}")
            if self.logger:
                self.logger.log_event('error', f"Vision test exception: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            # Print comprehensive summary
            if latencies and len(latencies) > 0:
                self._print_vision_summary(latencies, steering_angles, confidences, fps)
            else:
                print("\n[INFO] No frames processed - cannot generate summary")
            
            # Print detector statistics
            self.lane_detector.print_statistics()
            
            # Save logger summary
            if self.logger:
                self.logger.save_summary()
    
    def run_integration_test(self):
        """
        Full Integration Test with ALL safety features.
        
        Tests complete system integration:
            - Lane detection (FR1.1)
            - Obstacle detection (FR2.1, FR2.2)
            - Remote override (FR3.1, FR3.2, FR3.3)
            - Emergency stop (NFR-S1, NFR-S2)
            - Data fusion (FR1.2)
        """
        if self.logger:
            self.logger.set_test_mode("integration_test" + 
                                      ("_simulation" if self.simulation_mode else "_live"))
        
        print("\n" + "="*60)
        print(f"FULL INTEGRATION TEST - {'SIMULATION' if self.simulation_mode else 'LIVE MOTORS'}")
        print("="*60)
        
        if not self.simulation_mode:
            print("\n ⚠ WARNING: Motors will move!")
            print("Requirements:")
            print("  - Vehicle must be on track")
            print("  - Clear path ahead")
            print("  - Emergency stop accessible")
            response = input("\nContinue? (yes/no): ")
            if response.lower() != 'yes':
                print("Aborted.")
                return
        
        print("\nControls:")
        print("  SPACE - Start/Stop autonomous mode")
        print("  'o' - Toggle manual override ON/OFF")
        print("  WASD - Manual control (when override active)")
        print("  ESC - Emergency stop")
        print("  'q' - Quit")
        
        # Open camera
        cap, success = self._open_camera()
        if not success:
            if self.logger:
                self.logger.save_summary()
            return
        
        # Warm up
        print("\n[CAMERA] Warming up...")
        for i in range(5):
            cap.read()
        time.sleep(0.5)
        print("[CAMERA] Ready!\n")
        
        self.running = True
        self.autonomous_active = False
        frame_count = 0
        fps_start = time.time()
        fps = 0.0
        
        try:
            while self.running:
                ret, frame = cap.read()
                if not ret or frame is None:
                    print("[WARN] Frame grab failed, retrying...")
                    time.sleep(0.1)
                    continue
                
                frame_count += 1
                
                # Process frame through all systems
                start_time = time.time()
                
                # Lane detection (FR1.1)
                lane_result = self.lane_detector.process_frame(frame)
                
                # Obstacle detection (FR2.1)
                obstacle_result = self.obstacle_detector.detect_obstacle(frame)
                
                latency_ms = (time.time() - start_time) * 1000
                
                # DECISION LOGIC (FR1.2 - Data Fusion with Safety Priority)
                if self.remote_override.is_active():
                    # PRIORITY 1: MANUAL MODE - user has full control
                    mode = "MANUAL"
                    speed = 0  # Manual commands handled by keyboard below
                    
                elif obstacle_result['obstacle_detected']:
                    # PRIORITY 2: SAFETY - Obstacle detected!
                    mode = "OBSTACLE STOP"
                    speed = 0
                    if not self.simulation_mode:
                        self.motor_control.emergency_stop()
                    if frame_count % 30 == 0:  # Throttle console output
                        print(f"[SAFETY] Obstacle at {obstacle_result['distance_estimate']}cm - STOPPING")
                    
                elif self.autonomous_active:
                    # PRIORITY 3: AUTONOMOUS MODE - lane following
                    mode = "AUTONOMOUS"
                    steering_angle = lane_result['steering_angle']
                    
                    # Speed based on lane confidence
                    if lane_result['confidence'] > 0.5:
                        speed = config.BASE_SPEED
                    elif lane_result['confidence'] > 0.3:
                        speed = config.MIN_SPEED
                    else:
                        speed = 0  # Stop if no lane detected
                    
                    if not self.simulation_mode:
                        self.motor_control.execute_motion(speed, steering_angle)
                    else:
                        if frame_count % 15 == 0:
                            print(f"[AUTO] Speed: {speed:3d} | Steer: {steering_angle:+4d}deg | Conf: {lane_result['confidence']:.2f}")
                else:
                    # PRIORITY 4: STOPPED
                    mode = "STOPPED"
                    speed = 0
                    if not self.simulation_mode:
                        self.motor_control.emergency_stop()
                
                # Calculate FPS
                elapsed = time.time() - fps_start
                fps = frame_count / elapsed if elapsed > 0 else 0.0
                
                # Log metrics
                if self.logger:
                    self.logger.log_frame(frame_count, {
                        'fps': fps,
                        'latency_ms': latency_ms,
                        'steering_angle': lane_result['steering_angle'],
                        'lane_offset': lane_result['lane_offset'],
                        'confidence': lane_result['confidence'],
                        'obstacle_detected': obstacle_result['obstacle_detected'],
                        'distance_estimate': obstacle_result['distance_estimate'],
                        'mode': mode,
                        'speed': speed
                    })
                
                # Create combined visualization
                display_frame = self._create_full_display(
                    frame, lane_result, obstacle_result, mode, fps, latency_ms
                )
                
                cv2.imshow('Full Integration Test', display_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                    
                elif key == ord(' '):  # Toggle autonomous
                    if not self.remote_override.is_active():
                        self.autonomous_active = not self.autonomous_active
                        status = "STARTED" if self.autonomous_active else "STOPPED"
                        print(f"\n[AUTO] {status}")
                        if self.logger:
                            self.logger.log_event('mode_change', f'Autonomous {status}')
                    else:
                        print("[WARN] Disable override first!")
                        
                elif key == ord('o'):  # Toggle override
                    if self.remote_override.is_active():
                        self.remote_override.deactivate_override()
                        self.autonomous_active = False
                    else:
                        self.remote_override.activate_override()
                        self.autonomous_active = False
                        
                elif key == 27:  # ESC - Emergency stop
                    print("\n[EMERGENCY STOP] User initiated")
                    self.autonomous_active = False
                    self.remote_override.emergency_stop()
                    if not self.simulation_mode:
                        self.motor_control.emergency_stop()
                    if self.logger:
                        self.logger.log_event('emergency_stop', 'User triggered (ESC key)')
                
                # Manual control keys (only when override active)
                elif self.remote_override.is_active():
                    if key == ord('w'):
                        self.remote_override.process_manual_command('forward')
                    elif key == ord('s'):
                        self.remote_override.process_manual_command('backward')
                    elif key == ord('a'):
                        self.remote_override.process_manual_command('left')
                    elif key == ord('d'):
                        self.remote_override.process_manual_command('right')
                    elif key == ord(' '):
                        self.remote_override.process_manual_command('stop')
        
        except KeyboardInterrupt:
            print("\n[INTERRUPT] Test stopped by user")
        except Exception as e:
            print(f"\n[ERROR] Integration test exception: {e}")
            if self.logger:
                self.logger.log_event('error', f"Integration test exception: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            self.running = False
            
            # Safe shutdown
            if not self.simulation_mode and self.motor_control:
                self.motor_control.emergency_stop()
            self.remote_override.emergency_stop()
            
            cap.release()
            cv2.destroyAllWindows()
            
            print("\n[SHUTDOWN] Integration test complete")
            
            # Print all subsystem statistics
            print("\n" + "="*60)
            print("SESSION STATISTICS")
            print("="*60)
            
            self.lane_detector.print_statistics()
            self.obstacle_detector.print_statistics()
            self.remote_override.print_statistics()
            
            if not self.simulation_mode:
                self.motor_control.print_statistics()
            
            # Save logger summary
            if self.logger:
                self.logger.save_summary()
    
    def _create_full_display(self, frame: np.ndarray,
                            lane_result: Dict, obstacle_result: Dict,
                            mode: str, fps: float, latency_ms: float) -> np.ndarray:
        """
        Create combined display with all system information.
        
        Args:
            frame: Original camera frame
            lane_result: Lane detection results
            obstacle_result: Obstacle detection results
            mode: Current system mode
            fps: Current frame rate
            latency_ms: Processing latency
            
        Returns:
            Annotated display frame
        """
        # Start with lane detection overlay
        if lane_result['debug_frame'] is not None:
            display = lane_result['debug_frame'].copy()
        else:
            display = frame.copy()
        
        # Overlay obstacle detection visualization
        if obstacle_result['debug_frame'] is not None:
            # Blend obstacle zone visualization
            mask = np.all(obstacle_result['debug_frame'] == frame, axis=2)
            display[~mask] = obstacle_result['debug_frame'][~mask]
        
        height, width = display.shape[:2]
        
        # Top overlay - System metrics
        overlay = display.copy()
        cv2.rectangle(overlay, (10, 10), (450, 140), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)
        
        y = 35
        metrics = [
            f"Mode: {mode}",
            f"FPS: {fps:.1f} | Latency: {latency_ms:.1f}ms",
            f"Lane Conf: {lane_result['confidence']:.2f}",
            f"Obstacle: {'YES' if obstacle_result['obstacle_detected'] else 'NO'}"
        ]
        
        for text in metrics:
            cv2.putText(display, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (255, 255, 255), 2)
            y += 25
        
        # Bottom - Mode indicator
        mode_color = (0, 255, 0) if mode == "AUTONOMOUS" else (0, 0, 255)
        if mode == "OBSTACLE STOP":
            mode_color = (0, 165, 255)  # Orange
        elif mode == "MANUAL":
            mode_color = (255, 0, 255)  # Magenta
        
        cv2.putText(display, mode, (10, height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, mode_color, 3)
        
        return display
    
    def _draw_vision_metrics(self, frame: np.ndarray, latency_ms: float,
                            fps: float, result: Dict):
        """Draw metrics overlay for vision testing"""
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 160), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        y = 35
        
        # Latency with pass/fail indicator
        latency_color = (0, 255, 0) if latency_ms < config.LATENCY_TARGET_MS else (0, 0, 255)
        latency_status = "✓ PASS" if latency_ms < config.LATENCY_TARGET_MS else "✗ FAIL"
        cv2.putText(frame, f"Latency: {latency_ms:.1f}ms {latency_status}", 
                   (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, latency_color, 2)
        y += 25
        
        # Other metrics
        metrics = [
            f"FPS: {fps:.1f}",
            f"Steering: {result['steering_angle']:+4d} deg",
            f"Offset: {result['lane_offset']:+4d}px",
            f"Confidence: {result['confidence']:.2f}"
        ]
        
        for text in metrics:
            cv2.putText(frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (255, 255, 255), 2)
            y += 25
    
    def _print_vision_summary(self, latencies: list, steering_angles: list,
                             confidences: list, fps: float):
        """Print comprehensive vision test summary"""
        print("\n" + "="*60)
        print("LANE DETECTION TEST SUMMARY")
        print("="*60)
        
        if not latencies or len(latencies) == 0:
            print("No data collected")
            return
        
        # 1. Latency Analysis (NFR-P1)
        avg_latency = np.mean(latencies)
        max_latency = np.max(latencies)
        min_latency = np.min(latencies)
        latency_pass = avg_latency < config.LATENCY_TARGET_MS
        
        print(f"\n[1] LATENCY PERFORMANCE (NFR-P1: < {config.LATENCY_TARGET_MS}ms)")
        print(f"    Average: {avg_latency:6.2f}ms {'✓ PASS' if latency_pass else '✗ FAIL'}")
        print(f"    Min:     {min_latency:6.2f}ms")
        print(f"    Max:     {max_latency:6.2f}ms")
        print(f"    Std Dev: {np.std(latencies):6.2f}ms")
        
        if latency_pass:
            improvement = config.LATENCY_TARGET_MS / avg_latency
            print(f"    Performance: {improvement:.1f}× better than requirement")
        
        # 2. Frame Rate (NFR-P2)
        fps_pass = fps >= config.MIN_FPS
        print(f"\n[2] FRAME RATE (NFR-P2: ≥ {config.MIN_FPS} FPS)")
        print(f"    FPS: {fps:.1f} {'✓ PASS' if fps_pass else '✗ FAIL'}")
        
        if fps_pass:
            improvement = fps / config.MIN_FPS
            print(f"    Performance: {improvement:.1f}× better than requirement")
        
        # 3. Steering Behavior
        avg_steering = np.mean(np.abs(steering_angles))
        steering_smoothness = np.sum(np.abs(np.diff(steering_angles)))
        
        print(f"\n[3] STEERING BEHAVIOR")
        print(f"    Avg magnitude: {avg_steering:.1f} deg")
        print(f"    Total changes: {steering_smoothness:.1f} deg (lower = smoother)")
        print(f"    Max steering:  {np.max(np.abs(steering_angles)):.1f} deg")
        
        # 4. Detection Reliability
        avg_confidence = np.mean(confidences)
        low_conf_frames = np.sum(np.array(confidences) < 0.5)
        low_conf_pct = (low_conf_frames / len(confidences)) * 100
        
        print(f"\n[4] DETECTION RELIABILITY")
        print(f"    Avg confidence:        {avg_confidence:.2f}")
        print(f"    Low confidence (<0.5): {low_conf_frames} frames ({low_conf_pct:.1f}%)")
        
        # 5. Overall Assessment
        passed_tests = sum([
            latency_pass,
            fps_pass,
            avg_confidence > 0.5
        ])
        
        print(f"\n[5] OVERALL ASSESSMENT")
        print(f"    Tests passed: {passed_tests}/3")
        
        if passed_tests == 3:
            print("    Status: ✓ READY FOR INTEGRATION")
        elif passed_tests >= 2:
            print("    Status: ⚠ NEEDS TUNING")
        else:
            print("    Status: ✗ REQUIRES OPTIMIZATION")
        
        print("="*60 + "\n")


# ============================================================================
# MAIN MENU & ENTRY POINT
# ============================================================================

def print_menu():
    """Display main testing menu"""
    print("\n" + "="*60)
    print("AUTONOMOUS VEHICLE - TESTING SUITE v2.0")
    print("Student: Benyamin Mahamed (W1966430)")
    print("Target: Jonathan (77) - Assisted Mobility Platform")
    print("="*60)
    print("\n1. Heartbeat Test (Hardware validation)")
    print("2. Vision Test (Lane detection only)")
    print("3. Integration Test - SIMULATION (All systems)")
    print("4. Integration Test - LIVE MOTORS (Full autonomous)")
    print("5. Exit")
    print()


def main():
    """
    Main entry point for autonomous vehicle testing system.
    """
    # Ensure stdout is flushed for SSH stability
    sys.stdout.flush()

    print("\n" + "="*60)
    print("AUTONOMOUS VEHICLE TESTING SYSTEM")
    print("Enhanced v2.0 with Comprehensive Logging")
    print("="*60)
    print("\nStudent: Benyamin Mahamed (W1966430)")
    print("Project: Autonomous Self-Driving Car for Assisted Mobility")
    print("Target User: Jonathan (77) - Wheelchair User")
    print("="*60 + "\n")
    
    # START WITH BOTH AS NONE: This keeps the terminal responsive and the GPIO pins free
    vehicle_sim = None
    vehicle_live = None
    
    while True:
        # Flush the buffer to ensure the terminal is ready for fresh input
        if sys.stdin.isatty():
            try:
                import termios
                termios.tcflush(sys.stdin, termios.TCIFLUSH)
            except ImportError:
                pass

        print_menu()
        choice = input("Select test (1-5): ").strip()
        
        if not choice:
            continue

        if choice == '1':
            print("\n[INFO] Initializing hardware interface...")
            # If sim was running, kill it first
            if vehicle_sim is not None:
                vehicle_sim = None
                time.sleep(0.2)
            if vehicle_live is None:
                vehicle_live = AutonomousVehicle(simulation_mode=False, enable_logging=True)
            vehicle_live.run_heartbeat_test()
            
        elif choice == '2' or choice == '3':
            # Initialize SIM mode ONLY when Choice 2 or 3 is selected
            if vehicle_sim is None:
                print("\n[INFO] Initializing simulation subsystems...")
                vehicle_sim = AutonomousVehicle(simulation_mode=True, enable_logging=True)
            
            if choice == '2':
                vehicle_sim.run_vision_test()
            else:
                vehicle_sim.run_integration_test()
            
        elif choice == '4':
            print("\n" + "="*60)
            print("⚠ WARNING: LIVE MOTOR MODE")
            print("="*60)
            print("\nRequirements:")
            print("  - Vehicle must be on track with clear lane markings")
            print("  - Clear path ahead (no obstacles)")
            print("  - Emergency stop accessible (ESC key)")
            print("  - Adequate lighting for camera")
            print("\nSafety:")
            print("  - Press ESC for immediate emergency stop")
            print("  - Press 'o' to activate manual override")
            print("  - Press 'q' to quit safely")
            
            confirm = input("\nType 'CONFIRM' to proceed: ").strip()
            
            if confirm == 'CONFIRM':
                # KILL SIMULATION FIRST: This releases GPIO23
                if vehicle_sim is not None:
                    print("\n[INFO] Releasing simulation hardware resources...")
                    vehicle_sim = None 
                    time.sleep(0.5) # Allow hardware bus to settle
                
                print("[INFO] Initializing hardware interface...")
                if vehicle_live is None:
                    vehicle_live = AutonomousVehicle(simulation_mode=False, enable_logging=True)
                vehicle_live.run_integration_test()
            else:
                print("\n[ABORTED] Live motor test cancelled")
                
        elif choice == '5':
            print("\n" + "="*60)
            print("SHUTTING DOWN")
            print("="*60)
            if vehicle_live and vehicle_live.motor_control:
                vehicle_live.motor_control.emergency_stop()
            break
            
        else:
            print(f"\n[ERROR] Invalid choice '{choice}'. Please select 1-5.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n" + "="*60)
        print("EMERGENCY SHUTDOWN")
        print("="*60)
        print("\n[INTERRUPT] Keyboard interrupt detected")
        print("[SAFETY] Terminating all processes...\n")
        sys.exit(0)
    except Exception as e:
        print("\n\n" + "="*60)
        print("FATAL ERROR")
        print("="*60)
        print(f"\n[ERROR] {e}\n")
        import traceback
        traceback.print_exc()
        print("\n[CRITICAL] System terminated due to unhandled exception\n")
        sys.exit(1)
