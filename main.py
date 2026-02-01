# main.py - Autonomous Vehicle Testing & Integration
# Student: Benyamin Mahamed (W1966430)
# Project: Autonomous Self-Driving Car with Remote Override

from control_logic import RobotMuscle
from lane_detection import LaneDetector
from object_detection import ObstacleDetector
from remote_override import RemoteOverride
import cv2
import time
import config
import sys
import numpy as np


class Camera:
    """Camera wrapper using Picamera2 for Raspberry Pi"""
    
    def __init__(self, width=640, height=480):
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
    
    def is_opened(self):
        return self.camera is not None
    
    def read(self):
        if self.camera is None:
            return False, None
        
        try:
            frame = self.camera.capture_array()
            # Convert RGB to BGR for OpenCV
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            return True, frame
        except Exception as e:
            print(f"[CAMERA] Read error: {e}")
            return False, None
    
    def release(self):
        if self.camera:
            try:
                self.camera.stop()
                self.camera.close()
                print("[CAMERA] Released")
            except:
                pass


class AutonomousVehicle:
    """Main system integrating vision, obstacle detection, and control"""
    
    def __init__(self, simulation_mode=False):
        """
        Args:
            simulation_mode: If True, processes vision but doesn't send motor commands
        """
        self.simulation_mode = simulation_mode
        self.lane_detector = LaneDetector()
        self.obstacle_detector = ObstacleDetector()
        self.remote_override = RemoteOverride()
        
        if not simulation_mode:
            self.motor_control = RobotMuscle()
            print("[INIT] Full system initialized - MOTORS ACTIVE")
        else:
            self.motor_control = None
            print("[INIT] Simulation mode - motors disabled")
        
        self.running = False
        self.autonomous_active = False
    
    def _open_camera(self):
        """
        Open camera using Picamera2
        Returns: (camera_object, success_boolean)
        """
        print("\n[CAMERA] Opening camera...")
        
        cap = Camera(config.CAMERA_WIDTH, config.CAMERA_HEIGHT)
        
        if cap.is_opened():
            # Test capture
            ret, test_frame = cap.read()
            if ret and test_frame is not None:
                print(f"[CAMERA] Test frame OK: {test_frame.shape}")
                return cap, True
            else:
                print("[CAMERA] Failed to capture test frame")
                cap.release()
        
        print("[CAMERA] Failed to initialize")
        return None, False
    
    def run_heartbeat_test(self):
        """Hardware validation - servos and motors (FR1.2)"""
        if self.simulation_mode:
            print("[SKIP] Heartbeat test (simulation mode)")
            return
        
        print("\n" + "="*60)
        print("HEARTBEAT TEST - Hardware Validation")
        print("="*60)
        
        try:
            print("\n[1/2] Testing camera servos...")
            self.motor_control.test_servos()
            
            input("Press ENTER to test motors (ensure clear space)...")
            print("\n[2/2] Testing motors...")
            self.motor_control.test_motors()
            
            print("\n✓ Heartbeat test complete - hardware operational")
            
        except Exception as e:
            print(f"\n✗ Heartbeat test failed: {e}")
            self.motor_control.emergency_stop()
    
    def run_vision_test(self):
        """
        Vision-only testing - Lane Detection Algorithm Validation
        Tests FR1.1 and NFR-P1 (latency < 200ms)
        """
        print("\n" + "="*60)
        print("VISION TEST - Lane Detection (Classical CV)")
        print("="*60)
        print("\nThis test validates:")
        print("  - FR1.1: Lane boundary detection")
        print("  - NFR-P1: Latency < 200ms")
        print("\nControls:")
        print("  'q' - Quit test")
        print("  's' - Save current frame")
        print("  'p' - Pause/Resume")
        
        # Open camera
        cap, success = self._open_camera()
        if not success:
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
    
    def run_integration_test(self):
        """
        Full Integration Test with ALL safety features:
        - Lane detection
        - Obstacle detection  
        - Remote override capability
        - Emergency stop
        """
        print("\n" + "="*60)
        print(f"FULL INTEGRATION TEST - {'SIMULATION' if self.simulation_mode else 'LIVE MOTORS'}")
        print("="*60)
        
        if not self.simulation_mode:
            print("\n WARNING: Motors will move!")
            print("Ensure vehicle is on track with clear path")
            response = input("Continue? (yes/no): ")
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
                
                # Lane detection
                lane_result = self.lane_detector.process_frame(frame)
                
                # Obstacle detection
                obstacle_result = self.obstacle_detector.detect_obstacle(frame)
                
                latency_ms = (time.time() - start_time) * 1000
                
                # DECISION LOGIC (FR2.2 - Data Fusion with Priority)
                if self.remote_override.is_active():
                    # MANUAL MODE - user has full control
                    mode = "MANUAL"
                    # Manual commands handled by keyboard below
                    
                elif obstacle_result['obstacle_detected']:
                    # SAFETY PRIORITY - Obstacle detected!
                    mode = "OBSTACLE STOP"
                    if not self.simulation_mode:
                        self.motor_control.emergency_stop()
                    print(f"[SAFETY] Obstacle at {obstacle_result['distance_estimate']}cm - STOPPING")
                    
                elif self.autonomous_active:
                    # AUTONOMOUS MODE - lane following
                    mode = "AUTONOMOUS"
                    steering_angle = lane_result['steering_angle']
                    
                    # Speed based on lane confidence
                    if lane_result['confidence'] > 0.5:
                        speed = config.BASE_SPEED
                    elif lane_result['confidence'] > 0.3:
                        speed = config.MIN_SPEED
                    else:
                        speed = 0  # Stop if no lane
                    
                    if not self.simulation_mode:
                        self.motor_control.execute_motion(speed, steering_angle)
                    else:
                        if frame_count % 15 == 0:
                            print(f"[AUTO] Speed: {speed:3d} | Steer: {steering_angle:+4d}deg")
                else:
                    # STOPPED
                    mode = "STOPPED"
                    if not self.simulation_mode:
                        self.motor_control.emergency_stop()
                
                # Calculate FPS
                elapsed = time.time() - fps_start
                fps = frame_count / elapsed if elapsed > 0 else 0.0
                
                # Create combined visualization
                display_frame = self._create_full_display(
                    frame, lane_result, obstacle_result, mode, fps, latency_ms
                )
                
                cv2.imshow('Full Integration Test', display_frame)
                
                # Handle keyboard
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                    
                elif key == ord(' '):  # Toggle autonomous
                    if not self.remote_override.is_active():
                        self.autonomous_active = not self.autonomous_active
                        status = "STARTED" if self.autonomous_active else "STOPPED"
                        print(f"\n[AUTO] {status}")
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
                    print("\n[EMERGENCY STOP]")
                    self.autonomous_active = False
                    self.remote_override.emergency_stop()
                    if not self.simulation_mode:
                        self.motor_control.emergency_stop()
                
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
            import traceback
            traceback.print_exc()
        
        finally:
            self.running = False
            if not self.simulation_mode and self.motor_control:
                self.motor_control.emergency_stop()
            self.remote_override.emergency_stop()
            cap.release()
            cv2.destroyAllWindows()
            print("\n[SHUTDOWN] Integration test complete")
    
    def _create_full_display(self, frame, lane_result, obstacle_result, mode, fps, latency_ms):
        """Create combined display with all system info"""
        # Start with lane detection overlay
        if lane_result['debug_frame'] is not None:
            display = lane_result['debug_frame'].copy()
        else:
            display = frame.copy()
        
        # Add obstacle detection overlay
        if obstacle_result['debug_frame'] is not None:
            # Blend obstacle zone from obstacle debug frame
            pass  # Obstacle detector already draws on frame
        
        height, width = display.shape[:2]
        
        # Top overlay - System metrics
        overlay = display.copy()
        cv2.rectangle(overlay, (10, 10), (400, 140), (0, 0, 0), -1)
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
        
        cv2.putText(display, mode, (10, height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, mode_color, 3)
        
        return display
    
    def _draw_vision_metrics(self, frame, latency_ms, fps, result):
        """Draw metrics overlay for vision testing"""
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 160), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        y = 35
        
        # Latency with pass/fail indicator
        latency_color = (0, 255, 0) if latency_ms < config.LATENCY_TARGET_MS else (0, 0, 255)
        latency_status = "PASS" if latency_ms < config.LATENCY_TARGET_MS else "FAIL"
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
    
    def _print_vision_summary(self, latencies, steering_angles, confidences, fps):
        """Print comprehensive test summary"""
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
        print(f"    Average: {avg_latency:6.2f}ms {'PASS' if latency_pass else 'FAIL'}")
        print(f"    Min:     {min_latency:6.2f}ms")
        print(f"    Max:     {max_latency:6.2f}ms")
        print(f"    Std Dev: {np.std(latencies):6.2f}ms")
        
        # 2. Frame Rate
        print(f"\n[2] FRAME RATE")
        print(f"    FPS: {fps:.1f}")
        
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
            avg_confidence > 0.5,
            fps > 15
        ])
        
        print(f"\n[5] OVERALL ASSESSMENT")
        print(f"    Tests passed: {passed_tests}/3")
        
        if passed_tests == 3:
            print("    Status: READY FOR INTEGRATION")
        elif passed_tests >= 2:
            print("    Status: NEEDS TUNING")
        else:
            print("    Status: REQUIRES OPTIMIZATION")
        
        print("="*60 + "\n")


def print_menu():
    """Display main menu"""
    print("\n" + "="*60)
    print("AUTONOMOUS VEHICLE - TESTING SUITE")
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
    # Default to simulation mode for safety
    vehicle_sim = AutonomousVehicle(simulation_mode=True)
    vehicle_live = None
    
    while True:
        print_menu()
        choice = input("Select test (1-5): ").strip()
        
        if choice == '1':
            # Heartbeat requires real hardware
            if vehicle_live is None:
                vehicle_live = AutonomousVehicle(simulation_mode=False)
            vehicle_live.run_heartbeat_test()
            
        elif choice == '2':
            # Vision-only test (safe - no motors)
            vehicle_sim.run_vision_test()
            
        elif choice == '3':
            # Integration in simulation mode
            vehicle_sim.run_integration_test()
            
        elif choice == '4':
            # Full system with motors
            print("\n WARNING: Motors will move!")
            print("Requirements:")
            print("  - Vehicle must be on track")
            print("  - Clear path ahead")
            print("  - Emergency stop accessible")
            confirm = input("\nType 'CONFIRM' to proceed: ")
            
            if confirm == 'CONFIRM':
                if vehicle_live is None:
                    vehicle_live = AutonomousVehicle(simulation_mode=False)
                vehicle_live.run_integration_test()
            else:
                print("Aborted.")
                
        elif choice == '5':
            print("\nShutting down...")
            if vehicle_live and vehicle_live.motor_control:
                vehicle_live.motor_control.emergency_stop()
            print("Goodbye!")
            break
            
        else:
            print("Invalid choice. Please select 1-5.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[INTERRUPT] Emergency shutdown initiated")
        sys.exit(0)
    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)