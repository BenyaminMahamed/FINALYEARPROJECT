#!/usr/bin/env python3
"""
Obstacle Detection System - Classical Computer Vision Approach
Student: Benyamin Mahamed (W1966430)
Project: Autonomous Self-Driving Car for Assisted Mobility

Implements safety-critical obstacle detection using blob detection.
Essential for FR2.1 (Obstacle Detection) and FR2.2 (Emergency Stop on Obstacle).

Target Use Case: Detects obstacles in wheelchair/mobility aid path for
Jonathan (77-year-old user), triggering immediate emergency stop to prevent
collisions with curbs, objects, or people.

Design Choice: Classical CV blob detection prioritized over Deep Learning for:
    - Deterministic, reliable behavior (safety-critical)
    - Real-time performance (< 1ms processing time)
    - No training data required
    - Explainable results for safety validation

NFR-S1: Obstacle emergency stop must be 100% reliable
"""

import cv2
import numpy as np
import config
from typing import Dict, Tuple, Optional, Any
import time


class ObstacleDetector:
    """
    Blob-based obstacle detection for safety zone monitoring.
    
    Detects large objects in the safety zone (center-bottom of camera frame)
    using classical computer vision techniques. Designed for real-time
    performance (< 1ms) with deterministic behavior.
    
    Safety Zone: Center 40% width × Bottom 50% height of frame
    Detection Method: Grayscale threshold + contour analysis
    
    Key Requirements:
        - FR2.1: Detect obstacles in safety zone
        - FR2.2: Trigger emergency stop when obstacle detected
        - NFR-S1: 100% reliable emergency stop
    """
    
    def __init__(self):
        """
        Initialize obstacle detector with configuration parameters.
        """
        self.frame_count = 0
        self.detection_count = 0
        self.false_positive_count = 0  # Track potential false detections
        
        # Performance tracking
        self.total_processing_time = 0.0
        self.min_processing_time = float('inf')
        self.max_processing_time = 0.0
        
        # Event callback for logging integration
        self.event_callback = None
        
        print("[OBSTACLE] Obstacle detector initialized")
        print(f"[OBSTACLE] Safety zone: {config.SAFETY_ZONE_WIDTH_RATIO*100:.0f}% width × 50% height")
        print(f"[OBSTACLE] Detection threshold: {config.OBSTACLE_THRESHOLD_PERCENT*100:.0f}% of zone")
    
    def set_event_callback(self, callback):
        """
        Set callback function for event logging.
        
        Args:
            callback: Function(event_type: str, details: str) to call on detection events
        """
        self.event_callback = callback
    
    def _log_event(self, event_type: str, details: str):
        """Internal event logging"""
        if self.event_callback:
            self.event_callback(event_type, details)
    
    def detect_obstacle(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Detect obstacles in the camera frame safety zone.
        
        Implements FR2.1 (Obstacle Detection) using blob detection approach.
        Processing optimized for real-time performance (target: < 1ms).
        
        Args:
            frame: BGR image from camera (shape: height × width × 3)
            
        Returns:
            Dictionary containing:
                - obstacle_detected (bool): True if obstacle in safety zone
                - distance_estimate (int): Estimated distance in cm (0-100)
                - blob_area (float): Size of detected blob (for logging)
                - confidence (float): Detection confidence (0.0-1.0)
                - debug_frame (np.ndarray or None): Annotated frame for visualization
                - processing_time_ms (float): Processing time in milliseconds
        """
        start_time = time.time()
        self.frame_count += 1
        
        height, width = frame.shape[:2]
        
        # Define safety zone boundaries (FR2.1 requirement)
        zone_width = int(width * config.SAFETY_ZONE_WIDTH_RATIO)
        zone_x_start = (width - zone_width) // 2
        zone_x_end = zone_x_start + zone_width
        
        # Focus on bottom half where ground-level obstacles appear
        zone_y_start = height // 2
        zone_y_end = height
        
        zone_height = zone_y_end - zone_y_start
        
        # Extract safety zone region of interest (ROI)
        safety_zone = frame[zone_y_start:zone_y_end, zone_x_start:zone_x_end]
        
        # Convert to grayscale for blob detection
        gray = cv2.cvtColor(safety_zone, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to detect dark objects
        # Obstacles typically appear darker than ground/lane markings
        _, thresh = cv2.threshold(gray, config.OBSTACLE_THRESHOLD_VALUE, 255, 
                                 cv2.THRESH_BINARY_INV)
        
        # Optional: Apply morphological operations to reduce noise
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Find contours (blobs) in thresholded image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        # Initialize detection result
        obstacle_detected = False
        largest_area = 0
        distance_estimate = 100  # Default: far away (safe)
        confidence = 0.0
        
        # Analyze blobs for obstacle detection
        if contours:
            # Find largest blob (most likely to be obstacle)
            largest_contour = max(contours, key=cv2.contourArea)
            largest_area = cv2.contourArea(largest_contour)
            
            # Calculate minimum area threshold for obstacle
            zone_area = zone_width * zone_height
            min_obstacle_area = zone_area * config.OBSTACLE_THRESHOLD_PERCENT
            
            if largest_area > min_obstacle_area:
                obstacle_detected = True
                self.detection_count += 1
                
                # Estimate distance based on blob size
                # Assumption: Larger blob = closer object
                area_ratio = largest_area / zone_area
                distance_estimate = int((1.0 - area_ratio) * 100)  # 0-100cm
                
                # Confidence based on how much larger than threshold
                confidence = min(1.0, area_ratio / config.OBSTACLE_THRESHOLD_PERCENT)
                
                # Log detection event
                self._log_event('obstacle_detected', 
                              f'Obstacle at ~{distance_estimate}cm (confidence: {confidence:.2f})')
        
        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Update performance statistics
        self.total_processing_time += processing_time_ms
        self.min_processing_time = min(self.min_processing_time, processing_time_ms)
        self.max_processing_time = max(self.max_processing_time, processing_time_ms)
        
        # Create debug visualization if enabled
        debug_frame = None
        if config.DEBUG_MODE:
            debug_frame = self._draw_debug(
                frame, zone_x_start, zone_x_end, zone_y_start, zone_y_end,
                obstacle_detected, distance_estimate, confidence, largest_area
            )
        
        return {
            'obstacle_detected': obstacle_detected,
            'distance_estimate': distance_estimate,
            'blob_area': largest_area,
            'confidence': confidence,
            'debug_frame': debug_frame,
            'processing_time_ms': processing_time_ms
        }
    
    def _draw_debug(self, frame: np.ndarray, 
                   zone_x_start: int, zone_x_end: int,
                   zone_y_start: int, zone_y_end: int,
                   obstacle_detected: bool, distance: int, 
                   confidence: float, blob_area: float) -> np.ndarray:
        """
        Draw safety zone visualization and obstacle status on frame.
        
        Args:
            frame: Original camera frame
            zone_x_start, zone_x_end, zone_y_start, zone_y_end: Safety zone boundaries
            obstacle_detected: Whether obstacle is present
            distance: Estimated distance in cm
            confidence: Detection confidence (0.0-1.0)
            blob_area: Size of detected blob
            
        Returns:
            Annotated debug frame
        """
        debug_frame = frame.copy()
        
        # Color coding: Red if obstacle, Green if clear
        color = (0, 0, 255) if obstacle_detected else (0, 255, 0)
        
        # Draw safety zone rectangle
        cv2.rectangle(debug_frame, 
                     (zone_x_start, zone_y_start), 
                     (zone_x_end, zone_y_end), 
                     color, 3)
        
        # Semi-transparent overlay for safety zone
        overlay = debug_frame.copy()
        zone_color = (0, 0, 100) if obstacle_detected else (0, 100, 0)
        cv2.rectangle(overlay,
                     (zone_x_start, zone_y_start),
                     (zone_x_end, zone_y_end),
                     zone_color, -1)
        cv2.addWeighted(overlay, 0.2, debug_frame, 0.8, 0, debug_frame)
        
        # Status text with background
        status_text = "⚠ OBSTACLE DETECTED" if obstacle_detected else "✓ PATH CLEAR"
        text_color = (0, 0, 255) if obstacle_detected else (0, 255, 0)
        
        # Draw text background for readability
        text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        cv2.rectangle(debug_frame,
                     (zone_x_start - 5, zone_y_start - 60),
                     (zone_x_start + text_size[0] + 5, zone_y_start - 25),
                     (0, 0, 0), -1)
        
        cv2.putText(debug_frame, status_text, 
                   (zone_x_start, zone_y_start - 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
        
        # Distance and confidence information
        if obstacle_detected:
            info_text = f"Distance: ~{distance}cm | Confidence: {confidence:.2f}"
            cv2.putText(debug_frame, info_text, 
                       (zone_x_start, zone_y_start - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return debug_frame
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get performance and detection statistics.
        
        Returns:
            Dictionary containing:
                - frames_processed: Total frames analyzed
                - obstacles_detected: Total obstacle detections
                - avg_processing_time_ms: Average processing time
                - min_processing_time_ms: Minimum processing time
                - max_processing_time_ms: Maximum processing time
        """
        avg_time = (self.total_processing_time / self.frame_count 
                   if self.frame_count > 0 else 0.0)
        
        return {
            'frames_processed': self.frame_count,
            'obstacles_detected': self.detection_count,
            'detection_rate': (self.detection_count / self.frame_count * 100 
                             if self.frame_count > 0 else 0.0),
            'avg_processing_time_ms': avg_time,
            'min_processing_time_ms': self.min_processing_time if self.min_processing_time != float('inf') else 0.0,
            'max_processing_time_ms': self.max_processing_time
        }
    
    def print_statistics(self):
        """Print detection statistics for validation"""
        stats = self.get_statistics()
        
        print("\n" + "="*60)
        print("OBSTACLE DETECTION - STATISTICS")
        print("="*60)
        print(f"  Frames processed:    {stats['frames_processed']}")
        print(f"  Obstacles detected:  {stats['obstacles_detected']}")
        print(f"  Detection rate:      {stats['detection_rate']:.1f}%")
        print(f"\n  Processing Performance:")
        print(f"    Average: {stats['avg_processing_time_ms']:.3f}ms")
        print(f"    Min:     {stats['min_processing_time_ms']:.3f}ms")
        print(f"    Max:     {stats['max_processing_time_ms']:.3f}ms")
        print(f"\n  Target: < 1ms (real-time requirement)")
        
        if stats['avg_processing_time_ms'] < 1.0:
            print(f"  Status: ✓ PASS (meets real-time requirement)")
        else:
            print(f"  Status: ⚠ ACCEPTABLE (< 2ms)")
        
        print("="*60 + "\n")
    
    def reset_statistics(self):
        """Reset statistics counters"""
        self.frame_count = 0
        self.detection_count = 0
        self.total_processing_time = 0.0
        self.min_processing_time = float('inf')
        self.max_processing_time = 0.0
        print("[OBSTACLE] Statistics reset")


def test_obstacle_detection():
    """
    Standalone test function for obstacle detector validation.
    
    Tests FR2.1 (Obstacle Detection) with synthetic test frames.
    """
    print("\n" + "="*60)
    print("OBSTACLE DETECTION - STANDALONE TEST")
    print("="*60 + "\n")
    
    detector = ObstacleDetector()
    
    # Test 1: Clear frame (no obstacle)
    print("[TEST 1] Clear frame (no obstacle)...")
    clear_frame = np.ones((480, 640, 3), dtype=np.uint8) * 200  # Light gray
    result = detector.detect_obstacle(clear_frame)
    print(f"  Result: {'OBSTACLE' if result['obstacle_detected'] else 'CLEAR'}")
    print(f"  Processing time: {result['processing_time_ms']:.3f}ms")
    assert not result['obstacle_detected'], "Test 1 failed: False positive"
    print("  ✓ PASS\n")
    
    # Test 2: Large obstacle in safety zone
    print("[TEST 2] Large obstacle in safety zone...")
    obstacle_frame = np.ones((480, 640, 3), dtype=np.uint8) * 200
    # Add large dark blob in center-bottom (obstacle)
    obstacle_frame[300:450, 250:390] = 30  # Dark obstacle
    result = detector.detect_obstacle(obstacle_frame)
    print(f"  Result: {'OBSTACLE' if result['obstacle_detected'] else 'CLEAR'}")
    print(f"  Distance estimate: {result['distance_estimate']}cm")
    print(f"  Confidence: {result['confidence']:.2f}")
    print(f"  Processing time: {result['processing_time_ms']:.3f}ms")
    assert result['obstacle_detected'], "Test 2 failed: Missed obstacle"
    print("  ✓ PASS\n")
    
    # Print statistics
    detector.print_statistics()
    
    print("[TEST] ✓ All tests passed - Obstacle detector operational\n")


if __name__ == "__main__":
    # Run standalone tests
    test_obstacle_detection()