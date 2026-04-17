#!/usr/bin/env python3
"""
Obstacle Detection System - Classical Computer Vision Approach
Student: Benyamin Mahamed (W1966430)
Project: Autonomous Self-Driving Car for Assisted Mobility

Implements safety-critical obstacle detection using blob detection.
Essential for FR2.1 (Obstacle Detection) and FR2.2 (Emergency Stop).

NFR-S1: Obstacle emergency stop must be 100% reliable.
Achieved Latency: < 1ms (Processed on Pi 5 CPU).
"""

import cv2
import numpy as np
import config # Uses the global config instance
from typing import Dict, Tuple, Optional, Any
import time


class ObstacleDetector:
    """
    Blob-based obstacle detection for safety zone monitoring.
    Safety Zone: Center 40% width × Bottom 50% height of frame.
    
    Design Choice: Priority 2 Safety Trigger.
    If an obstacle is detected here, it overrides Priority 3 (Autonomy).
    """
    
    def __init__(self):
        """Initialize detector using parameters from the global config."""
        self.frame_count = 0
        self.detection_count = 0
        
        # Performance tracking
        self.total_processing_time = 0.0
        self.min_processing_time = float('inf')
        self.max_processing_time = 0.0
        
        # Event callback for logging integration
        self.event_callback = None
        
        print("[OBSTACLE] Obstacle detector initialized")
        # Fixed: Accessing attributes from the config object
        print(f"[OBSTACLE] Safety zone: {config.SAFETY_ZONE_WIDTH_RATIO*100:.0f}% width")
        print(f"[OBSTACLE] Detection threshold: {config.OBSTACLE_THRESHOLD_PERCENT*100:.1f}% area")
    
    def set_event_callback(self, callback):
        self.event_callback = callback
    
    def _log_event(self, event_type: str, details: str):
        if self.event_callback:
            self.event_callback(event_type, details)
    
    def detect_obstacle(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Main detection pipeline. Target latency < 1ms.
        """
        start_time = time.time()
        self.frame_count += 1
        
        height, width = frame.shape[:2]
        
        # Define safety zone boundaries from config
        zone_width = int(width * config.SAFETY_ZONE_WIDTH_RATIO)
        zone_x_start = (width - zone_width) // 2
        zone_x_end = zone_x_start + zone_width
        
        # Focus on bottom 50% for ground obstacles
        zone_y_start = int(height * (1 - config.SAFETY_ZONE_HEIGHT_RATIO))
        zone_y_end = height
        zone_height = zone_y_end - zone_y_start
        
        # Extract ROI
        safety_zone = frame[zone_y_start:zone_y_end, zone_x_start:zone_x_end]
        
        # Preprocessing: Grayscale + Threshold
        gray = cv2.cvtColor(safety_zone, cv2.COLOR_BGR2GRAY)
        
        # Apply binary inverse threshold (detects dark objects on light ground)
        _, thresh = cv2.threshold(gray, config.OBSTACLE_THRESHOLD_VALUE, 255, 
                                 cv2.THRESH_BINARY_INV)
        
        # Reduce noise via morphological closing
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Contour detection
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, 
                                     cv2.CHAIN_APPROX_SIMPLE)
        
        obstacle_detected = False
        largest_area = 0
        distance_estimate = 100 
        confidence = 0.0
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            largest_area = cv2.contourArea(largest_contour)
            
            zone_area = zone_width * zone_height
            min_obstacle_area = zone_area * config.OBSTACLE_THRESHOLD_PERCENT
            
            # Deterministic Trigger: Large blob in zone = E-STOP
            if largest_area > min_obstacle_area:
                obstacle_detected = True
                self.detection_count += 1
                
                # Heuristic distance: Larger area = closer to lens
                area_ratio = largest_area / zone_area
                distance_estimate = int((1.0 - area_ratio) * 100)
                confidence = min(1.0, area_ratio / config.OBSTACLE_THRESHOLD_PERCENT)
                
                self._log_event('obstacle_detected', 
                              f'Area: {largest_area:.0f}px | Conf: {confidence:.2f}')
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Update stats
        self.total_processing_time += processing_time_ms
        self.min_processing_time = min(self.min_processing_time, processing_time_ms)
        self.max_processing_time = max(self.max_processing_time, processing_time_ms)
        
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
        
        debug_frame = frame.copy()
        # Red if hazardous, Green if clear
        color = (0, 0, 255) if obstacle_detected else (0, 255, 0)
        
        # Draw Safety Square
        cv2.rectangle(debug_frame, (zone_x_start, zone_y_start), 
                     (zone_x_end, zone_y_end), color, 3)
        
        status_text = "OBSTACLE STOP" if obstacle_detected else "PATH CLEAR"
        cv2.putText(debug_frame, status_text, (zone_x_start, zone_y_start - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return debug_frame

    def get_statistics(self) -> Dict[str, Any]:
        avg_time = (self.total_processing_time / self.frame_count 
                   if self.frame_count > 0 else 0.0)
        return {
            'frames_processed': self.frame_count,
            'obstacles_detected': self.detection_count,
            'avg_processing_time_ms': avg_time,
            'min_processing_time_ms': self.min_processing_time if self.min_processing_time != float('inf') else 0.0,
            'max_processing_time_ms': self.max_processing_time
        }
