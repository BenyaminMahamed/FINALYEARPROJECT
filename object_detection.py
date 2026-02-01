# object_detection.py - Simple Obstacle Detection
# Uses blob detection to identify obstacles in path

import cv2
import numpy as np
import config


class ObstacleDetector:
    """
    Simple obstacle detection using blob detection
    Detects large objects in the safety zone (center of frame)
    """
    
    def __init__(self):
        self.frame_count = 0
        print("[OBSTACLE] Detector initialized")
    
    def detect_obstacle(self, frame):
        """
        Detect obstacles in the safety zone
        
        Args:
            frame: BGR image from camera
            
        Returns:
            dict with keys: 'obstacle_detected', 'distance_estimate', 'debug_frame'
        """
        self.frame_count += 1
        height, width = frame.shape[:2]
        
        # Define safety zone (center portion of frame)
        zone_width = int(width * config.SAFETY_ZONE_WIDTH_RATIO)
        zone_x_start = (width - zone_width) // 2
        zone_x_end = zone_x_start + zone_width
        
        # Focus on bottom half (where obstacles would appear)
        zone_y_start = height // 2
        zone_y_end = height
        
        # Extract safety zone
        safety_zone = frame[zone_y_start:zone_y_end, zone_x_start:zone_x_end]
        
        # Convert to grayscale
        gray = cv2.cvtColor(safety_zone, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to detect dark objects (common for obstacles)
        _, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours (blobs)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        obstacle_detected = False
        largest_area = 0
        distance_estimate = 100  # Default: far away
        
        # Check for large blobs (potential obstacles)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            largest_area = cv2.contourArea(largest_contour)
            
            # Threshold: if blob is large enough, it's an obstacle
            min_obstacle_area = (zone_width * (zone_y_end - zone_y_start)) * 0.15  # 15% of zone
            
            if largest_area > min_obstacle_area:
                obstacle_detected = True
                
                # Estimate distance based on blob size (larger = closer)
                max_area = zone_width * (zone_y_end - zone_y_start)
                area_ratio = largest_area / max_area
                distance_estimate = int((1.0 - area_ratio) * 100)  # 0-100cm estimate
        
        # Create debug visualization
        debug_frame = self._draw_debug(frame, zone_x_start, zone_x_end, 
                                       zone_y_start, zone_y_end, 
                                       obstacle_detected, distance_estimate)
        
        return {
            'obstacle_detected': obstacle_detected,
            'distance_estimate': distance_estimate,
            'debug_frame': debug_frame if config.DEBUG_MODE else None
        }
    
    def _draw_debug(self, frame, zone_x_start, zone_x_end, zone_y_start, zone_y_end, 
                    obstacle_detected, distance):
        """Draw safety zone and obstacle status on frame"""
        debug_frame = frame.copy()
        
        # Draw safety zone rectangle
        color = (0, 0, 255) if obstacle_detected else (0, 255, 0)  # Red if obstacle, green if clear
        cv2.rectangle(debug_frame, 
                     (zone_x_start, zone_y_start), 
                     (zone_x_end, zone_y_end), 
                     color, 2)
        
        # Status text
        status_text = "OBSTACLE!" if obstacle_detected else "CLEAR"
        text_color = (0, 0, 255) if obstacle_detected else (0, 255, 0)
        
        cv2.putText(debug_frame, f"Status: {status_text}", 
                   (zone_x_start, zone_y_start - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
        
        cv2.putText(debug_frame, f"Est. Distance: {distance}cm", 
                   (zone_x_start, zone_y_start - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return debug_frame