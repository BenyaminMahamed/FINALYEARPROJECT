# lane_detection.py - Classical Computer Vision Module
# Implements FR1.1: Lane boundary detection using OpenCV

import cv2
import numpy as np
import config


class LaneDetector:
    """
    Classical CV lane detection using Hough Transform
    Optimized for low-latency performance on Raspberry Pi
    """
    
    def __init__(self):
        self.frame_count = 0
        self.last_left_lane = None
        self.last_right_lane = None
        print("[VISION-CV] Lane detector initialized")
    
    def process_frame(self, frame):
        """
        Main processing pipeline for lane detection
        
        Args:
            frame: BGR image from camera
            
        Returns:
            dict with keys: 'steering_angle', 'lane_offset', 'confidence', 'debug_frame'
        """
        self.frame_count += 1
        
        # Step 1: Apply ROI mask
        roi_frame = self._apply_roi(frame)
        
        # Step 2: Preprocessing
        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (config.BLUR_KERNEL_SIZE, config.BLUR_KERNEL_SIZE), 0)
        
        # Step 3: Edge detection
        edges = cv2.Canny(blurred, config.CANNY_LOW_THRESHOLD, config.CANNY_HIGH_THRESHOLD)
        
        # Step 4: Hough line detection
        lines = cv2.HoughLinesP(
            edges,
            rho=config.HOUGH_RHO,
            theta=np.pi / 180 * config.HOUGH_THETA,
            threshold=config.HOUGH_THRESHOLD,
            minLineLength=config.HOUGH_MIN_LINE_LENGTH,
            maxLineGap=config.HOUGH_MAX_LINE_GAP
        )
        
        # Step 5: Separate and average left/right lanes
        left_lane, right_lane = self._classify_lanes(lines, frame.shape)
        
        # Step 6: Calculate steering command
        steering_angle, lane_offset, confidence = self._calculate_steering(
            left_lane, right_lane, frame.shape
        )
        
        # Debug visualization
        debug_frame = self._draw_debug_overlay(frame, left_lane, right_lane, steering_angle)
        
        return {
            'steering_angle': steering_angle,
            'lane_offset': lane_offset,
            'confidence': confidence,
            'debug_frame': debug_frame if config.DEBUG_MODE else None
        }
    
    def _apply_roi(self, frame):
        """Apply region of interest mask to focus on road area"""
        height, width = frame.shape[:2]
        
        # Define ROI polygon (trapezoid focused on road)
        roi_top = int(height * config.ROI_TOP_RATIO)
        roi_bottom = int(height * config.ROI_BOTTOM_RATIO)
        
        mask = np.zeros_like(frame)
        polygon = np.array([[
            (0, roi_bottom),
            (0, roi_top),
            (width, roi_top),
            (width, roi_bottom)
        ]], np.int32)
        
        cv2.fillPoly(mask, polygon, (255, 255, 255))
        masked_frame = cv2.bitwise_and(frame, mask)
        
        return masked_frame
    
    def _classify_lanes(self, lines, frame_shape):
        """
        Separate detected lines into left and right lane boundaries
        Filters by slope to remove invalid lines
        """
        if lines is None:
            return self.last_left_lane, self.last_right_lane
        
        height, width = frame_shape[:2]
        left_lines = []
        right_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calculate slope
            if x2 - x1 == 0:
                continue
            slope = (y2 - y1) / (x2 - x1)
            
            # Filter by slope (remove near-horizontal/vertical)
            if abs(slope) < config.MIN_LANE_SLOPE or abs(slope) > config.MAX_LANE_SLOPE:
                continue
            
            # Classify as left or right based on slope and position
            if slope < 0 and x1 < width // 2:  # Left lane (negative slope)
                left_lines.append(line[0])
            elif slope > 0 and x1 > width // 2:  # Right lane (positive slope)
                right_lines.append(line[0])
        
        # Average the lines
        left_lane = self._average_lines(left_lines, frame_shape) if left_lines else self.last_left_lane
        right_lane = self._average_lines(right_lines, frame_shape) if right_lines else self.last_right_lane
        
        # Update memory for temporal consistency
        if left_lane is not None:
            self.last_left_lane = left_lane
        if right_lane is not None:
            self.last_right_lane = right_lane
        
        return left_lane, right_lane
    
    def _average_lines(self, lines, frame_shape):
        """Average multiple line segments into a single lane line"""
        if not lines:
            return None
        
        height, width = frame_shape[:2]
        x_coords = []
        y_coords = []
        
        for line in lines:
            x1, y1, x2, y2 = line
            x_coords.extend([x1, x2])
            y_coords.extend([y1, y2])
        
        # Fit line using least squares
        if len(x_coords) < 2:
            return None
        
        poly = np.polyfit(y_coords, x_coords, 1)
        
        # Extrapolate to ROI boundaries
        y1 = int(height * config.ROI_TOP_RATIO)
        y2 = height
        x1 = int(np.polyval(poly, y1))
        x2 = int(np.polyval(poly, y2))
        
        return (x1, y1, x2, y2)
    
    def _calculate_steering(self, left_lane, right_lane, frame_shape):
        """
        Calculate steering angle and lane offset from detected lanes
        
        Returns:
            steering_angle (degrees), lane_offset (pixels), confidence (0-1)
        """
        height, width = frame_shape[:2]
        
        # Default: go straight
        steering_angle = 0
        lane_offset = 0
        confidence = 0.0
        
        # Calculate lane center
        if left_lane is not None and right_lane is not None:
            # Both lanes detected - high confidence
            left_x = left_lane[2]  # Bottom x coordinate
            right_x = right_lane[2]
            lane_center = (left_x + right_x) // 2
            confidence = 1.0
            
        elif left_lane is not None:
            # Only left lane - medium confidence
            left_x = left_lane[2]
            lane_center = left_x + (width // 4)  # Estimate right lane
            confidence = 0.6
            
        elif right_lane is not None:
            # Only right lane - medium confidence
            right_x = right_lane[2]
            lane_center = right_x - (width // 4)  # Estimate left lane
            confidence = 0.6
        else:
            # No lanes detected - return default
            return steering_angle, lane_offset, confidence
        
        # Calculate offset from frame center
        frame_center = width // 2
        lane_offset = lane_center - frame_center
        
        # Convert offset to steering angle (proportional control)
        # Negative offset = lane center is left, turn right (positive angle)
        steering_angle = -int(lane_offset * config.STEER_KP)
        steering_angle = max(-config.MAX_STEER_ANGLE, min(config.MAX_STEER_ANGLE, steering_angle))
        
        return steering_angle, lane_offset, confidence
    
    def _draw_debug_overlay(self, frame, left_lane, right_lane, steering_angle):
        """Draw lane lines and steering vector for debugging"""
        debug_frame = frame.copy()
        height, width = debug_frame.shape[:2]
        
        # Draw ROI boundary
        if config.SHOW_ROI:
            roi_top = int(height * config.ROI_TOP_RATIO)
            cv2.line(debug_frame, (0, roi_top), (width, roi_top), (0, 255, 255), 2)
        
        # Draw lane lines
        if config.SHOW_LANE_LINES:
            if left_lane is not None:
                cv2.line(debug_frame, (left_lane[0], left_lane[1]), 
                        (left_lane[2], left_lane[3]), (0, 255, 0), 3)
            if right_lane is not None:
                cv2.line(debug_frame, (right_lane[0], right_lane[1]), 
                        (right_lane[2], right_lane[3]), (0, 255, 0), 3)
        
        # Draw steering direction
        center_x = width // 2
        bottom_y = height - 20
        arrow_length = 80
        angle_rad = np.deg2rad(steering_angle)
        end_x = int(center_x + arrow_length * np.sin(angle_rad))
        end_y = int(bottom_y - arrow_length * np.cos(angle_rad))
        
        cv2.arrowedLine(debug_frame, (center_x, bottom_y), (end_x, end_y), 
                       (0, 0, 255), 3, tipLength=0.3)
        
        # Add text overlay
        cv2.putText(debug_frame, f"Steer: {steering_angle}°", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return debug_frame