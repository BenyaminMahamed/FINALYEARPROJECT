# camera_utils.py - Camera access utilities for Raspberry Pi

import cv2
import numpy as np

try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False
    print("[CAMERA] Picamera2 not available, will try OpenCV")


class Camera:
    """Unified camera interface supporting both Picamera2 and OpenCV"""
    
    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height
        self.camera = None
        self.camera_type = None
        
        # Try Picamera2 first (best for Raspberry Pi Camera Module)
        if PICAMERA2_AVAILABLE:
            try:
                print("[CAMERA] Attempting Picamera2...")
                self.camera = Picamera2()
                config = self.camera.create_preview_configuration(
                    main={"size": (width, height), "format": "RGB888"}
                )
                self.camera.configure(config)
                self.camera.start()
                self.camera_type = "picamera2"
                print("[CAMERA] Using Picamera2 (Raspberry Pi Camera)")
                return
            except Exception as e:
                print(f"[CAMERA] Picamera2 failed: {e}")
                if self.camera:
                    try:
                        self.camera.close()
                    except:
                        pass
                self.camera = None
        
        # Fallback to OpenCV
        print("[CAMERA] Attempting OpenCV...")
        
        # Try V4L2 backend
        self.camera = cv2.VideoCapture(0, cv2.CAP_V4L2)
        if self.camera.isOpened():
            self.camera_type = "opencv_v4l2"
            print("[CAMERA] Using OpenCV with V4L2")
        else:
            # Try standard OpenCV
            self.camera = cv2.VideoCapture(0)
            if self.camera.isOpened():
                self.camera_type = "opencv"
                print("[CAMERA] Using OpenCV standard")
        
        if self.camera and self.camera_type and self.camera_type.startswith("opencv"):
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer lag
    
    def is_opened(self):
        """Check if camera is successfully opened"""
        if self.camera_type == "picamera2":
            return self.camera is not None
        elif self.camera_type and self.camera_type.startswith("opencv"):
            return self.camera.isOpened()
        return False
    
    def read(self):
        """Read a frame from the camera"""
        if self.camera_type == "picamera2":
            try:
                frame = self.camera.capture_array()
                # Convert RGB to BGR for OpenCV compatibility
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                return True, frame
            except Exception as e:
                print(f"[CAMERA] Picamera2 read error: {e}")
                return False, None
        
        elif self.camera_type and self.camera_type.startswith("opencv"):
            return self.camera.read()
        
        return False, None
    
    def release(self):
        """Release the camera"""
        if self.camera_type == "picamera2":
            if self.camera:
                self.camera.stop()
                self.camera.close()
        elif self.camera_type and self.camera_type.startswith("opencv"):
            if self.camera:
                self.camera.release()
        
        print("[CAMERA] Released")