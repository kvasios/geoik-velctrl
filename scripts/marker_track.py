#!/usr/bin/env python3
"""
Marker Tracking for Robot End-Effector Control
Detects ArUco markers/boards with RealSense camera and sends target poses to robot
"""

import numpy as np
import cv2
import pyrealsense2 as rs
import socket
import threading
import time
import yaml
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
from scipy.spatial.transform import Rotation, Slerp
import argparse
from pathlib import Path


@dataclass
class MarkerPose:
    """6D pose of detected marker"""
    position: np.ndarray  # [x, y, z] in meters
    orientation: np.ndarray  # quaternion [qx, qy, qz, qw]
    tvec: np.ndarray  # translation vector from detection
    rvec: np.ndarray  # rotation vector from detection
    detected: bool = True
    num_markers: int = 1  # Number of markers detected (for boards)


class MarkerTracker:
    """Real-time marker tracking with RealSense camera"""
    
    def __init__(self, 
                 robot_ip: str = "192.168.122.100", 
                 robot_port: int = 8888,
                 marker_size: float = 0.05,  # 5cm marker
                 use_board: bool = False,
                 board_rows: int = 4,
                 board_cols: int = 4,
                 board_marker_size: float = 0.04,  # 4cm
                 board_spacing: float = 0.01,  # 1cm
                 aruco_dict: str = "4x4_50",
                 camera_fps: int = 30,
                 debug: bool = False):
        
        # Network setup
        self.robot_ip = robot_ip
        self.robot_port = robot_port
        self.robot_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        # Debug mode
        self.debug = debug
        self.command_send_count = 0
        self.last_command_log_time = time.time()
        
        # Camera-to-end-effector transform: +90° around Z axis
        # Camera X → EE Y, Camera Y → EE -X, Camera Z → EE Z
        # (camera and EE Z both point from sensor towards the marker / scene)
        self.direct_mapping = lambda cam: np.array([-cam[1], cam[0], cam[2]])
        self.orientation_transform_matrix = np.array([
            [0, -1,  0],  # EE X = -Camera Y
            [1,  0,  0],  # EE Y = Camera X
            [0,  0,  1]   # EE Z = Camera Z
        ])
        
        # Tracking state
        self.tracking_active = False
        self.current_marker_pose: Optional[MarkerPose] = None
        self.last_detection_time = 0.0
        self.detection_timeout = 0.5  # seconds
        
        # Pose measurement smoothing (temporal filter for camera jitter)
        self.measurement_smoothing = 0.7  # EMA smoothing for raw pose measurements (0.0=no filter, 0.9=max)
        self.smoothed_measurement_pos: Optional[np.ndarray] = None
        self.smoothed_measurement_quat: Optional[np.ndarray] = None
        self.outlier_threshold = 0.15  # 15cm - reject measurements that jump too far
        
        # ArUco dictionary mapping
        dict_map = {
            '4x4_50': cv2.aruco.DICT_4X4_50,
            '5x5_100': cv2.aruco.DICT_5X5_100,
            '6x6_250': cv2.aruco.DICT_6X6_250,
            'apriltag': cv2.aruco.DICT_APRILTAG_36h11,
        }
        
        # Setup ArUco detector with improved parameters for stability
        aruco_dict_enum = dict_map.get(aruco_dict, cv2.aruco.DICT_4X4_50)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_enum)
        self.detector_params = cv2.aruco.DetectorParameters()
        
        # Improve detection stability with aggressive corner refinement
        self.detector_params.adaptiveThreshWinSizeMin = 3
        self.detector_params.adaptiveThreshWinSizeMax = 23
        self.detector_params.adaptiveThreshWinSizeStep = 10
        self.detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector_params.cornerRefinementWinSize = 5
        self.detector_params.cornerRefinementMaxIterations = 100  # Increased from 30
        self.detector_params.cornerRefinementMinAccuracy = 0.01   # Tighter from 0.1
        
        # Additional detection parameters for stability
        self.detector_params.minMarkerPerimeterRate = 0.03
        self.detector_params.maxMarkerPerimeterRate = 4.0
        self.detector_params.polygonalApproxAccuracyRate = 0.03
        
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.detector_params)
        
        # Marker/Board configuration
        self.use_board = use_board
        self.marker_size = marker_size  # meters (for single marker)
        
        if use_board:
            # Create ChArUco board (chessboard with ArUco markers)
            square_length = board_marker_size + board_spacing
            marker_length = board_marker_size
            self.charuco_board = cv2.aruco.CharucoBoard(
                (board_cols, board_rows),  # squaresX, squaresY
                square_length,
                marker_length,
                self.aruco_dict
            )
            self.board_rows = board_rows
            self.board_cols = board_cols
            self.square_length = square_length
            self.marker_length = marker_length
            marker_type = f"{board_rows}x{board_cols} ChArUco board"
            print(f"ChArUco board created: {board_cols}x{board_rows} squares, "
                  f"square={square_length*1000:.2f}mm, marker={marker_length*1000:.2f}mm")
        else:
            self.charuco_board = None
            marker_type = "single marker"
        
        # RealSense pipeline setup
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # Enable color stream only (max resolution for better detection)
        resolution_attempts = [
            (1280, 800, "1280x800")
        ]
        
        profile = None
        for width, height, name in resolution_attempts:
            try:
                self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, camera_fps)
                profile = self.pipeline.start(self.config)
                print(f"✓ Using {name} resolution")
                break
            except RuntimeError:
                self.config = rs.config()  # Reset config for next attempt
                continue
        
        if profile is None:
            raise RuntimeError("Failed to start RealSense camera with any supported resolution")
        
        # Get camera intrinsics
        color_profile = profile.get_stream(rs.stream.color)
        intrinsics = color_profile.as_video_stream_profile().get_intrinsics()
        
        self.camera_matrix = np.array([
            [intrinsics.fx, 0, intrinsics.ppx],
            [0, intrinsics.fy, intrinsics.ppy],
            [0, 0, 1]
        ])
        self.dist_coeffs = np.array(intrinsics.coeffs)
        
        
        # GUI state
        self.running = True
        self.frame_count = 0
        self.fps = 0.0
        self.last_fps_time = time.time()
        
        # Command sending thread
        self.command_thread = threading.Thread(target=self._command_loop, daemon=True)
        self.command_thread.start()
        
        print("Marker Tracker initialized")
        if debug:
            print(f"Mode: DEBUG (commands NOT sent to robot)")
        else:
            print(f"Robot: {robot_ip}:{robot_port}")
        print(f"Marker type: {marker_type}")
        if use_board:
            print(f"ChArUco board: {board_rows}x{board_cols} squares")
            print(f"  Square length: {self.square_length*100:.1f}cm")
            print(f"  Marker length: {self.marker_length*100:.1f}cm")
            print(f"  Spacing: {board_spacing*100:.1f}cm")
        else:
            print(f"Marker size: {marker_size*100:.1f}cm")
        print(f"ArUco dict: {aruco_dict}")
        print(f"Camera FPS: {camera_fps}")
        print(f"\nMeasurement filtering (camera jitter reduction):")
        print(f"  Temporal smoothing: {self.measurement_smoothing:.2f} (0=none, 0.9=max)")
        print(f"  Outlier threshold: {self.outlier_threshold*1000:.0f}mm (jump rejection)")
        print(f"  Corner refinement: {self.detector_params.cornerRefinementMaxIterations} iterations")
        print(f"\nEye-in-hand: Camera on end-effector, +90° Z rotation")
        print("  -Camera Y → EE X, Camera X → EE Y, Camera Z → EE Z")
        print("  Server transforms EE-relative poses to base frame in real-time")
        print("\nControls:")
        print("  't' - Start/Stop tracking")
        print("  'q' - Quit")
        print("\nWaiting for marker detection...")
    
    def transform_camera_to_ee(self, camera_pos: np.ndarray, camera_quat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Transform position and orientation from camera frame to end-effector frame
        
        Camera is mounted on end-effector with -90° rotation around Z axis.
        Server will transform from EE frame to base frame using real-time robot state.
        """
        # Transform position: Camera → EE frame
        ee_pos = self.direct_mapping(camera_pos)
        
        # Transform orientation: Camera → EE frame
        camera_rot = Rotation.from_quat(camera_quat)
        camera_matrix = camera_rot.as_matrix()
        ee_matrix = self.orientation_transform_matrix @ camera_matrix @ self.orientation_transform_matrix.T
        ee_rot = Rotation.from_matrix(ee_matrix)
        ee_quat = ee_rot.as_quat()
        
        return ee_pos, ee_quat
    
    def detect_marker(self, color_image: np.ndarray) -> Optional[MarkerPose]:
        """Detect ArUco marker/ChArUco board and estimate 6D pose"""
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        
        # Detect markers
        corners, ids, rejected = self.detector.detectMarkers(gray)
        
        if ids is None or len(ids) == 0:
            return None
        
        if self.use_board:
            # ChArUco board detection (more stable than GridBoard)
            # Step 1: Interpolate chessboard corners from detected markers
            result = cv2.aruco.interpolateCornersCharuco(
                corners, ids, gray, self.charuco_board,
                cameraMatrix=self.camera_matrix,
                distCoeffs=self.dist_coeffs
            )
            
            # Handle different OpenCV versions (returns 3 or 4 values)
            if len(result) == 3:
                num_corners, charuco_corners, charuco_ids = result
            else:
                charuco_corners, charuco_ids, _ = result
                num_corners = len(charuco_corners) if charuco_corners is not None else 0
            
            if charuco_corners is None or num_corners < 4:
                return None  # Need at least 4 corners for pose estimation
            
            # Step 2: Estimate pose from chessboard corners (sub-pixel accurate)
            retval, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                charuco_corners, charuco_ids, self.charuco_board,
                self.camera_matrix, self.dist_coeffs,
                None, None
            )
            
            if retval == 0:  # No valid pose
                return None
            
            # ChArUco returns tvec/rvec as 1D arrays (3,) not nested
            tvec = tvec.flatten()
            rvec = rvec.flatten()
            num_detected = len(ids)  # Number of ArUco markers detected
            
        else:
            # Single marker pose estimation
            corner = corners[0]
            
            # Estimate pose
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                corner, self.marker_size, self.camera_matrix, self.dist_coeffs
            )
            
            rvec = rvec[0].flatten()
            tvec = tvec[0].flatten()
            num_detected = 1
        
        # Convert rotation vector to quaternion
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        rotation = Rotation.from_matrix(rotation_matrix)
        quat_raw = rotation.as_quat()  # [qx, qy, qz, qw]
        
        # Position in camera frame (tvec is already 1D array [x, y, z])
        position_raw = tvec  # [x, y, z] in meters
        
        # Apply temporal smoothing to reduce camera jitter
        if self.smoothed_measurement_pos is None:
            # First detection - initialize smoothed state
            self.smoothed_measurement_pos = position_raw.copy()
            self.smoothed_measurement_quat = quat_raw.copy()
            position_smoothed = position_raw
            quat_smoothed = quat_raw
        else:
            # Check for outliers (sudden large jumps indicate detection error)
            pos_diff = np.linalg.norm(position_raw - self.smoothed_measurement_pos)
            if pos_diff > self.outlier_threshold:
                # Outlier detected - keep previous smoothed value
                position_smoothed = self.smoothed_measurement_pos
                quat_smoothed = self.smoothed_measurement_quat
            else:
                # Apply EMA smoothing to position
                alpha = 1.0 - self.measurement_smoothing
                self.smoothed_measurement_pos = (self.measurement_smoothing * self.smoothed_measurement_pos + 
                                                 alpha * position_raw)
                position_smoothed = self.smoothed_measurement_pos
                
                # Apply SLERP smoothing to orientation
                rot_prev = Rotation.from_quat(self.smoothed_measurement_quat)
                rot_raw = Rotation.from_quat(quat_raw)
                rot_smoothed = Rotation.from_quat([rot_prev.as_quat(), rot_raw.as_quat()])
                slerp = Slerp([0, 1], rot_smoothed)
                rot_filtered = slerp(alpha)
                self.smoothed_measurement_quat = rot_filtered.as_quat()
                quat_smoothed = self.smoothed_measurement_quat
        
        self.last_detection_time = time.time()
        
        return MarkerPose(
            position=position_smoothed,
            orientation=quat_smoothed,
            tvec=position_smoothed,  # Use smoothed for consistency
            rvec=rvec,  # Keep raw rvec for visualization
            detected=True,
            num_markers=num_detected
        )
    
    def draw_marker_frame(self, image: np.ndarray, marker_pose: MarkerPose):
        """Draw 3D coordinate frame on detected marker"""
        if marker_pose is None or not marker_pose.detected:
            return
        
        # Draw axis (X=red, Y=green, Z=blue)
        # Use appropriate marker size for axis length
        if self.use_board:
            axis_length = self.marker_length * 1.5  # Use marker_length for ChArUco
        else:
            axis_length = self.marker_size * 1.5  # Use marker_size for single marker
        
        axis_points = np.float32([
            [0, 0, 0],
            [axis_length, 0, 0],
            [0, axis_length, 0],
            [0, 0, axis_length]
        ]).reshape(-1, 3)
        
        # Project 3D points to image plane
        imgpts, _ = cv2.projectPoints(
            axis_points, marker_pose.rvec, marker_pose.tvec,
            self.camera_matrix, self.dist_coeffs
        )
        
        imgpts = imgpts.astype(int)
        origin = tuple(imgpts[0].ravel())
        
        # Draw axes
        cv2.line(image, origin, tuple(imgpts[1].ravel()), (0, 0, 255), 3)  # X-axis (red)
        cv2.line(image, origin, tuple(imgpts[2].ravel()), (0, 255, 0), 3)  # Y-axis (green)
        cv2.line(image, origin, tuple(imgpts[3].ravel()), (255, 0, 0), 3)  # Z-axis (blue)
    
    def start_tracking(self):
        """Start tracking - sends marker pose measurements to server"""
        if self.current_marker_pose is not None and self.current_marker_pose.detected:
            self.tracking_active = True
            print(f"\n✓ Tracking started!")
            print(f"  Server will lock reference and compute visual servo")
        else:
            print("\n✗ Cannot start tracking - no marker detected!")
    
    def stop_tracking(self, reason: str = "User stopped"):
        """Stop tracking"""
        if self.tracking_active:
            self.tracking_active = False
            # Reset measurement smoothing filter for next tracking session
            self.smoothed_measurement_pos = None
            self.smoothed_measurement_quat = None
            print(f"\n✗ Tracking stopped: {reason}")
    
    def send_robot_command(self, position: np.ndarray, orientation: np.ndarray):
        """Send EE-relative pose measurement to robot via UDP

        Format: \"x y z qx qy qz qw\" (all in end-effector frame).
        Server (visual-servo mode) interprets this as T_ee_marker_meas.
        """
        try:
            # Format: "x y z qx qy qz qw" (EE-relative frame)
            message = f"{position[0]:.6f} {position[1]:.6f} {position[2]:.6f} " + \
                     f"{orientation[0]:.6f} {orientation[1]:.6f} {orientation[2]:.6f} {orientation[3]:.6f}"
            
            # In debug mode, don't actually send (just visualize)
            if not self.debug:
                self.robot_socket.sendto(message.encode(), (self.robot_ip, self.robot_port))
            
            # Debug logging
            if self.debug:
                self.command_send_count += 1
                current_time = time.time()
                if current_time - self.last_command_log_time >= 1.0:
                    print(f"\n[DEBUG] Commands sent: {self.command_send_count} Hz")
                    print(f"  Marker in EE Frame (sent to server):")
                    print(f"    Position: X:{position[0]:+.3f} Y:{position[1]:+.3f} Z:{position[2]:+.3f}m")
                    print(f"    Quat:     x:{orientation[0]:+.3f} y:{orientation[1]:+.3f} z:{orientation[2]:+.3f} w:{orientation[3]:+.3f}")
                    print(f"  (Server performs visual servo control with gain/smoothing/deadband)")
                    self.command_send_count = 0
                    self.last_command_log_time = current_time
        except Exception as e:
            print(f"Error sending command: {e}")
    
    def _command_loop(self):
        """Background thread to send commands at fixed rate"""
        rate = 100  # Hz
        dt = 1.0 / rate
        
        while self.running:
            if self.tracking_active and self.current_marker_pose is not None and self.current_marker_pose.detected:
                # Compute marker pose in EE frame (eye-in-hand measurement)
                pos_ee, quat_ee = self.transform_camera_to_ee(
                    self.current_marker_pose.position,
                    self.current_marker_pose.orientation
                )
                # Send measurement to server; server handles visual servoing
                self.send_robot_command(pos_ee, quat_ee)
            
            time.sleep(dt)
    
    def check_detection_timeout(self):
        """Check if marker detection has timed out"""
        if self.tracking_active:
            time_since_detection = time.time() - self.last_detection_time
            if time_since_detection > self.detection_timeout:
                self.stop_tracking("Marker lost")
    
    def render_debug_window(self):
        """Render debug visualization of marker measurement"""
        if not self.debug or not self.tracking_active or not self.current_marker_pose:
            return
        
        # Create a simple info canvas
        canvas_size = 400
        canvas = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 40
        
        # Draw title
        cv2.putText(canvas, "Marker Measurement (Debug)", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Transform current marker to EE frame
        pos_ee, quat_ee = self.transform_camera_to_ee(
            self.current_marker_pose.position,
            self.current_marker_pose.orientation
        )
        
        # Show measurements
        info_y = 80
        cv2.putText(canvas, "Current Marker Pose:", (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        cv2.putText(canvas, "Camera Frame:", (10, info_y + 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 1)
        cv2.putText(canvas, f"  Pos: X:{self.current_marker_pose.position[0]:+.3f} "
                           f"Y:{self.current_marker_pose.position[1]:+.3f} "
                           f"Z:{self.current_marker_pose.position[2]:+.3f}m",
                   (10, info_y + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 200, 255), 1)
        
        cv2.putText(canvas, "EE Frame (sent to server):", (10, info_y + 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(canvas, f"  Pos: X:{pos_ee[0]:+.3f} Y:{pos_ee[1]:+.3f} Z:{pos_ee[2]:+.3f}m",
                   (10, info_y + 125), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.putText(canvas, f"  Quat: x:{quat_ee[0]:+.2f} y:{quat_ee[1]:+.2f} "
                           f"z:{quat_ee[2]:+.2f} w:{quat_ee[3]:+.2f}",
                   (10, info_y + 150), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        cv2.putText(canvas, "Measurement Filtering:", (10, info_y + 190),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(canvas, f"  Temporal smoothing: {self.measurement_smoothing:.2f}",
                   (10, info_y + 215), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(canvas, f"  Outlier threshold: {self.outlier_threshold*1000:.0f}mm",
                   (10, info_y + 240), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        cv2.putText(canvas, "Server performs visual servo control:", (10, info_y + 280),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 255), 1)
        cv2.putText(canvas, "  - Locks marker ref on first frame", (10, info_y + 305),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
        cv2.putText(canvas, "  - Computes EE target to restore ref", (10, info_y + 325),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
        cv2.putText(canvas, "  - Applies gain/smoothing/deadband", (10, info_y + 345),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
        
        # Show warning
        cv2.putText(canvas, "[DEBUG MODE - Commands NOT sent to robot]", 
                   (10, canvas_size - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 2)
        
        cv2.imshow('Debug: Marker Measurement', canvas)
    
    def render_gui(self, color_image: np.ndarray) -> np.ndarray:
        """Render GUI with overlays"""
        display = color_image.copy()
        h, w = display.shape[:2]
        
        # Update FPS counter
        self.frame_count += 1
        current_time = time.time()
        if current_time - self.last_fps_time >= 1.0:
            self.fps = self.frame_count / (current_time - self.last_fps_time)
            self.frame_count = 0
            self.last_fps_time = current_time
        
        # Status panel background
        panel_height = 150
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (w, panel_height), (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)
        
        # Title
        cv2.putText(display, "Marker Tracker", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # FPS
        cv2.putText(display, f"FPS: {self.fps:.1f}", (w - 120, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Detection status
        if self.current_marker_pose is not None and self.current_marker_pose.detected:
            if self.use_board:
                status_text = f"ChArUco board detected ({self.current_marker_pose.num_markers} markers) - Press 't'"
            else:
                status_text = "Marker detected - Press 't' to track"
            status_color = (0, 255, 0)
            self.draw_marker_frame(display, self.current_marker_pose)
            
            # Show marker position in camera and EE frames
            pos_cam = self.current_marker_pose.position
            pos_ee, _ = self.transform_camera_to_ee(pos_cam, self.current_marker_pose.orientation)
            cv2.putText(display, f"Camera: X:{pos_cam[0]:+.3f} Y:{pos_cam[1]:+.3f} Z:{pos_cam[2]:+.3f}m",
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 200, 255), 1)
            cv2.putText(display, f"EE:     X:{pos_ee[0]:+.3f} Y:{pos_ee[1]:+.3f} Z:{pos_ee[2]:+.3f}m",
                       (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 100), 1)
        else:
            if self.use_board:
                status_text = "No ChArUco board detected"
            else:
                status_text = "No marker detected"
            status_color = (0, 0, 255)
        
        cv2.putText(display, status_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        # Tracking status
        if self.tracking_active:
            tracking_text = "TRACKING ACTIVE"
            tracking_color = (0, 255, 255)
            
            # Draw tracking indicator
            cv2.circle(display, (w - 40, 60), 15, tracking_color, -1)
        else:
            tracking_text = "Tracking: OFF"
            tracking_color = (100, 100, 100)
        
        cv2.putText(display, tracking_text, (10, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, tracking_color, 2)
        
        # Controls
        controls = "Controls: [t] Track  [q] Quit"
        cv2.putText(display, controls, (10, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return display
    
    def run(self):
        """Main GUI loop"""
        cv2.namedWindow('Marker Tracker', cv2.WINDOW_AUTOSIZE)
        
        try:
            while self.running:
                # Get color frame directly (no depth needed)
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                
                if not color_frame:
                    continue
                
                # Convert to numpy array
                color_image = np.asanyarray(color_frame.get_data())
                
                # Detect marker
                self.current_marker_pose = self.detect_marker(color_image)
                
                # Check for detection timeout
                self.check_detection_timeout()
                
                # Render GUI
                display = self.render_gui(color_image)
                
                # Show frame
                cv2.imshow('Marker Tracker', display)
                
                # Render debug window if enabled
                if self.debug:
                    self.render_debug_window()
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\nQuitting...")
                    break
                elif key == ord('t'):
                    if not self.tracking_active:
                        self.start_tracking()
                    else:
                        self.stop_tracking("User stopped")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        self.running = False
        self.pipeline.stop()
        cv2.destroyAllWindows()
        self.robot_socket.close()
        print("Cleanup complete")


def load_marker_config(config_path: str) -> Dict:
    """Load marker configuration from YAML file"""
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    aruco_dict = config.get('aruco_dict', '4x4_50')
    measured_marker_size = config.get('measured_marker_size')
    
    # Determine if single marker or ChArUco board
    if 'squares_x' in config or 'squares_y' in config:
        # ChArUco board configuration
        squares_x = config.get('squares_x', config.get('board_cols', 3))
        squares_y = config.get('squares_y', config.get('board_rows', 3))
        
        measured_square_size = config.get('measured_square_size')
        
        # Prefer square measurement if provided (more accurate), otherwise use marker measurement
        if measured_square_size is not None:
            print("✓ Using measured square size (most accurate)")
            
            # Direct measurement - no scaling needed
            scaled_square_length = measured_square_size
            original_marker_length = config.get('_original_marker_length_meters')
            original_square_length = config.get('_original_square_length_meters')
            
            if original_marker_length and original_square_length:
                # Calculate marker size from square size (maintain original ratio)
                ratio = original_marker_length / original_square_length
                scaled_marker_length = scaled_square_length * ratio
                scaled_spacing = scaled_square_length - scaled_marker_length
            else:
                # Fallback: assume marker scales with square
                if measured_marker_size is not None:
                    scaled_marker_length = measured_marker_size
                    scaled_spacing = scaled_square_length - scaled_marker_length
                else:
                    raise ValueError("Need either measured_marker_size or original reference values")
            
            # Calculate total board dimensions
            total_width = squares_x * scaled_square_length
            total_height = squares_y * scaled_square_length
            
            print(f"  Measured square: {measured_square_size*1000:.2f}mm")
            print(f"  Calculated marker: {scaled_marker_length*1000:.2f}mm")
            print(f"  Calculated spacing: {scaled_spacing*1000:.2f}mm")
            print(f"  Calculated board size: {total_width*100:.1f}cm × {total_height*100:.1f}cm")
            
        elif measured_marker_size is not None:
            print("✓ Using measured marker size - inferring square size (assumes uniform scaling)")
            
            # Get original reference values (for calculating scale factor)
            original_marker_length = config.get('_original_marker_length_meters')
            original_square_length = config.get('_original_square_length_meters')
            
            if original_marker_length and original_square_length:
                # Calculate scale factor from measured vs expected marker size
                scale_factor = measured_marker_size / original_marker_length
                
                # Scale square length proportionally (assumes uniform printing scale)
                scaled_square_length = original_square_length * scale_factor
                scaled_marker_length = measured_marker_size
                scaled_spacing = scaled_square_length - scaled_marker_length
                
                # Calculate total board dimensions
                total_width = squares_x * scaled_square_length
                total_height = squares_y * scaled_square_length
                
                print(f"  Measured marker: {measured_marker_size*1000:.2f}mm")
                print(f"  Inferred square length: {scaled_square_length*1000:.2f}mm")
                print(f"  Inferred spacing: {scaled_spacing*1000:.2f}mm")
                print(f"  Calculated board size: {total_width*100:.1f}cm × {total_height*100:.1f}cm")
                print(f"  ⚠ Note: For best accuracy, measure square size directly")
            else:
                raise ValueError("Missing original reference values in config file. Please regenerate the marker.")
        else:
            raise ValueError(
                "Please measure EITHER:\n"
                "  - 'measured_marker_size' (easier, assumes uniform scaling)\n"
                "  - 'measured_square_size' (more accurate, recommended)\n"
                "Example: If marker measures 4.5cm, set: measured_marker_size: 0.045\n"
                "        If square measures 5.1cm, set: measured_square_size: 0.051"
            )
        
        # Return detection config for ChArUco
        return {
            'aruco_dict': aruco_dict,
            'use_board': True,
            'board_rows': squares_y,  # squares_y maps to rows
            'board_cols': squares_x,  # squares_x maps to cols
            'board_marker_size_meters': scaled_marker_length,
            'board_spacing_meters': scaled_spacing,
        }
    
    else:
        # Single marker configuration
        if measured_marker_size is not None:
            print("✓ Using measured marker size from config file")
            return {
                'aruco_dict': aruco_dict,
                'use_board': False,
                'marker_size_meters': measured_marker_size,
            }
        else:
            raise ValueError(
                "Please measure the marker side and fill in 'measured_marker_size' in the YAML file (in meters).\n"
                "Example: If marker measures 5.0cm, set: measured_marker_size: 0.05"
            )


def main():
    parser = argparse.ArgumentParser(
        description='ArUco Marker Tracking for Robot Control',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use YAML config file (recommended)
  python3 marker_track.py --config markers/board_4x4_4x4_50.yaml

  # Use command-line arguments
  python3 marker_track.py --board --board-rows 4 --board-cols 4 --board-marker-size 0.04
        """
    )
    
    # Config file (takes precedence)
    parser.add_argument('--config', type=str,
                       help='Path to YAML config file (overrides other marker args)')
    
    # Network parameters
    parser.add_argument('--robot-ip', type=str, default='192.168.122.100',
                       help='Robot IP address (default: 192.168.122.100)')
    parser.add_argument('--robot-port', type=int, default=8888,
                       help='Robot UDP port (default: 8888)')
    
    # Marker/Board selection (ignored if --config is used)
    marker_group = parser.add_mutually_exclusive_group()
    marker_group.add_argument('--single', action='store_true',
                             help='Use single marker (default if no config)')
    marker_group.add_argument('--board', action='store_true',
                             help='Use marker board (more robust)')
    
    # Single marker parameters (ignored if --config is used)
    parser.add_argument('--marker-size', type=float, default=0.05,
                       help='Single marker size in meters (default: 0.05)')
    
    # Board parameters (ignored if --config is used)
    parser.add_argument('--board-rows', type=int, default=4,
                       help='Board rows (default: 4)')
    parser.add_argument('--board-cols', type=int, default=4,
                       help='Board columns (default: 4)')
    parser.add_argument('--board-marker-size', type=float, default=0.04,
                       help='Board marker size in meters (default: 0.04)')
    parser.add_argument('--board-spacing', type=float, default=0.01,
                       help='Board marker spacing in meters (default: 0.01)')
    
    # ArUco dictionary (ignored if --config is used)
    parser.add_argument('--dict', type=str, default='4x4_50',
                       choices=['4x4_50', '5x5_100', '6x6_250', 'apriltag'],
                       help='ArUco dictionary (default: 4x4_50)')
    
    # Camera parameters
    parser.add_argument('--fps', type=int, default=30,
                       help='Camera FPS (default: 30)')
    
    # Debug mode
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode (visualize commands, do NOT send to robot)')
    
    args = parser.parse_args()
    
    # Auto-detect YAML config if not provided
    if not args.config:
        markers_dir = Path('markers')
        if markers_dir.exists():
            yaml_files = list(markers_dir.glob('*.yaml'))
            if yaml_files:
                # Use the most recently modified YAML file
                latest_yaml = max(yaml_files, key=lambda p: p.stat().st_mtime)
                args.config = str(latest_yaml)
                print(f"Auto-detected config: {args.config}")
    
    # Load config from YAML if provided
    if args.config:
        try:
            config = load_marker_config(args.config)
            
            # Extract parameters from config
            use_board = config.get('use_board', False)
            aruco_dict = config.get('aruco_dict', '4x4_50')
            
            if use_board:
                marker_size = None  # Not used for boards
                board_rows = config.get('board_rows', 4)
                board_cols = config.get('board_cols', 4)
                board_marker_size = config.get('board_marker_size_meters', 0.04)
                board_spacing = config.get('board_spacing_meters', 0.01)
            else:
                marker_size = config.get('marker_size_meters', 0.05)
                board_rows = None
                board_cols = None
                board_marker_size = None
                board_spacing = None
            
        except Exception as e:
            print(f"Error loading config file: {e}")
            print("Falling back to command-line arguments...")
            use_board = args.board
            marker_size = args.marker_size
            board_rows = args.board_rows
            board_cols = args.board_cols
            board_marker_size = args.board_marker_size
            board_spacing = args.board_spacing
            aruco_dict = args.dict
    else:
        # Use command-line arguments
        use_board = args.board
        marker_size = args.marker_size
        board_rows = args.board_rows
        board_cols = args.board_cols
        board_marker_size = args.board_marker_size
        board_spacing = args.board_spacing
        aruco_dict = args.dict
    
    try:
        tracker = MarkerTracker(
            robot_ip=args.robot_ip,
            robot_port=args.robot_port,
            marker_size=marker_size if not use_board else 0.05,  # Dummy value for boards
            use_board=use_board,
            board_rows=board_rows if use_board else 4,
            board_cols=board_cols if use_board else 4,
            board_marker_size=board_marker_size if use_board else 0.04,
            board_spacing=board_spacing if use_board else 0.01,
            aruco_dict=aruco_dict,
            camera_fps=args.fps,
            debug=args.debug
        )
        
        if args.debug:
            print("\n" + "="*60)
            print("DEBUG MODE ENABLED")
            print("  - Commands will be visualized but NOT sent to robot")
            print("  - Safe for testing without hardware")
            print("="*60 + "\n")
        
        tracker.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

