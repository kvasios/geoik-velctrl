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
                 debug: bool = False,
                 smoothing_factor: float = 0.85,  # Higher for 30Hz marker rate
                 position_scale: float = 0.5,  # Scale factor for position deltas
                 position_deadband: float = 0.002):  # 2mm deadband
        
        # Network setup
        self.robot_ip = robot_ip
        self.robot_port = robot_port
        self.robot_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        # Debug mode
        self.debug = debug
        self.command_send_count = 0
        self.last_command_log_time = time.time()
        
        # Camera-to-end-effector transform: -90° around Z axis
        # Camera X → EE -Y, Camera Y → EE X, Camera Z → EE Z
        self.direct_mapping = lambda cam: np.array([cam[1], -cam[0], cam[2]])
        self.orientation_transform_matrix = np.array([
            [0,  1,  0],  # EE X = Camera Y
            [-1, 0,  0],  # EE Y = -Camera X
            [0,  0,  1]   # EE Z = Camera Z
        ])
        
        # Control parameters
        self.smoothing_factor = smoothing_factor
        self.position_scale = position_scale  # Gain for position control
        self.position_deadband = position_deadband  # Deadband to prevent oscillation
        
        # Smoothed state
        self.smoothed_position = np.array([0.0, 0.0, 0.0])
        self.smoothed_orientation = np.array([0.0, 0.0, 0.0, 1.0])
        
        # Tracking state
        self.tracking_active = False
        self.marker_locked_pose: Optional[MarkerPose] = None  # "Zero" reference when 't' was pressed
        self.current_marker_pose: Optional[MarkerPose] = None
        self.last_processed_marker_pose: Optional[MarkerPose] = None  # Track if we've already processed this detection
        self.robot_base_pose = {
            'position': np.array([0.0, 0.0, 0.0]),  # Robot base reference (identity)
            'orientation': np.array([0.0, 0.0, 0.0, 1.0])
        }
        self.target_tcp_pose = {
            'position': np.array([0.0, 0.0, 0.0]),
            'orientation': np.array([0.0, 0.0, 0.0, 1.0])
        }
        self.last_detection_time = 0.0
        self.detection_timeout = 0.5  # seconds
        
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
        
        # Improve detection stability
        self.detector_params.adaptiveThreshWinSizeMin = 3
        self.detector_params.adaptiveThreshWinSizeMax = 23
        self.detector_params.adaptiveThreshWinSizeStep = 10
        self.detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector_params.cornerRefinementWinSize = 5
        self.detector_params.cornerRefinementMaxIterations = 30
        self.detector_params.cornerRefinementMinAccuracy = 0.1
        
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
        print(f"Control parameters:")
        print(f"  Smoothing: {smoothing_factor:.2f} (0=none, 0.95=max)")
        print(f"  Position scale: {position_scale:.2f} (gain/sensitivity)")
        print(f"  Position deadband: {position_deadband*1000:.1f}mm (anti-oscillation)")
        print(f"\nEye-in-hand: Camera on end-effector, -90° Z rotation")
        print("  Camera Y → EE X, -Camera X → EE Y, Camera Z → EE Z")
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
        quat = rotation.as_quat()  # [qx, qy, qz, qw]
        
        # Position in camera frame (tvec is already 1D array [x, y, z])
        position = tvec  # [x, y, z] in meters
        
        self.last_detection_time = time.time()
        
        return MarkerPose(
            position=position,
            orientation=quat,
            tvec=tvec,
            rvec=rvec,
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
        """Lock marker pose and start tracking (differential control like VR)"""
        if self.current_marker_pose is not None and self.current_marker_pose.detected:
            # Lock the current marker pose as "zero" reference
            self.marker_locked_pose = self.current_marker_pose
            
            # Initialize target to robot base (identity)
            self.target_tcp_pose['position'] = self.robot_base_pose['position'].copy()
            self.target_tcp_pose['orientation'] = self.robot_base_pose['orientation'].copy()
            
            self.tracking_active = True
            print(f"\n✓ Tracking started - marker locked as reference!")
            print(f"  Marker will move robot differentially (like VR controller)")
        else:
            print("\n✗ Cannot start tracking - no marker detected!")
    
    def compute_visual_servoing_command(self):
        """
        Differential control: marker motion = TCP motion (like VR controller).
        Applies smoothing exactly like vr_to_robot_converter.py
        
        IMPORTANT: Only update smoothing when we get NEW marker detection (camera ~30Hz),
        but send commands at 100Hz (repeat last smoothed command until new detection).
        """
        if not self.tracking_active or self.marker_locked_pose is None or self.current_marker_pose is None:
            return
        
        # Check if this is a new detection (only smooth on new data!)
        # Compare by identity - if it's the same object, we've already processed it
        if self.current_marker_pose is self.last_processed_marker_pose:
            # No new detection - just keep sending the last smoothed command (already in target_tcp_pose)
            return
        
        # Mark this detection as processed
        self.last_processed_marker_pose = self.current_marker_pose
        
        # Calculate marker pose delta from locked pose IN CAMERA FRAME
        marker_pos_delta_cam = self.current_marker_pose.position - self.marker_locked_pose.position
        
        # Transform delta to robot frame using configured mapping
        marker_pos_delta_robot = self.direct_mapping(marker_pos_delta_cam)
        
        # Apply position scale (gain) to delta
        scaled_delta = marker_pos_delta_robot * self.position_scale
        
        # Apply deadband per-axis to prevent micro-oscillations
        for i in range(3):
            if abs(scaled_delta[i]) < self.position_deadband:
                scaled_delta[i] = 0.0
        
        # Apply smoothing to position delta (EMA filter) IN ROBOT FRAME
        self.smoothed_position = (self.smoothing_factor * self.smoothed_position + 
                                  (1 - self.smoothing_factor) * scaled_delta)
        
        # Calculate relative rotation IN CAMERA FRAME
        locked_rot_cam = Rotation.from_quat(self.marker_locked_pose.orientation)
        current_rot_cam = Rotation.from_quat(self.current_marker_pose.orientation)
        relative_rot_cam = current_rot_cam * locked_rot_cam.inv()
        
        # Transform relative rotation to robot frame using matrix transform
        # R_robot = T @ R_camera @ T^T (where T is the coordinate transform matrix)
        relative_matrix_cam = relative_rot_cam.as_matrix()
        relative_matrix_robot = self.orientation_transform_matrix @ relative_matrix_cam @ self.orientation_transform_matrix.T
        relative_rot_robot = Rotation.from_matrix(relative_matrix_robot)
        
        # Apply relative rotation to base orientation to get target orientation
        base_rot = Rotation.from_quat(self.robot_base_pose['orientation'])
        target_rot = relative_rot_robot * base_rot  # Unsmoothed target
        
        # Apply smoothing to TARGET orientation using Slerp (spherical linear interpolation)
        # This matches VR code: smooth the target, not the delta
        slerp_t = 1 - self.smoothing_factor
        current_smoothed_rot = Rotation.from_quat(self.smoothed_orientation)
        key_rotations = Rotation.from_quat([current_smoothed_rot.as_quat(), target_rot.as_quat()])
        slerp = Slerp([0, 1], key_rotations)
        smoothed_target_rot = slerp(slerp_t)
        self.smoothed_orientation = smoothed_target_rot.as_quat()
        
        # Normalize quaternion to ensure it remains a unit quaternion
        self.smoothed_orientation = self.smoothed_orientation / np.linalg.norm(self.smoothed_orientation)
        
        # Update target TCP pose with smoothed values
        self.target_tcp_pose['position'] = self.robot_base_pose['position'] + self.smoothed_position
        self.target_tcp_pose['orientation'] = self.smoothed_orientation
    
    def stop_tracking(self, reason: str = "User stopped"):
        """Stop tracking"""
        if self.tracking_active:
            self.tracking_active = False
            self.marker_locked_pose = None
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
                    if self.marker_locked_pose and self.current_marker_pose:
                        delta_cam = self.current_marker_pose.position - self.marker_locked_pose.position
                        delta_ee = self.direct_mapping(delta_cam)
                        scaled_delta = delta_ee * self.position_scale
                        print(f"  Marker Delta (Camera): X:{delta_cam[0]:+.3f} Y:{delta_cam[1]:+.3f} Z:{delta_cam[2]:+.3f}m")
                        print(f"  Marker Delta (EE):     X:{delta_ee[0]:+.3f} Y:{delta_ee[1]:+.3f} Z:{delta_ee[2]:+.3f}m")
                        print(f"  Scaled Delta (x{self.position_scale:.2f}): X:{scaled_delta[0]:+.3f} Y:{scaled_delta[1]:+.3f} Z:{scaled_delta[2]:+.3f}m")
                        print(f"  Smoothed Delta (EE):   X:{self.smoothed_position[0]:+.3f} Y:{self.smoothed_position[1]:+.3f} Z:{self.smoothed_position[2]:+.3f}m")
                    print(f"  (Server transforms EE→Base using real-time robot state)")
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
        """Render debug visualization of command frame"""
        if not self.debug or not self.tracking_active:
            return
        
        # Create a 3D visualization canvas
        canvas_size = 600
        canvas = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 40
        
        # Draw title
        cv2.putText(canvas, "EE-Relative Command (Debug)", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw coordinate system in center
        center = (canvas_size // 2, canvas_size // 2)
        scale = 100  # pixels per meter
        
        # Draw base frame (gray)
        base_x_end = (int(center[0] + scale * 0.5), center[1])
        base_y_end = (center[0], int(center[1] - scale * 0.5))
        cv2.line(canvas, center, base_x_end, (100, 100, 100), 2)
        cv2.line(canvas, center, base_y_end, (100, 100, 100), 2)
        cv2.putText(canvas, "Base", (center[0] + 10, center[1] + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
        
        # Draw target TCP frame (colored)
        if self.target_tcp_pose is not None:
            pos = self.target_tcp_pose['position']
            quat = self.target_tcp_pose['orientation']
            
            # Convert position to canvas coordinates
            tcp_x = int(center[0] + pos[0] * scale)
            tcp_y = int(center[1] - pos[2] * scale)  # Flip Y for display
            
            # Draw target position
            cv2.circle(canvas, (tcp_x, tcp_y), 5, (0, 255, 255), -1)
            
            # Draw target orientation as arrows
            rot = Rotation.from_quat(quat)
            # X-axis (red)
            x_dir = rot.apply([0.3, 0, 0])
            x_end = (int(tcp_x + x_dir[0] * scale), int(tcp_y - x_dir[2] * scale))
            cv2.arrowedLine(canvas, (tcp_x, tcp_y), x_end, (0, 0, 255), 2, tipLength=0.3)
            
            # Y-axis (green)
            y_dir = rot.apply([0, 0.3, 0])
            y_end = (int(tcp_x + y_dir[0] * scale), int(tcp_y - y_dir[2] * scale))
            cv2.arrowedLine(canvas, (tcp_x, tcp_y), y_end, (0, 255, 0), 2, tipLength=0.3)
            
            # Z-axis (blue)
            z_dir = rot.apply([0, 0, 0.3])
            z_end = (int(tcp_x + z_dir[0] * scale), int(tcp_y - z_dir[2] * scale))
            cv2.arrowedLine(canvas, (tcp_x, tcp_y), z_end, (255, 0, 0), 2, tipLength=0.3)
            
            # Show numerical values
            info_y = 80
            cv2.putText(canvas, f"Target TCP Pose:", (10, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(canvas, f"  Position: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]m", 
                       (10, info_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(canvas, f"  Quat: [{quat[0]:.3f}, {quat[1]:.3f}, {quat[2]:.3f}, {quat[3]:.3f}]",
                       (10, info_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Show marker delta if available
            if self.marker_locked_pose and self.current_marker_pose:
                # Camera frame delta
                delta_pos_cam = self.current_marker_pose.position - self.marker_locked_pose.position
                cv2.putText(canvas, f"Marker Delta (Camera Frame):", (10, info_y + 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 1)
                cv2.putText(canvas, f"  X:{delta_pos_cam[0]:+.3f} Y:{delta_pos_cam[1]:+.3f} Z:{delta_pos_cam[2]:+.3f}m",
                           (10, info_y + 115), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 200, 255), 1)
                
                # EE frame delta (transformed from camera)
                delta_pos_ee = self.direct_mapping(delta_pos_cam)
                cv2.putText(canvas, f"EE Delta (raw):", (10, info_y + 145),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 100), 1)
                cv2.putText(canvas, f"  X:{delta_pos_ee[0]:+.3f} Y:{delta_pos_ee[1]:+.3f} Z:{delta_pos_ee[2]:+.3f}m",
                           (10, info_y + 170), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 100), 1)
                
                # Scaled delta
                scaled_delta = delta_pos_ee * self.position_scale
                cv2.putText(canvas, f"Scaled Delta (x{self.position_scale:.2f}):", (10, info_y + 200),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 150, 50), 1)
                cv2.putText(canvas, f"  X:{scaled_delta[0]:+.3f} Y:{scaled_delta[1]:+.3f} Z:{scaled_delta[2]:+.3f}m",
                           (10, info_y + 225), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 150, 50), 1)
                
                # Smoothed output (in EE frame, sent to server)
                cv2.putText(canvas, f"Smoothed EE Delta (sent):", (10, info_y + 255),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(canvas, f"  X:{self.smoothed_position[0]:+.3f} Y:{self.smoothed_position[1]:+.3f} Z:{self.smoothed_position[2]:+.3f}m",
                           (10, info_y + 280), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                
                cv2.putText(canvas, f"Settings: scale={self.position_scale:.2f} smooth={self.smoothing_factor:.2f} db={self.position_deadband*1000:.1f}mm",
                           (10, info_y + 315), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                cv2.putText(canvas, f"Cam→EE: -90deg Z | Server: EE→Base (real-time)",
                           (10, info_y + 340), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
        
        # Show warning
        cv2.putText(canvas, "[DEBUG MODE - Commands NOT sent to robot]", 
                   (10, canvas_size - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
        
        cv2.imshow('Debug: TCP Command Frame', canvas)
    
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
    
    # Control parameters
    parser.add_argument('--smoothing', type=float, default=0.85,
                       help='Smoothing factor (0.0=no smoothing, 0.95=max, default: 0.85 for 30Hz)')
    parser.add_argument('--position-scale', type=float, default=0.5,
                       help='Position gain/scale factor (0.1-2.0, default: 0.5)')
    parser.add_argument('--position-deadband', type=float, default=0.002,
                       help='Position deadband in meters (default: 0.002 = 2mm)')
    
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
            debug=args.debug,
            smoothing_factor=args.smoothing,
            position_scale=args.position_scale,
            position_deadband=args.position_deadband
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

