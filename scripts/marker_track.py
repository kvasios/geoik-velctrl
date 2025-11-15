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
from dataclasses import dataclass
from typing import Optional, Tuple
from scipy.spatial.transform import Rotation
import argparse


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
                 robot_ip: str = "192.168.18.1", 
                 robot_port: int = 8888,
                 marker_size: float = 0.05,  # 5cm marker
                 use_board: bool = False,
                 board_rows: int = 4,
                 board_cols: int = 4,
                 board_marker_size: float = 0.04,  # 4cm
                 board_spacing: float = 0.01,  # 1cm
                 aruco_dict: str = "4x4_50",
                 camera_fps: int = 30):
        
        # Network setup
        self.robot_ip = robot_ip
        self.robot_port = robot_port
        self.robot_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        # Tracking state
        self.tracking_active = False
        self.marker_locked_pose: Optional[MarkerPose] = None
        self.current_marker_pose: Optional[MarkerPose] = None
        self.last_detection_time = 0.0
        self.detection_timeout = 0.5  # seconds
        
        # ArUco dictionary mapping
        dict_map = {
            '4x4_50': cv2.aruco.DICT_4X4_50,
            '5x5_100': cv2.aruco.DICT_5X5_100,
            '6x6_250': cv2.aruco.DICT_6X6_250,
            'apriltag': cv2.aruco.DICT_APRILTAG_36h11,
        }
        
        # Setup ArUco detector
        aruco_dict_enum = dict_map.get(aruco_dict, cv2.aruco.DICT_4X4_50)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_enum)
        self.detector_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.detector_params)
        
        # Marker/Board configuration
        self.use_board = use_board
        self.marker_size = marker_size  # meters (for single marker)
        
        if use_board:
            # Create ArUco board
            self.board = cv2.aruco.GridBoard(
                (board_cols, board_rows),
                board_marker_size,
                board_spacing,
                self.aruco_dict
            )
            self.board_rows = board_rows
            self.board_cols = board_cols
            marker_type = f"{board_rows}x{board_cols} board"
        else:
            self.board = None
            marker_type = "single marker"
        
        # RealSense pipeline setup
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # Enable color and depth streams
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, camera_fps)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, camera_fps)
        
        # Start pipeline
        profile = self.pipeline.start(self.config)
        
        # Get camera intrinsics
        color_profile = profile.get_stream(rs.stream.color)
        intrinsics = color_profile.as_video_stream_profile().get_intrinsics()
        
        self.camera_matrix = np.array([
            [intrinsics.fx, 0, intrinsics.ppx],
            [0, intrinsics.fy, intrinsics.ppy],
            [0, 0, 1]
        ])
        self.dist_coeffs = np.array(intrinsics.coeffs)
        
        # Align depth to color
        self.align = rs.align(rs.stream.color)
        
        # GUI state
        self.running = True
        self.frame_count = 0
        self.fps = 0.0
        self.last_fps_time = time.time()
        
        # Command sending thread
        self.command_thread = threading.Thread(target=self._command_loop, daemon=True)
        self.command_thread.start()
        
        print("Marker Tracker initialized")
        print(f"Robot: {robot_ip}:{robot_port}")
        print(f"Marker type: {marker_type}")
        if use_board:
            print(f"Board markers: {board_marker_size*100:.1f}cm, spacing: {board_spacing*100:.1f}cm")
        else:
            print(f"Marker size: {marker_size*100:.1f}cm")
        print(f"ArUco dict: {aruco_dict}")
        print(f"Camera FPS: {camera_fps}")
        print("\nControls:")
        print("  't' - Start/Stop tracking")
        print("  'q' - Quit")
        print("\nWaiting for marker detection...")
    
    def detect_marker(self, color_image: np.ndarray) -> Optional[MarkerPose]:
        """Detect ArUco marker/board and estimate 6D pose"""
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        
        # Detect markers
        corners, ids, rejected = self.detector.detectMarkers(gray)
        
        if ids is None or len(ids) == 0:
            return None
        
        if self.use_board:
            # Use board pose estimation (more robust with multiple markers)
            retval, rvec, tvec = cv2.aruco.estimatePoseBoard(
                corners, ids, self.board, 
                self.camera_matrix, self.dist_coeffs,
                None, None
            )
            
            if retval == 0:  # No valid pose
                return None
            
            num_detected = len(ids)
            
        else:
            # Single marker pose estimation
            corner = corners[0]
            
            # Estimate pose
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                corner, self.marker_size, self.camera_matrix, self.dist_coeffs
            )
            
            rvec = rvec[0]
            tvec = tvec[0]
            num_detected = 1
        
        # Convert rotation vector to quaternion
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        rotation = Rotation.from_matrix(rotation_matrix)
        quat = rotation.as_quat()  # [qx, qy, qz, qw]
        
        # Position in camera frame
        position = tvec[0] if self.use_board else tvec[0]  # [x, y, z] in meters
        
        self.last_detection_time = time.time()
        
        return MarkerPose(
            position=position,
            orientation=quat,
            tvec=tvec if self.use_board else tvec[0],
            rvec=rvec,
            detected=True,
            num_markers=num_detected
        )
    
    def draw_marker_frame(self, image: np.ndarray, marker_pose: MarkerPose):
        """Draw 3D coordinate frame on detected marker"""
        if marker_pose is None or not marker_pose.detected:
            return
        
        # Draw axis (X=red, Y=green, Z=blue)
        axis_length = self.marker_size * 1.5
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
        """Lock marker pose and start tracking"""
        if self.current_marker_pose is not None and self.current_marker_pose.detected:
            self.marker_locked_pose = self.current_marker_pose
            self.tracking_active = True
            print(f"\n✓ Tracking started at pose: {self.marker_locked_pose.position}")
            print(f"  Position: [{self.marker_locked_pose.position[0]:.3f}, "
                  f"{self.marker_locked_pose.position[1]:.3f}, "
                  f"{self.marker_locked_pose.position[2]:.3f}]")
        else:
            print("\n✗ Cannot start tracking - no marker detected!")
    
    def stop_tracking(self, reason: str = "User stopped"):
        """Stop tracking"""
        if self.tracking_active:
            self.tracking_active = False
            self.marker_locked_pose = None
            print(f"\n✗ Tracking stopped: {reason}")
    
    def send_robot_command(self, position: np.ndarray, orientation: np.ndarray):
        """Send pose command to robot via UDP"""
        try:
            message = f"{position[0]:.6f} {position[1]:.6f} {position[2]:.6f} " + \
                     f"{orientation[0]:.6f} {orientation[1]:.6f} {orientation[2]:.6f} {orientation[3]:.6f}"
            
            self.robot_socket.sendto(message.encode(), (self.robot_ip, self.robot_port))
        except Exception as e:
            print(f"Error sending command: {e}")
    
    def _command_loop(self):
        """Background thread to send commands at fixed rate"""
        rate = 100  # Hz
        dt = 1.0 / rate
        
        while self.running:
            if self.tracking_active and self.marker_locked_pose is not None:
                # Send locked pose to robot
                self.send_robot_command(
                    self.marker_locked_pose.position,
                    self.marker_locked_pose.orientation
                )
            
            time.sleep(dt)
    
    def check_detection_timeout(self):
        """Check if marker detection has timed out"""
        if self.tracking_active:
            time_since_detection = time.time() - self.last_detection_time
            if time_since_detection > self.detection_timeout:
                self.stop_tracking("Marker lost")
    
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
                status_text = f"Board detected ({self.current_marker_pose.num_markers} markers) - Press 't'"
            else:
                status_text = "Marker detected - Press 't' to track"
            status_color = (0, 255, 0)
            self.draw_marker_frame(display, self.current_marker_pose)
            
            # Show marker position
            pos = self.current_marker_pose.position
            cv2.putText(display, f"Position: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]m",
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        else:
            if self.use_board:
                status_text = "No board detected"
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
                # Get frames
                frames = self.pipeline.wait_for_frames()
                aligned_frames = self.align.process(frames)
                
                color_frame = aligned_frames.get_color_frame()
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


def main():
    parser = argparse.ArgumentParser(description='ArUco Marker Tracking for Robot Control')
    
    # Network parameters
    parser.add_argument('--robot-ip', type=str, default='192.168.18.1',
                       help='Robot IP address (default: 192.168.18.1)')
    parser.add_argument('--robot-port', type=int, default=8888,
                       help='Robot UDP port (default: 8888)')
    
    # Marker/Board selection
    marker_group = parser.add_mutually_exclusive_group()
    marker_group.add_argument('--single', action='store_true',
                             help='Use single marker (default)')
    marker_group.add_argument('--board', action='store_true',
                             help='Use marker board (more robust)')
    
    # Single marker parameters
    parser.add_argument('--marker-size', type=float, default=0.05,
                       help='Single marker size in meters (default: 0.05)')
    
    # Board parameters
    parser.add_argument('--board-rows', type=int, default=4,
                       help='Board rows (default: 4)')
    parser.add_argument('--board-cols', type=int, default=4,
                       help='Board columns (default: 4)')
    parser.add_argument('--board-marker-size', type=float, default=0.04,
                       help='Board marker size in meters (default: 0.04)')
    parser.add_argument('--board-spacing', type=float, default=0.01,
                       help='Board marker spacing in meters (default: 0.01)')
    
    # ArUco dictionary
    parser.add_argument('--dict', type=str, default='4x4_50',
                       choices=['4x4_50', '5x5_100', '6x6_250', 'apriltag'],
                       help='ArUco dictionary (default: 4x4_50)')
    
    # Camera parameters
    parser.add_argument('--fps', type=int, default=30,
                       help='Camera FPS (default: 30)')
    
    args = parser.parse_args()
    
    try:
        tracker = MarkerTracker(
            robot_ip=args.robot_ip,
            robot_port=args.robot_port,
            marker_size=args.marker_size,
            use_board=args.board,
            board_rows=args.board_rows,
            board_cols=args.board_cols,
            board_marker_size=args.board_marker_size,
            board_spacing=args.board_spacing,
            aruco_dict=args.dict,
            camera_fps=args.fps
        )
        tracker.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

