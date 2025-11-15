# Marker Tracking Scripts

## Overview

This directory contains scripts for controlling the Franka robot using visual tracking.

### `marker_track.py`

Real-time AprilTag marker tracking with RealSense camera for robot end-effector control.

**Features:**
- Live camera feed with marker detection visualization
- 6D pose estimation (position + orientation)
- Interactive tracking control (press 't' to lock/unlock)
- Automatic tracking abort on marker loss
- Visual coordinate frame overlay
- 100Hz command rate to robot

## Setup

### 1. Install Python dependencies

```bash
cd scripts
pip install -r requirements.txt
```

### 2. Print AprilTag marker

Print an AprilTag from the 36h11 family. You can generate one here:
https://github.com/AprilRobotics/apriltag-imgs/tree/master/tag36h11

Default marker size is **5cm** (change with `--marker-size` flag).

### 3. Attach RealSense camera to robot

Mount the RealSense camera on the robot end-effector pointing toward the workspace.

## Usage

### Basic Usage

```bash
python3 marker_track.py
```

### With Custom Parameters

```bash
python3 marker_track.py \
  --robot-ip 192.168.18.1 \
  --robot-port 8888 \
  --marker-size 0.05 \
  --fps 30
```

### GUI Controls

- **`t`** - Start/stop tracking (locks marker pose)
- **`q`** - Quit application

### Workflow

1. **Start the script** - Camera feed opens showing live video
2. **Show marker to camera** - When detected, you'll see "Marker detected - Press 't' to track"
3. **Press `t`** - Tracking starts, robot receives locked pose commands at 100Hz
4. **Move marker away** - If marker is lost, tracking automatically stops
5. **Press `q`** to quit

## Marker Pose Convention

The detected marker pose is in the **camera frame**:
- **X-axis** (red): Right
- **Y-axis** (green): Down  
- **Z-axis** (blue): Forward (into the marker)

Position is in **meters** relative to camera optical center.

## Command Format

Poses are sent to the robot via UDP as:
```
x y z qx qy qz qw
```

Where:
- `x y z`: Position in meters
- `qx qy qz qw`: Orientation quaternion

## Troubleshooting

### No camera detected
```bash
# Check RealSense connection
rs-enumerate-devices

# Test camera
realsense-viewer
```

### Marker not detected
- Ensure good lighting
- Check marker is not occluded
- Verify marker size matches `--marker-size` parameter
- Use AprilTag 36h11 family

### Poor tracking
- Increase marker size for better detection at distance
- Reduce camera motion blur (better lighting, lower exposure)
- Adjust FPS if experiencing lag

## Architecture

```
RealSense Camera → AprilTag Detection → Pose Estimation → UDP Commands → Robot
     (30Hz)              (CV)              (PnP)            (100Hz)      (1kHz)
```

The command loop runs at 100Hz independently of the camera frame rate to provide smooth control.

