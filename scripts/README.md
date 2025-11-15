# Marker Tracking Scripts

## Overview

This directory contains scripts for controlling the Franka robot using visual tracking.

### `marker_track.py`

Real-time ChArUco marker tracking with RealSense camera for robot end-effector control.

**Features:**
- Live camera feed with marker detection visualization
- 6D pose estimation (position + orientation)
- Interactive tracking control (press 't' to lock/unlock)
- Automatic tracking abort on marker loss
- Visual coordinate frame overlay
- 100Hz command rate to robot (camera runs at ~30Hz)
- Debug mode for safe testing without robot

## Setup

### 1. Install Python dependencies

```bash
cd scripts
pip install -r requirements.txt
```

### 2. Generate and print ChArUco board

```bash
# Generate default 4x4 ChArUco board (20cm x 20cm)
python3 generate_marker.py

# This creates:
# - markers/board_*.pdf  (print this)
# - markers/board_*.yaml (auto-loaded by marker_track.py)
```

After printing, **measure one marker side** and update the `measured_marker_size` field in the YAML file.

### 3. Attach RealSense camera to robot

Mount the RealSense camera on the robot end-effector pointing toward the workspace.

## Usage

### Debug Mode (Safe Testing - Recommended First!)

Test without sending commands to robot:

```bash
python3 marker_track.py --debug
```

This opens:
- Main window: Camera feed with marker detection
- Debug window: Visualization of TCP commands that WOULD be sent

Use this to verify tracking behavior before connecting to real robot!

### Basic Usage (Live Robot Control)

```bash
python3 marker_track.py
```

Auto-detects the latest generated marker YAML in `markers/` directory.

### With Custom Parameters

```bash
python3 marker_track.py \
  --robot-ip 192.168.122.100 \
  --robot-port 8888 \
  --config markers/board_4x4_4x4_50.yaml \
  --fps 30 \
  --smoothing 0.7
```

**Smoothing Parameter:**
- `--smoothing 0.0` = No smoothing (raw, responsive)
- `--smoothing 0.7` = Moderate smoothing (default, balanced)
- `--smoothing 0.9` = Heavy smoothing (slow, very stable)

Like the VR code, we apply exponential moving average (EMA) to position and Slerp to orientation.

### GUI Controls

- **`t`** - Start/stop tracking (locks marker pose)
- **`q`** - Quit application

### Workflow

1. **Start in debug mode first** - `python3 marker_track.py --debug`
2. **Show ChArUco board to camera** - When detected, you'll see "ChArUco board detected - Press 't'"
3. **Press `t`** - Tracking starts (marker pose locked as reference)
4. **Move board** - Watch the debug window show how robot would move (differential control)
5. **Verify behavior looks correct** - Check position/orientation deltas make sense
6. **Press `q`** and restart without `--debug` for live robot control
7. **Move marker away** - If marker is lost, tracking automatically stops

## Marker Pose Convention

The detected marker pose is in the **camera frame**:
- **X-axis** (red): Right
- **Y-axis** (green): Down  
- **Z-axis** (blue): Forward (into the marker)

Position is in **meters** relative to camera optical center.

## Control Strategy

**Differential Control (like VR controller):**

1. When you press `t`, the current marker pose is locked as "zero reference"
2. As you move the marker, the **delta** (difference from locked pose) is calculated
3. This delta is sent directly to the robot as a target TCP pose
4. Robot moves to maintain the marker at the locked pose relative to the camera

**Command Format:**

Poses are sent to the robot via UDP at 100Hz as:
```
x y z qx qy qz qw
```

Where:
- `x y z`: Position in meters (differential from base)
- `qx qy qz qw`: Orientation quaternion

**Note:** Camera runs at ~30Hz, but commands are sent at 100Hz (same command repeated until new detection)

## Troubleshooting

### No camera detected
```bash
# Check RealSense connection
rs-enumerate-devices

# Test camera
realsense-viewer
```

### ChArUco board not detected
- Ensure good lighting
- Check board is not occluded
- Verify measured marker size is correct in YAML file
- Board should be flat (not warped or curved)
- Need at least 4 corners visible for pose estimation

### Poor tracking / Jumpy motion
- Use debug mode to verify: `--debug`
- Check that measured marker size in YAML is accurate
- Ensure board is printed at correct scale (measure with ruler!)
- ChArUco should be very stable - if jumpy, check lighting and board flatness

### Commands not being sent
- Make sure you're NOT in debug mode (remove `--debug` flag)
- Check robot IP and port are correct
- Verify `franka_velocity_server` is running on robot

## Architecture

```
RealSense Camera → ChArUco Detection → Pose Estimation → Differential Control → UDP → Robot
     (~30Hz)           (OpenCV)            (PnP)              (100Hz)          (UDP)  (1kHz)
                                                                                        
                                           ┌─────────────┐
                                           │ Lock Pose   │ (when 't' pressed)
                                           │ Calculate Δ │
                                           │ Send Target │
                                           └─────────────┘
```

**Key Points:**
- Camera detection: ~30Hz (limited by hardware)
- Command sending: 100Hz (independent thread)
- Robot control loop: 1kHz (in `franka_velocity_server`)
- Same command is sent multiple times until new detection arrives (like VR code)

