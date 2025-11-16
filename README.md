# geoik-velctrl

Geometric IK-based velocity control with eye-in-hand visual servoing for Franka Emika robots. Track ArUco markers in real-time and maintain desired end-effector-to-marker poses.

![Demo](data/media/demo.gif)

## Quick Start

### 1. Install and Run C++ Velocity Server

Using [servobox](https://www.servobox.dev):

```bash
servobox pkg-install geoik-velctrl
servobox run geoik-velctrl <robot-ip> false vs
```

Arguments: `<robot-ip>` `<bidexhand: true/false>` `<mode: vs for visual servo, vr for VR>`

### 2. Install Python Marker Tracking

Create environment with micromamba (or conda/mamba):

```bash
micromamba create -n marker-track python=3.10
micromamba activate marker-track
pip install -r requirements.txt
```

### 3. Run Marker Tracker

```bash
python3 scripts/marker_track.py --config markers/board_4x4_4x4_50.yaml
```

Press `t` to start/stop tracking. The robot maintains the locked end-effector-to-marker pose as you move the marker.

## Attribution

This project builds upon excellent work from:
- **Franka Robotics GmbH** - Robot control examples (Apache-2.0)
- **Zhengyang Kris Weng et al.** - VR teleoperation framework (MIT)
- **Pablo Lopez-Custodio et al.** - GeoFIK inverse kinematics algorithm

See [ATTRIBUTION.md](ATTRIBUTION.md) for detailed credits and citations.

## License

- Project code: MIT (see [LICENSE](LICENSE))
- Third-party components: See [LICENSE-APACHE-2.0](LICENSE-APACHE-2.0) and [LICENSE-FRANKA-VR-TELEOP](LICENSE-FRANKA-VR-TELEOP)

