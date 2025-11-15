# geoik-velctrl

A geometric IK-based velocity control server for the Franka Emika robot.

## Overview

This standalone server receives end-effector pose commands via UDP and controls the Franka robot using:
- **Geometric IK** (GeoFIK algorithm) for analytical inverse kinematics
- **Weighted optimization** for manipulability, joint limits, and base stability
- **Ruckig trajectory generation** for smooth, jerk-limited motion profiles
- **Joint-space velocity control** at 1kHz via libfranka

## Quick Start

### Prerequisites

- **libfranka** (matching your robot firmware version)
- **Eigen3**: `sudo apt install libeigen3-dev`
- **Ruckig**: `sudo apt install libruckig-dev`

### Build

```bash
mkdir build && cd build
cmake -DFRANKA_INSTALL_PATH=/path/to/libfranka/install ..
make -j4
```

### Run

```bash
./franka_velocity_server <robot-hostname> [bidexhand]
```

Arguments:
- `<robot-hostname>`: IP address or hostname of your Franka robot
- `[bidexhand]`: Optional. Set to `true` to limit J7 range (default: `false`)

The server listens on **UDP port 8888** for pose commands in the format:

```
x y z qx qy qz qw
```

Where:
- `x y z`: Position in meters (robot base frame)
- `qx qy qz qw`: Orientation quaternion

Example:
```
0.3 0.0 0.5 0.0 0.0 0.0 1.0
```

## Attribution

This project builds upon excellent work from:
- **Franka Robotics GmbH** - Robot control examples (Apache-2.0)
- **Zhengyang Kris Weng et al.** - VR teleoperation framework (MIT)
- **Pablo Lopez-Custodio et al.** - GeoFIK inverse kinematics algorithm

See [ATTRIBUTION.md](ATTRIBUTION.md) for detailed credits and citations.

## License

- Project code: MIT (see [LICENSE](LICENSE))
- Third-party components: See [LICENSE-APACHE-2.0](LICENSE-APACHE-2.0) and [LICENSE-FRANKA-VR-TELEOP](LICENSE-FRANKA-VR-TELEOP)

