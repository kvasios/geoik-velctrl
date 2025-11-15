## Attribution

This project reuses and builds upon work from several upstream sources.  
The files in `robot_client/` should be understood in that context.

### Franka Robotics GmbH (Apache-2.0)

- **Files**: `examples_common.cpp`, `examples_common.h`, parts of `franka_vr_control_client.cpp`
- **Source**: Franka example code, as adapted in  
  [`franka-vr-teleop` VR robot client](https://github.com/wengmister/franka-vr-teleop)
- **License**: Apache-2.0 (see `LICENSE-APACHE-2.0`)
- **Copyright**: Copyright (c) 2023 Franka Robotics GmbH  
- **Notes**: Original copyright and license headers are preserved in the
  corresponding source files.

### Franka VR Teleoperation Client (MIT)

- **Repository**: [`franka-vr-teleop`](https://github.com/wengmister/franka-vr-teleop)
- **Author**: Zhengyang Kris Weng et al.
- **License**: MIT (see `LICENSE-FRANKA-VR-TELEOP`)
- **Files adapted in this project**:
  - `robot_client/src/franka_vr_control_client.cpp`
  - `robot_client/include/geofik.h`
  - `robot_client/src/geofik.cpp`
  - `robot_client/include/weighted_ik.h`
  - `robot_client/src/weighted_ik.cpp`
- **Notes**:
  - These files implement a VR-based Franka velocity control client, weighted IK,
    and a geometric IK solver for the Franka arm.
  - They are adapted from the VR robot client described in the
    `franka-vr-teleop` repository and used here under the terms of its MIT
    license.

If you use this project in academic work or publications, please also
consider citing the associated teleoperation framework:

```bibtex
@misc{weng2025levr,
      title={LeVR: A Modular VR Teleoperation Framework for Imitation Learning in Dexterous Manipulation},
      author={Zhengyang Kris Weng and Matthew L. Elwin and Han Liu},
      year={2025},
      eprint={2509.14349},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2509.14349}
}
```

### GeoFIK (Algorithmic Source for IK)

- **Repository**: [`GeoFIK`](https://github.com/PabloLopezCustodio/GeoFIK)
- **Author**: Pablo C. Lopez-Custodio et al.
- **License**: As of 2025-11-15, the GeoFIK repository does not declare an
  explicit open-source license. The implementation in this project is taken
  from `franka-vr-teleop` (MIT-licensed) and not directly from the GeoFIK
  repository, but the underlying IK formulation follows that work.
- **Scientific reference**:  
  Lopez-Custodio PC, Gong Y, Figueredo LFC,  
  “GeoFIK: A Fast and Reliable Geometric Solver for the IK of the Franka Arm
  based on Screw Theory Enabling Multiple Redundancy Parameters”,  
  PREPRINT: arXiv:2503.03992v1 (2025), [`https://arxiv.org/abs/2503.03992`](https://arxiv.org/abs/2503.03992).

BibTeX:

```bibtex
@misc{lopezcustodio2025geofikfastreliablegeometric,
      title={GeoFIK: A Fast and Reliable Geometric Solver for the IK of the Franka Arm based on Screw Theory Enabling Multiple Redundancy Parameters},
      author={Pablo C. Lopez-Custodio and Yuhe Gong and Luis F. C. Figueredo},
      year={2025},
      eprint={2503.03992},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2503.03992}
}
```

### Your Project

You may choose an overall license for your own standalone velocity server
(for example, MIT or Apache-2.0). Whatever you choose, you must:

- Keep the existing copyright and license headers.
- Ship `LICENSE-APACHE-2.0` and `LICENSE-FRANKA-VR-TELEOP` alongside your
  binaries or source distributions.
- Keep this `ATTRIBUTION.md` (or an equivalent NOTICE/ATTRIBUTION file) so
  that upstream work is clearly credited.


