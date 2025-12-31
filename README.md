## Demo
![Demo GIF](assets/demo.gif)

## Acknowledgement

This project is currently being conducted under the supervision of **Dr. Hwa-Seop Lim**, **Dr. Taek-Geun Yoo** at the **AI Research Division, Korea Institute of Science and Technology (KIST)**.

The system design and implementation were developed by referencing and extending several open-source research frameworks and codebases, including:

- **MuJoCo Menagerie (Google DeepMind)**  
  Used as a reference for standardized robot model definitions, kinematic consistency, and simulation-ready articulation structures.

- **MINK (Kevin Zakka)**  
  Referenced for practical inverse kinematics formulations and solver design insights, particularly for stable numerical IK behavior.

- **OpenTeleVision / TeleVision**  
  Served as the foundational VR-based teleoperation framework enabling real-time manipulation via immersive interfaces.

These references significantly informed the kinematic modeling, inverse kinematics formulation, and VR teleoperation pipeline used in this project.

---

## Future Work

The following improvements and extensions are planned for future development:

- **Improve MIT Mode stability**  
  Jitter is observed during teleoperation in MIT Mode. A smoothing and gain-tuning update is planned to address this.

- **Joint control refinement & gain tuning**  
  PID(Kp/Kd) and input filtering parameters will be iteratively optimized for more stable and compliant motion control.

---

**More updates coming soon.**