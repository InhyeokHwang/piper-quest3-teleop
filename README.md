## Demo
![Demo GIF](assets/demo.gif)

## Acknowledgement

This project is currently being conducted under the supervision of **Dr. Hwa-Seop Lim** at the **AI Research Division, Korea Institute of Science and Technology (KIST)**.

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

- **Further improving IK convergence robustness**  
  Inverse kinematics stability has been significantly improved compared to earlier versions, showing reliable convergence in most operational scenarios.  
  Future work will focus on enhancing robustness near kinematic singularities and under highly unfavorable joint configurations to ensure consistently stable real-time teleoperation.

- **Robot joint visualization in Quest 3**  
  Add real-time visualization of the robot arm’s joint states and configurations within the Quest environment to improve operator awareness and control precision during teleoperation.
