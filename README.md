
## Acknowledgement

This project is currently being conducted under the supervision of Dr. Hwa-Seop Lim at the AI Research Division, Korea Institute of Science and Technology (KIST).
It was developed by referencing and extending the ideas and codebase from OpenTeleVision, a VR-based teleoperation framework for robotic manipulation.

## Future Work

The following improvements and extensions are planned for future development:

- **Improving IK convergence stability**  
  Address convergence failures in inverse kinematics, particularly under singular or unfavorable joint configurations, to ensure smoother and more reliable teleoperation.

- **Gripper activation via Quest3 controller input**
  Note: Gripper control has been successfully integrated and validated in the simulation environment. However, issues remain when applying the same gripper control to the real robot, and further debugging and hardware–software integration work is    ongoing.
  
- **Robot joint visualization in Quest3**  
  Add real-time visualization of the robot arm’s joint positions within the Quest environment to improve operator awareness and control precision.

- **Motion preference and posture biasing**  
  Introduce joint preference or posture biasing mechanisms to encourage natural, stable, and task-consistent robot arm configurations during teleoperation.
