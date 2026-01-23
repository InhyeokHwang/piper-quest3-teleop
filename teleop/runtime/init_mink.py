# teleop/runtime/init_mink.py
import numpy as np
import mujoco
import mink
from loop_rate_limiters import RateLimiter

from .. import config

def init_mink(q_zero: np.ndarray):
    model = mujoco.MjModel.from_xml_path(config.PIPER_MJCF_PATH)
    data = mujoco.MjData(model)
    configuration = mink.Configuration(model)

    limits = [mink.ConfigurationLimit(model=model)]
    max_vel = float(getattr(config, "MINK_MAX_VEL", np.pi))
    max_velocities = {f"joint{i}": max_vel for i in range(1, 7)}
    limits.append(mink.VelocityLimit(model, max_velocities))

    solver = getattr(config, "MINK_SOLVER", "daqp")
    rate_hz = float(getattr(config, "MINK_RATE_HZ", 200.0))
    rate = RateLimiter(frequency=rate_hz, warn=False)

    end_effector_task = mink.FrameTask(
        frame_name=config.MINK_EE_SITE,
        frame_type="site",
        position_cost=1.0,
        orientation_cost=0.3,
        lm_damping=float(getattr(config, "MINK_LM_DAMPING", 1e-4)),
    )
    posture_task = mink.PostureTask(model, cost=float(getattr(config, "MINK_POSTURE_COST", 1e-3)))
    tasks = [end_effector_task, posture_task]

    # posture target
    q_rest_full = configuration.q.copy()
    q_rest_full[:6] = q_zero
    posture_task.set_target(q_rest_full)

    return model, data, configuration, tasks, limits, solver, rate
