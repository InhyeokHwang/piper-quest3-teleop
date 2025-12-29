# teleop/control/smoothing.py
import numpy as np
from .rotation_smoothing import PoseSmoothingParams, smooth_target_T
from .velocity_smoothing import VelocitySmoothingParams, smooth_velocity
from .. import config

def make_pose_params():
    # raw_pose가 있을 때 (원본 pose)
    # filtered_pose = previous_pose * (1- alpha) + raw_pose * alpha 로 
    return PoseSmoothingParams(
        alpha_pos=float(getattr(config, "EE_FILTER_ALPHA", 0.2)),   
        alpha_rot=float(getattr(config, "EE_ROT_FILTER_ALPHA", 0.1)),
        pos_deadband=float(getattr(config, "EE_POS_DEADBAND", 0.001)),
        rot_deadband_rad=float(getattr(config, "EE_ROT_DEADBAND_RAD", np.deg2rad(1.0))),
    )

def make_vel_params():
    return VelocitySmoothingParams(
        ema_beta=float(getattr(config, "IK_VEL_EMA_BETA", 0.3)),
        accel_max=float(getattr(config, "IK_ACCEL_MAX", 25.0)),
        vel_max=getattr(config, "IK_VEL_MAX", None),
    )

def smooth_target(T_filt, target_T, pose_params):
    if target_T is None:
        return T_filt
    if T_filt is None:
        return target_T.copy()
    return smooth_target_T(T_filt, target_T, pose_params)

def smooth_vel(vel, vel_filt, dt, vel_params):
    return smooth_velocity(vel, vel_filt, dt, vel_params)
