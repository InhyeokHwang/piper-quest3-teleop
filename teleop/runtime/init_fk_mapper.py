# teleop/runtime/init_fk_mapper.py
import numpy as np
from ..kinematics.piper_forward_kinematics import PiperForwardKinematics, DHType
from ..mapping.vr_mapper import VRToRobotMapper, VRMapperConfig

def init_fk_and_start():
    fk = PiperForwardKinematics(DHType.STANDARD) 
    q_zero = np.zeros(6, dtype=float) # [0, 0, 0, 0, 0, 0]

    #  compute_fk는 다음과 같은 행렬을 만듦
    #  [[ r11  r12  r13   px ]
    #   [ r21  r22  r23   py ]
    #   [ r31  r32  r33   pz ]
    #   [  0    0    0     1 ]]
    T_ee0 = fk.compute_fk(q_zero) # T_ee는 all zero 에 대한 
    EE_START = T_ee0[:3, 3].copy() # px, py, pz
    R_ee0 = T_ee0[:3, :3].copy() # [[r11 r12 r13] [r21 r22 r23] [r31 r32 r33]
    print("[INIT] EE_START from FK(q=0):", EE_START)
    return fk, q_zero, EE_START, R_ee0

def init_mapper(EE_START, R_ee0, debug: bool):
    mapper = VRToRobotMapper(
        ee_start_pos=EE_START,
        ee_start_rot=R_ee0,
        config=VRMapperConfig(position_scale=1.0),
        debug=debug,
    )
    return mapper
