import numpy as np

def init_T_zero(EE_START, R_ee0):
    T_zero = np.eye(4, dtype=float)
    T_zero[:3, :3] = R_ee0
    T_zero[:3, 3] = np.asarray(EE_START, dtype=float).reshape(3,)
    return T_zero

def init_mujoco_state_zero(model, data): 
    for k in range(6):
        j = model.joint(f"joint{k+1}")
        adr = int(np.asarray(j.qposadr).item())
        data.qpos[adr] = 0.0
        vadr = int(np.asarray(j.dofadr).item())
        data.qvel[vadr] = 0.0

def init_gripper_indices(model):
    j7 = model.joint("joint7")
    j8 = model.joint("joint8")
    q_idx7 = int(np.asarray(j7.qposadr).item())
    q_idx8 = int(np.asarray(j8.qposadr).item())
    return q_idx7, q_idx8